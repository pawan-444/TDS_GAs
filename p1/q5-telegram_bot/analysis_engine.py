"""Deterministic dataframe analysis for frequent natural-language tasks."""
import re
from typing import Any

import pandas as pd


class AnalysisError(ValueError):
    pass


def _column(question: str, frame: pd.DataFrame) -> str:
    lower = question.lower()
    matches = [str(column) for column in frame.columns if str(column).lower() in lower]
    if matches:
        return matches[0]
    quoted = re.search(r"[\"']([^\"']+)[\"']", question)
    if quoted and quoted.group(1) in frame.columns:
        return quoted.group(1)
    raise AnalysisError(f"Specify a column. Available columns: {', '.join(map(str, frame.columns))}")


def _jsonable(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.where(pd.notna(value), None).to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.where(pd.notna(value), None).to_dict()
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def analyse(frame: pd.DataFrame, question: str) -> Any:
    lower = question.lower()
    if "column" in lower and ("list" in lower or "what" in lower):
        return list(map(str, frame.columns))
    if "how many rows" in lower or "row count" in lower or lower.strip() == "count":
        return int(len(frame))
    if "missing" in lower:
        return _jsonable(frame.isna().sum())
    group_match = re.search(r"group(?:ed)?\s+by\s+[\"']?([^\"'?,]+)[\"']?.*?(sum|mean|average|median|count)\s+(?:of\s+)?[\"']?([^\"'?,]+)", lower)
    if group_match:
        group_name, operation, value_name = (part.strip() for part in group_match.groups())
        columns = {str(item).lower(): str(item) for item in frame.columns}
        if group_name not in columns or value_name not in columns:
            raise AnalysisError("Group-by columns must match dataset column names")
        operation = {"average": "mean"}.get(operation, operation)
        grouped = frame.groupby(columns[group_name], dropna=False)[columns[value_name]].agg(operation).reset_index()
        return _jsonable(grouped)
    filter_match = re.search(r"(?:where|filter)\s+[\"']?([^\"'=]+)[\"']?\s*(?:=|equals)\s*[\"']?([^\"']+)[\"']?", question, re.IGNORECASE)
    if filter_match:
        column_name, expected = (part.strip() for part in filter_match.groups())
        columns = {str(item).lower(): str(item) for item in frame.columns}
        if column_name.lower() not in columns:
            raise AnalysisError("Filter column must match a dataset column name")
        return _jsonable(frame[frame[columns[column_name.lower()]].astype(str).str.casefold() == expected.casefold()])
    pivot_match = re.search(r"pivot.*?(?:index|rows)\s+[\"']?([^\"',]+).*?(?:columns?)\s+[\"']?([^\"',]+).*?(sum|mean|count).*?(?:of\s+)?[\"']?([^\"',]+)", lower)
    if pivot_match:
        index, columns_name, operation, values = (part.strip() for part in pivot_match.groups())
        names = {str(item).lower(): str(item) for item in frame.columns}
        if not {index, columns_name, values} <= names.keys():
            raise AnalysisError("Pivot fields must match dataset column names")
        table = pd.pivot_table(frame, index=names[index], columns=names[columns_name], values=names[values], aggfunc=operation, dropna=False).reset_index()
        return _jsonable(table)
    column = _column(question, frame)
    series = frame[column]
    operations = (("mean", series.mean), ("average", series.mean), ("median", series.median), ("sum", series.sum), ("standard deviation", series.std), ("std", series.std), ("mode", lambda: series.mode().tolist()), ("count", series.count))
    for label, fn in operations:
        if label in lower:
            return _jsonable(fn())
    n_match = re.search(r"(?:top|bottom)\s+(\d+)", lower)
    if n_match:
        n = int(n_match.group(1))
        numeric = pd.to_numeric(series, errors="raise")
        result = frame.loc[numeric.nlargest(n).index if "top" in lower else numeric.nsmallest(n).index]
        return _jsonable(result)
    if "unique" in lower or "categor" in lower:
        return _jsonable(series.value_counts(dropna=False))
    if "sort" in lower:
        return _jsonable(frame.sort_values(column, ascending="descending" not in lower))
    raise AnalysisError("I can calculate count, mean, median, mode, sum, std, missing values, unique values, sorting, and top/bottom values.")
