import pandas as pd

from analysis_engine import analyse


def test_mean_and_top_values() -> None:
    frame = pd.DataFrame({"amount": [3, 1, 9], "kind": ["a", "b", "a"]})
    assert analyse(frame, "mean of amount") == 13 / 3
    assert analyse(frame, "top 2 amount") == [{"amount": 9, "kind": "a"}, {"amount": 3, "kind": "a"}]


def test_missing_values() -> None:
    frame = pd.DataFrame({"amount": [1, None]})
    assert analyse(frame, "show missing values") == {"amount": 1}
