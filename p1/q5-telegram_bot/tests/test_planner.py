from planner import make_plan


def test_plan_for_dataset_question() -> None:
    plan = make_plan("What is the mean of amount? https://example.org/data.csv")
    assert plan.needs_download and plan.needs_python
    assert not plan.needs_llm


def test_plan_for_conversation() -> None:
    assert make_plan("Explain regression simply").needs_llm
