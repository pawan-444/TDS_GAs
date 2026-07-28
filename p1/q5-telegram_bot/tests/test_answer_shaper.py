from answer_shaper import decode_llm_answer, requested_answer_shape


def test_outer_contract_template_preserves_inner_shape() -> None:
    question = 'Reply only {"answer": {"state": "<state name>"}, "log_url": "<url>"}'
    assert requested_answer_shape(question, "Assam") == {"state": "Assam"}


def test_llm_json_is_unwrapped() -> None:
    assert decode_llm_answer('{"answer": {"values": [1, 2]}}') == {"values": [1, 2]}
