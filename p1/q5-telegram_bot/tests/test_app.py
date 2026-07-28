from app import create_app


def test_routes_are_registered() -> None:
    paths = {route.path for route in create_app().routes}
    assert {"/", "/health", "/webhook", "/status", "/logs/{run_id}.jsonl"} <= paths
