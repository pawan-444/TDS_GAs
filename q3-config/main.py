import os
import yaml
from dotenv import load_dotenv

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULTS = {
    "port": 8000,
    "workers": 1,
    "debug": False,
    "log_level": "info",
    "api_key": "default-secret-000",
}


def to_bool(value):
    return str(value).strip().lower() in (
        "true",
        "1",
        "yes",
        "on",
    )


@app.get("/effective-config")
def effective_config(set: list[str] = Query(default=[])):
    cfg = DEFAULTS.copy()

    # YAML
    with open("config.development.yaml") as f:
        cfg.update(yaml.safe_load(f))

    # .env
    if os.getenv("NUM_WORKERS"):
        cfg["workers"] = int(os.getenv("NUM_WORKERS"))

    if os.getenv("APP_API_KEY"):
        cfg["api_key"] = os.getenv("APP_API_KEY")

    # OS ENV
    mapping = {
        "APP_PORT": ("port", int),
        "APP_WORKERS": ("workers", int),
        "APP_DEBUG": ("debug", to_bool),
        "APP_LOG_LEVEL": ("log_level", str),
        "APP_API_KEY": ("api_key", str),
    }

    for env_name, (key, caster) in mapping.items():
        if env_name in os.environ:
            cfg[key] = caster(os.environ[env_name])

    # CLI overrides
    for item in set:
        if "=" not in item:
            continue

        key, value = item.split("=", 1)

        if key in ("port", "workers"):
            cfg[key] = int(value)

        elif key == "debug":
            cfg[key] = to_bool(value)

        else:
            cfg[key] = value

    # mask secret
    cfg["api_key"] = "****"

    return cfg


@app.get("/")
def root():
    return {"status": "ok"}
