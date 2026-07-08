from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
import time
import uuid

EMAIL = "23f2002204@ds.study.iitm.ac.in"

app = FastAPI()

START = time.time()

http_requests_total = Counter("http_requests_total", "Total HTTP Requests")

logs = []


@app.middleware("http")
async def log_requests(request: Request, call_next):

    http_requests_total.inc()

    request_id = str(uuid.uuid4())

    response = await call_next(request)

    logs.append(
        {
            "level": "INFO",
            "ts": time.time(),
            "path": request.url.path,
            "request_id": request_id,
        }
    )

    if len(logs) > 100:
        logs.pop(0)

    return response


@app.get("/work")
def work(n: int):

    x = 0

    for _ in range(n):
        x += 1

    return {"email": EMAIL, "done": n}


@app.get("/healthz")
def health():

    return {"status": "ok", "uptime_s": time.time() - START}


@app.get("/metrics")
def metrics():

    return PlainTextResponse(generate_latest().decode(), media_type=CONTENT_TYPE_LATEST)


@app.get("/logs/tail")
def tail(limit: int = 10):

    return logs[-limit:]


@app.get("/")
def root():
    return {"status": "ok"}
