from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
import time

EMAIL = "23f2002204@ds.study.iitm.ac.in"

ALLOWED_ORIGINS = [
    "https://app-c1bw0x.example.com",
]

# Exam page bhi allow karni hai
ALLOWED_ORIGIN_REGEX = r"https://.*"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=ALLOWED_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limit: 12 requests / 10 seconds
RATE_LIMIT = 12
WINDOW = 10

clients = {}


@app.middleware("http")
async def request_context_and_rate_limit(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        request_id = str(uuid.uuid4())

    request.state.request_id = request_id

    client = request.headers.get("X-Client-Id", "anonymous")
    now = time.time()

    history = clients.get(client, [])
    history = [t for t in history if now - t < WINDOW]

    if len(history) >= RATE_LIMIT:
        response = JSONResponse(
            status_code=429, content={"detail": "Rate limit exceeded"}
        )
        response.headers["X-Request-ID"] = request_id
        return response

    history.append(now)
    clients[client] = history

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/ping")
def ping(request: Request):
    return {"email": EMAIL, "request_id": request.state.request_id}


@app.get("/")
def root():
    return {"status": "ok"}
