from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List

API_KEY = "ak_utw1c0r9nw24vdcyinvnysrd"

EMAIL = "23f2002204@ds.study.iitm.ac.in"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Event(BaseModel):
    user: str
    amount: float
    ts: int


class RequestBody(BaseModel):
    events: List[Event]


@app.post("/analytics")
def analytics(body: RequestBody, x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        if x_api_key != API_KEY:
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

    revenue = 0.0
    totals = {}

    for e in body.events:
        if e.amount > 0:
            revenue += e.amount
            totals[e.user] = totals.get(e.user, 0) + e.amount

    top_user = max(totals, key=totals.get) if totals else ""

    return {
        "email": EMAIL,
        "total_events": len(body.events),
        "unique_users": len(set(e.user for e in body.events)),
        "revenue": revenue,
        "top_user": top_user,
    }
