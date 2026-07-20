from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime, timezone
import json

LOG = "/tmp/owui_tool_probe.log"

app = FastAPI(
    title="OWUI Tool Probe",
    description="OpenAPI tool server probe for Open WebUI tool-call experiments.",
    version="0.2.0",
    servers=[
        {"url": "http://192.168.143.192:8966"},
        {"url": "http://172.17.0.1:8966"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EchoIn(BaseModel):
    text: str = Field(..., description="Sentinel text to echo back.")

def write_log(event: str, data: dict):
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **data,
    }
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

@app.middleware("http")
async def log_request_without_body(request: Request, call_next):
    write_log("http_request", {
        "method": request.method,
        "url": str(request.url),
        "path": request.url.path,
        "headers": {
            k: v for k, v in request.headers.items()
            if k.lower() in ("authorization", "content-type", "user-agent", "host")
        },
    })
    return await call_next(request)

@app.get("/", operation_id="owui_probe_root")
async def root():
    return {"ok": True, "service": "owui_tool_probe", "openapi": "/openapi.json"}

@app.get("/health", operation_id="owui_probe_health")
async def health():
    return {"ok": True, "service": "owui_tool_probe"}

@app.get(
    "/echo",
    operation_id="owui_probe_echo",
    summary="Echo a sentinel string through the OWUI external tool path.",
    description="Returns probe_result containing OWUI_TOOL_RESULT plus the provided text.",
)
async def echo(text: str = Query(..., description="Sentinel text to echo back.")):
    result = {
        "probe_result": f"OWUI_TOOL_RESULT::{text}",
        "server_time_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_log("echo_call", {"text": text, "result": result})
    return result
