"""
Authenticated external gateway for Kven II.

Exposes only:
    GET  /v1/models
    POST /v1/chat/completions

Requests are authenticated with a Bearer token and proxied to the
existing Kven II API. The Authorization header is never forwarded
to the upstream service.
"""

import logging
import os
import secrets
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware


LOGGER = logging.getLogger("kven.client_gateway")

UPSTREAM_URL = os.getenv(
    "KVEN_UPSTREAM_URL",
    "http://127.0.0.1:10000",
).rstrip("/")

API_KEY = os.getenv("KVEN_CLIENT_API_KEY", "").strip()

MAX_REQUEST_BYTES = int(
    os.getenv(
        "KVEN_MAX_REQUEST_BYTES",
        str(32 * 1024 * 1024),
    )
)

CONNECT_TIMEOUT_SECONDS = float(
    os.getenv("KVEN_CONNECT_TIMEOUT_SECONDS", "10")
)

WRITE_TIMEOUT_SECONDS = float(
    os.getenv("KVEN_WRITE_TIMEOUT_SECONDS", "60")
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not API_KEY:
        raise RuntimeError(
            "KVEN_CLIENT_API_KEY is not configured"
        )

    timeout = httpx.Timeout(
        connect=CONNECT_TIMEOUT_SECONDS,
        read=None,
        write=WRITE_TIMEOUT_SECONDS,
        pool=10.0,
    )

    limits = httpx.Limits(
        max_connections=20,
        max_keepalive_connections=10,
    )

    app.state.http_client = httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        follow_redirects=False,
    )

    try:
        yield
    finally:
        await app.state.http_client.aclose()


app = FastAPI(
    title="Kven II Client Gateway",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
    max_age=86400,
)


def require_bearer_token(
    authorization: str | None = Header(default=None),
) -> None:
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    scheme, separator, supplied_token = authorization.partition(" ")

    if (
        not separator
        or scheme.lower() != "bearer"
        or not supplied_token
        or not secrets.compare_digest(supplied_token, API_KEY)
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


@app.exception_handler(HTTPException)
async def openai_http_exception_handler(
    request: Request,
    exc: HTTPException,
) -> JSONResponse:
    del request

    error_type = (
        "authentication_error"
        if exc.status_code == 401
        else "invalid_request_error"
    )

    return JSONResponse(
        status_code=exc.status_code,
        headers=exc.headers,
        content={
            "error": {
                "message": str(exc.detail),
                "type": error_type,
                "param": None,
                "code": None,
            }
        },
    )


def check_request_size(request: Request) -> None:
    content_length = request.headers.get("content-length")

    if content_length is None:
        return

    try:
        declared_size = int(content_length)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid Content-Length header",
        ) from exc

    if declared_size < 0:
        raise HTTPException(
            status_code=400,
            detail="Invalid Content-Length header",
        )

    if declared_size > MAX_REQUEST_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Request body is too large",
        )


def build_upstream_headers(request: Request) -> dict[str, str]:
    headers = {
        "accept": request.headers.get(
            "accept",
            "application/json",
        ),
        "accept-encoding": "identity",
    }

    content_type = request.headers.get("content-type")
    if content_type:
        headers["content-type"] = content_type

    user_agent = request.headers.get("user-agent")
    if user_agent:
        headers["user-agent"] = user_agent

    request_id = request.headers.get("x-request-id")
    if request_id:
        headers["x-request-id"] = request_id

    return headers


def build_client_headers(
    upstream_response: httpx.Response,
) -> dict[str, str]:
    allowed_response_headers = (
        "content-type",
        "cache-control",
        "x-request-id",
        "retry-after",
    )

    headers: dict[str, str] = {}

    for name in allowed_response_headers:
        value = upstream_response.headers.get(name)
        if value:
            headers[name] = value

    return headers


async def iter_upstream(
    upstream_response: httpx.Response,
) -> AsyncIterator[bytes]:
    try:
        async for chunk in upstream_response.aiter_raw():
            yield chunk
    finally:
        await upstream_response.aclose()


async def proxy_request(
    request: Request,
    upstream_path: str,
) -> Response:
    check_request_size(request)

    body = await request.body()

    if len(body) > MAX_REQUEST_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Request body is too large",
        )

    upstream_request = request.app.state.http_client.build_request(
        method=request.method,
        url=f"{UPSTREAM_URL}{upstream_path}",
        params=request.query_params,
        headers=build_upstream_headers(request),
        content=body,
    )

    try:
        upstream_response = await request.app.state.http_client.send(
            upstream_request,
            stream=True,
        )
    except httpx.ConnectError:
        LOGGER.warning("Unable to connect to Kven II upstream")
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": "Kven II upstream is unavailable",
                    "type": "server_error",
                    "param": None,
                    "code": None,
                }
            },
        )
    except httpx.TimeoutException:
        LOGGER.warning("Timeout while connecting to Kven II upstream")
        return JSONResponse(
            status_code=504,
            content={
                "error": {
                    "message": "Kven II upstream timed out",
                    "type": "server_error",
                    "param": None,
                    "code": None,
                }
            },
        )
    except httpx.HTTPError:
        LOGGER.exception("HTTP error while contacting Kven II upstream")
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": "Kven II upstream request failed",
                    "type": "server_error",
                    "param": None,
                    "code": None,
                }
            },
        )

    response_headers = build_client_headers(upstream_response)

    content_type = upstream_response.headers.get(
        "content-type",
        "",
    )
    media_type = content_type.split(";", 1)[0].strip().lower()

    if media_type == "text/event-stream":
        response_headers.setdefault(
            "cache-control",
            "no-cache",
        )
        response_headers["x-accel-buffering"] = "no"

        return StreamingResponse(
            iter_upstream(upstream_response),
            status_code=upstream_response.status_code,
            headers=response_headers,
        )

    try:
        response_body = await upstream_response.aread()
    finally:
        await upstream_response.aclose()

    return Response(
        content=response_body,
        status_code=upstream_response.status_code,
        headers=response_headers,
    )


@app.get(
    "/v1/models",
    dependencies=[Depends(require_bearer_token)],
)
async def models(request: Request) -> Response:
    return await proxy_request(
        request=request,
        upstream_path="/v1/models",
    )


@app.post(
    "/v1/chat/completions",
    dependencies=[Depends(require_bearer_token)],
)
async def chat_completions(request: Request) -> Response:
    return await proxy_request(
        request=request,
        upstream_path="/v1/chat/completions",
    )
