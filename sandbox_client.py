import asyncio
import json
import logging

import httpx

import kven2_time as sys_time

logger = logging.getLogger(__name__)

SANDBOX_BASE_URL = "http://127.0.0.1:8954"
DEFAULT_READ_FILE_CHARS = 12000
DEFAULT_WEB_SEARCH_RESULTS = 5
MAX_WEB_SEARCH_RESULTS = 10
DEFAULT_FETCH_URL_CHARS = 20000
MAX_FETCH_URL_CHARS = 50000


def _json_preview(value, limit: int = 800) -> str:
    """Compact bounded JSON preview for logs; never raises."""
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except Exception:
        text = repr(value)
    if len(text) > limit:
        return text[:limit] + "...[truncated]"
    return text


async def execute_get_time_tool() -> dict:
    """Execute get_time through the existing time adapter."""
    started = asyncio.get_running_loop().time()
    try:
        raw_time = await sys_time.get_external_time()
        duration_ms = int((asyncio.get_running_loop().time() - started) * 1000)

        parsed = None
        if isinstance(raw_time, str):
            try:
                parsed = json.loads(raw_time)
            except Exception:
                parsed = None

        result = parsed if isinstance(parsed, dict) else {"time": raw_time}
        tool_result = {
            "tool_name": "get_time",
            "status": "ok",
            "duration_ms": duration_ms,
            "source": "kven2_time.get_external_time",
            "result": result,
        }
        logger.info(
            "[TOOLS_EXEC] executed tool=get_time status=ok duration_ms=%s result=%s",
            duration_ms,
            _json_preview(result, limit=500),
        )
        return tool_result
    except Exception as exc:
        duration_ms = int((asyncio.get_running_loop().time() - started) * 1000)
        tool_result = {
            "tool_name": "get_time",
            "status": "error",
            "duration_ms": duration_ms,
            "source": "kven2_time.get_external_time",
            "error": str(exc),
        }
        logger.error("[TOOLS_EXEC] tool=get_time failed: %s", exc, exc_info=True)
        return tool_result


async def execute_read_file_tool(arguments: dict) -> dict:
    """Execute read_file through agent_sandbox.py /read_file."""
    started = asyncio.get_running_loop().time()
    path = arguments.get("path")
    max_chars = int(arguments.get("max_chars", DEFAULT_READ_FILE_CHARS))

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(
                f"{SANDBOX_BASE_URL}/read_file",
                params={"path": path, "max_chars": max_chars},
            )
            duration_ms = int((asyncio.get_running_loop().time() - started) * 1000)
            try:
                result = response.json()
            except Exception:
                result = {"raw_response": response.text}

        status = "ok" if response.status_code == 200 else "error"
        tool_result = {
            "tool_name": "read_file",
            "status": status,
            "duration_ms": duration_ms,
            "source": "agent_sandbox.py:/read_file",
            "http_status": response.status_code,
            "arguments": {"path": path, "max_chars": max_chars},
            "result": result,
        }
        logger.info(
            "[TOOLS_EXEC] executed tool=read_file status=%s duration_ms=%s path=%s http_status=%s chars=%s truncated=%s",
            status,
            duration_ms,
            path,
            response.status_code,
            result.get("chars") if isinstance(result, dict) else None,
            result.get("truncated") if isinstance(result, dict) else None,
        )
        return tool_result
    except Exception as exc:
        duration_ms = int((asyncio.get_running_loop().time() - started) * 1000)
        tool_result = {
            "tool_name": "read_file",
            "status": "error",
            "duration_ms": duration_ms,
            "source": "agent_sandbox.py:/read_file",
            "arguments": {"path": path, "max_chars": max_chars},
            "error": str(exc),
        }
        logger.error("[TOOLS_EXEC] tool=read_file failed path=%s: %s", path, exc, exc_info=True)
        return tool_result


async def execute_web_search_tool(arguments: dict) -> dict:
    """Execute web_search through agent_sandbox.py /web_search."""
    started = asyncio.get_running_loop().time()
    query = str(arguments.get("query", "")).strip()
    max_results = arguments.get("max_results", DEFAULT_WEB_SEARCH_RESULTS)
    try:
        max_results = int(max_results)
    except Exception:
        max_results = DEFAULT_WEB_SEARCH_RESULTS
    max_results = max(1, min(max_results, MAX_WEB_SEARCH_RESULTS))

    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            response = await client.get(
                f"{SANDBOX_BASE_URL}/web_search",
                params={"query": query, "max_results": max_results},
            )
            duration_ms = int((asyncio.get_running_loop().time() - started) * 1000)
            try:
                result = response.json()
            except Exception:
                result = {"raw_response": response.text}

        status = "ok" if response.status_code == 200 else "error"
        tool_result = {
            "tool_name": "web_search",
            "status": status,
            "duration_ms": duration_ms,
            "source": "agent_sandbox.py:/web_search",
            "http_status": response.status_code,
            "arguments": {"query": query, "max_results": max_results},
            "result": result,
        }
        logger.info(
            "[TOOLS_EXEC] executed tool=web_search status=%s duration_ms=%s query=%r http_status=%s results_count=%s",
            status,
            duration_ms,
            query,
            response.status_code,
            result.get("results_count") if isinstance(result, dict) else None,
        )
        return tool_result
    except Exception as exc:
        duration_ms = int((asyncio.get_running_loop().time() - started) * 1000)
        tool_result = {
            "tool_name": "web_search",
            "status": "error",
            "duration_ms": duration_ms,
            "source": "agent_sandbox.py:/web_search",
            "arguments": {"query": query, "max_results": max_results},
            "error": str(exc),
        }
        logger.error("[TOOLS_EXEC] tool=web_search failed query=%r: %s", query, exc, exc_info=True)
        return tool_result


async def execute_fetch_url_tool(arguments: dict) -> dict:
    """Execute fetch_url through agent_sandbox.py /fetch_url."""
    started = asyncio.get_running_loop().time()
    url = str(arguments.get("url", "")).strip()
    max_chars = arguments.get("max_chars", DEFAULT_FETCH_URL_CHARS)
    try:
        max_chars = int(max_chars)
    except Exception:
        max_chars = DEFAULT_FETCH_URL_CHARS
    max_chars = max(1, min(max_chars, MAX_FETCH_URL_CHARS))

    try:
        async with httpx.AsyncClient(timeout=75.0) as client:
            response = await client.get(
                f"{SANDBOX_BASE_URL}/fetch_url",
                params={"url": url, "max_chars": max_chars},
            )
            duration_ms = int((asyncio.get_running_loop().time() - started) * 1000)
            try:
                result = response.json()
            except Exception:
                result = {"raw_response": response.text}

        status = "ok" if response.status_code == 200 else "error"
        tool_result = {
            "tool_name": "fetch_url",
            "status": status,
            "duration_ms": duration_ms,
            "source": "agent_sandbox.py:/fetch_url",
            "http_status": response.status_code,
            "arguments": {"url": url, "max_chars": max_chars},
            "result": result,
        }
        logger.info(
            "[TOOLS_EXEC] executed tool=fetch_url status=%s duration_ms=%s url=%r http_status=%s chars=%s truncated=%s",
            status,
            duration_ms,
            url,
            response.status_code,
            result.get("chars") if isinstance(result, dict) else None,
            result.get("truncated") if isinstance(result, dict) else None,
        )
        return tool_result
    except Exception as exc:
        duration_ms = int((asyncio.get_running_loop().time() - started) * 1000)
        tool_result = {
            "tool_name": "fetch_url",
            "status": "error",
            "duration_ms": duration_ms,
            "source": "agent_sandbox.py:/fetch_url",
            "arguments": {"url": url, "max_chars": max_chars},
            "error": str(exc),
        }
        logger.error("[TOOLS_EXEC] tool=fetch_url failed url=%r: %s", url, exc, exc_info=True)
        return tool_result


async def execute_gateway_tool(requested_call: dict) -> dict:
    """Dispatch one validated gateway tool call."""
    name = requested_call.get("name")
    arguments = requested_call.get("arguments", {})

    if name == "get_time":
        return await execute_get_time_tool()

    if name == "read_file":
        return await execute_read_file_tool(arguments)

    if name == "web_search":
        return await execute_web_search_tool(arguments)

    if name == "fetch_url":
        return await execute_fetch_url_tool(arguments)

    logger.warning("[TOOLS_EXEC] refused unsupported tool=%s", name)
    return {
        "tool_name": name,
        "status": "error",
        "duration_ms": 0,
        "error": f"Unsupported tool: {name}",
    }
