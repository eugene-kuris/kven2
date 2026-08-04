from copy import deepcopy

TOOL_REQUEST_KEYS = (
    "tools",
    "tool_choice",
    "parallel_tool_calls",
)

PROJECT_READ_ROOT = "/opt/kven2"
DEFAULT_READ_FILE_CHARS = 12000
MAX_READ_FILE_CHARS = 50000
DEFAULT_WEB_SEARCH_RESULTS = 5
MAX_WEB_SEARCH_RESULTS = 10
DEFAULT_FETCH_URL_CHARS = 20000
MAX_FETCH_URL_CHARS = 50000

KVEN_TOOL_REGISTRY = {
    "get_time": {
        "enabled": True,
        "risk": "safe_readonly",
        "sandbox_method": "GET",
        "sandbox_path": "/time",
        "description": (
            "Return the real current server date, time, timezone, and "
            "weekday for answers that depend on the current temporal "
            "state."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    "read_file": {
        "enabled": True,
        "risk": "read_file",
        "sandbox_method": "GET",
        "sandbox_path": "/read_file",
        "description": "Read a text file from the Kven II project filesystem through agent_sandbox.py.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to a text file, usually under /opt/kven2.",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters to return. Default 12000, maximum 50000.",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    "web_search": {
        "enabled": True,
        "risk": "network_search",
        "sandbox_method": "GET",
        "sandbox_path": "/web_search",
        "description": "Search the public web through agent_sandbox.py and return compact title/url/snippet results.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query. Keep it concise and specific.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum search results to return. Default 5, maximum 10.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },    "fetch_url": {
        "enabled": True,
        "risk": "network_fetch_untrusted",
        "sandbox_method": "GET",
        "sandbox_path": "/fetch_url",
        "description": "Fetch one explicit http/https URL through agent_sandbox.py and return content as untrusted data.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Explicit http or https URL to fetch. Local/LAN URLs are allowed in personal lab mode.",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum response characters to return. Default 20000, maximum 50000.",
                },
            },
            "required": ["url"],
            "additionalProperties": False,
        },
    },

}


def export_openai_tools() -> list[dict]:
    """Export enabled Kven tools in OpenAI function format."""
    tools = []

    for name, definition in KVEN_TOOL_REGISTRY.items():
        if not definition.get("enabled", False):
            continue

        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": str(
                        definition.get(
                            "description",
                            "",
                        )
                    ),
                    "parameters": deepcopy(
                        definition.get(
                            "parameters",
                            {
                                "type": "object",
                                "properties": {},
                            },
                        )
                    ),
                },
            }
        )

    return tools
