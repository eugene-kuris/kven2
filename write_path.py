# /opt/kven2/write_path.py

import re
import logging
import asyncio

from small_model import call_small_model
from sqlite import insert_memory
from consolidation import consolidate

logger = logging.getLogger("Kven.Memory")

WRITE_PATH_VERSION = "strict-source-grounding-single-fact-2026-07-16-v3"

# Conservative defaults: memory quality is more important than quantity.
MAX_DIALOGUE_CHARS = 3000
MAX_EXTRACTED_ITEMS = 2
MAX_MEMORY_ITEM_CHARS = 1500

EXTRACT_PROMPT = """
SYSTEM: You are a strict long-term memory archivist for an LLM agent.

TASK:
Extract only durable, useful memory entries from the USER-SUPPLIED SOURCE TEXT.

CORE RULE:
A question, request, command, or conversation event is not a durable fact.
Do not describe what the user asked the assistant to do.

SOURCE POLICY:
- Use only facts, decisions, constraints, corrections, explicit preferences, and
  confirmed observations stated directly by the user in SOURCE TEXT.
- Preserve identifiers, versions, paths, hostnames, quantities, and model names
  exactly as written.
- If the user explicitly says "remember", "запомни", "зафиксируй",
  "рабочий факт", or "baseline", prioritize the stated fact.
- Do not infer purpose, urgency, expertise, access, possession, expectations,
  workflow, intentions, or preferences unless explicitly stated.
- Do not extract information merely because the user asked about it.
- Do not extract assistant suggestions, assistant answers, tool output,
  retrieved context, implementation options, or future possibilities.
- Do not store temporary UI/chat artifacts, smoke tests, or the wording of a query.
- Prefer zero entries over a weak or inferred entry.
- Prefer one precise entry over several paraphrases.
- Write in the same language as SOURCE TEXT whenever possible.

NEVER OUTPUT ENTRIES LIKE:
- User requested ...
- User asked ...
- User is asking ...
- User expects ...
- User needs ...
- User wants ...
- User has access to ...
- User has knowledge of ...
- ... is needed for identification purposes
- ... is a baseline requirement
- ... is required for document analysis

OUTPUT RULES:
- Output ONLY lines beginning with FACT: or MEMORY:
- One memory entry per line
- No markdown
- No code blocks
- No explanations
- No numbering
- No JSON
- If nothing durable and useful exists, output nothing
- Maximum 2 entries

GOOD EXAMPLES:
FACT: Current Kven II baseline uses OWUI 0.9.6
FACT: HP ProLiant DL380p Gen8 is a 2U server
MEMORY: Пользователь явно предпочитает небольшие обратимые изменения кода

BAD EXAMPLES:
FACT: User requested server serial number
MEMORY: User expects an immediate response
FACT: Server serial number is needed for identification purposes
FACT: User has knowledge of server hardware details
MEMORY: User prefers direct responses to technical queries
FACT: Research container code extraction is required for document analysis

USER-SUPPLIED SOURCE TEXT:
{dialogue}

Output:
"""


# These prefixes usually describe the current conversation instead of durable state.
# Explicitly stated preferences are handled separately by source-aware checks.
BLOCKED_PATTERNS = [
    # Questions, requests, expectations and inferred user state.
    r"^(the\s+)?user\s+(requested|requests|asked|asks|is\s+asking)\b",
    r"^(the\s+)?user\s+(expects?|needs?|wants?)\b",
    r"^(the\s+)?user\s+has\s+(access|knowledge|awareness)\b",
    r"^(the\s+)?user\s+is\s+(considering|thinking|thinking\s+about|evaluating|exploring|focused\s+on|working\s+on|asking)\b",
    r"^(the\s+)?user\s+is\s+planning\b",
    r"^пользователь\s+(просит|попросил|спрашивает|спросил|ожидает|нуждается|хочет)\b",
    r"^пользователь\s+(имеет\s+доступ|обладает\s+знаниями|знает)\b",
    r"^пользователь\s+(рассматривает|думает|обдумывает|оценивает|изучает|планирует)\b",

    # Unsupported inferred purpose / requirement.
    r"\bis\s+needed\s+for\s+identification\s+purposes\b",
    r"\bis\s+required\s+for\s+document\s+analysis\b",
    r"\bis\s+a\s+baseline\s+requirement\b",
    r"\bneeded\s+for\s+identification\b",
    r"\brequired\s+for\s+document\b",
    r"\bнужен\s+для\s+идентификац",
    r"\bтребуется\s+для\s+анализа\s+документ",

    # Assistant self-description / offers / current-response artifacts.
    r"\boptions?\s+proposed\b",
    r"\bimplementation\s+options?\b",
    r"\bwill\s+assist\b",
    r"\bready\s+to\s+(provide|help|assist|prepare)\b",
    r"\bcan\s+(provide|prepare|help|assist)\b",
    r"\bassistant\s+(will|can|is\s+ready)\b",
    r"\bKven\s+II\s+(will|can|is\s+ready)\b",
    r"\bthree\s+implementation\s+options\b",
    r"\bчем\s+могу\s+помочь\b",
    r"\bготов\s+(помочь|подготовить|предоставить)\b",

    # Meta-memory artifacts.
    r"\brequested\s+to\s+remember\b",
    r"\buser\s+requested\s+to\s+remember\b",
    r"\bпопросил\s+запомнить\b",

    # Ephemeral states that should not become invariants.
    r"semantic\s+and\s+episodic\s+memory.*\b(currently\s+)?empty\b",
    r"семантическ.*эпизодическ.*памят.*\bпуст",
]


MEMORY_QUESTION_MARKERS = [
    "что ты помнишь",
    "что помнишь",
    "что я говорил",
    "что мы решили",
    "что мы зафиксировали",
    "что известно",
    "вспомни",
    "напомни",
    "расскажи что помнишь",
    "what do you remember",
    "what we decided",
    "what did we decide",
]


# Exact/near-exact smoke tests only. Do not use broad substrings such as
# "работает": a real observation may legitimately contain that word.
TECHNICAL_TEST_PATTERNS = [
    r"^(test|тест|ping|pong)\s*[.!?]*$",
    r"^(проверка\s+(работы|связи)|smoke\s*test)\s*[.!?]*$",
    r"^(ответь\s+только\s+одним\s+словом|одним\s+словом)\b.{0,60}$",
    r"^(работает\??|does\s+it\s+work\??)\s*$",
]


EXPLICIT_MEMORY_MARKERS = [
    "запомни",
    "запомнить",
    "зафиксируй",
    "зафиксировать",
    "фиксируем",
    "рабочий факт",
    "baseline",
    "точка отсчёта",
    "remember that",
    "store this",
    "record this",
]


CORRECTION_MARKERS = [
    "здесь ошибка",
    "это ошибка",
    "исправление",
    "исправь",
    "уточнение",
    "уточню",
    "на самом деле",
    "верное значение",
    "правильно:",
    "correction",
    "actually",
    "the correct",
]


EXPLICIT_PREFERENCE_MARKERS = [
    "я предпочитаю",
    "мне нравится",
    "мне удобнее",
    "для меня важно",
    "я хочу, чтобы",
    "не делай",
    "избегай",
    "предпочитаю",
    "i prefer",
    "i like",
    "i want you to",
    "please avoid",
    "do not",
]


QUERY_OR_COMMAND_PREFIXES = [
    # Russian.
    "назови", "скажи", "расскажи", "покажи", "найди", "объясни",
    "подскажи", "проверь", "сравни", "посчитай", "выведи", "дай",
    "можешь", "можно ли", "нужно ли", "стоит ли",
    "какой", "какая", "какие", "какое", "что", "где", "когда",
    "почему", "зачем", "сколько", "кто", "как",
    # English.
    "tell me", "name", "show", "find", "explain", "check", "compare",
    "calculate", "give me", "can you", "could you", "would you",
    "what", "where", "when", "why", "how", "who", "should",
]


def _contains_any(text: str, markers: list[str]) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in markers)


def strip_reasoning(text: str) -> str:
    """
    Удаляет reasoning-блоки различных моделей.
    """

    patterns = [
        r'<\|think_start\|>.*?<\|think_end\|>',
        r'<think>.*?</think>',
    ]

    cleaned = text or ""

    for pattern in patterns:
        cleaned = re.sub(
            pattern,
            '',
            cleaned,
            flags=re.DOTALL | re.IGNORECASE
        )

    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

    return cleaned.strip()


def _strip_external_context_blocks(text: str) -> str:
    """
    Defensive cleanup: write_path must not learn from OWUI/Kven retrieval wrappers
    if an unsanitized route calls it directly.
    """

    cleaned = text or ""

    block_patterns = [
        r"<memory_context\b[^>]*>.*?</memory_context>",
        r"<context\b[^>]*>.*?</context>",
        r"<source\b[^>]*>.*?</source>",
    ]

    for pattern in block_patterns:
        cleaned = re.sub(
            pattern,
            "",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )

    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def _content_to_text(content) -> str:
    """
    Normalizes OpenAI string or multimodal content arrays to plain text.
    """

    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                text_parts.append(part)
        content = "\n".join(text_parts)

    return _strip_external_context_blocks(
        strip_reasoning(str(content or ""))
    ).strip()


def _extract_last_user_text(messages: list) -> str:
    """
    Возвращает чистый текст последнего user-сообщения.
    Используется для deterministic guard перед вызовом малой модели.
    """

    if not messages:
        return ""

    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") != "user":
            continue
        return _content_to_text(messages[idx].get("content", ""))

    return ""


def _split_user_segments(text: str) -> list[str]:
    """
    Splits a mixed user turn into statement/question-sized segments.
    """

    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"(?<=[.!?])\s+|\n+", normalized)
    return [part.strip() for part in parts if part and part.strip()]


def _is_query_or_command_segment(segment: str) -> bool:
    """
    Returns True for a segment that is primarily a question or instruction.
    """

    value = re.sub(r"^[\-\*•\d.)\s]+", "", (segment or "").strip())
    lowered = value.lower()

    if not lowered:
        return True

    if value.endswith("?"):
        return True

    for prefix in QUERY_OR_COMMAND_PREFIXES:
        if lowered == prefix or lowered.startswith(prefix + " "):
            return True
        if lowered.startswith(prefix + ":"):
            return True

    return False


def _extract_durable_source_text(user_text: str) -> str:
    """
    Keeps only user-authored declarative material.

    For mixed turns such as "У нас есть две T4. Можно ли их добавить?",
    only the first declarative sentence is sent to the extractor.
    """

    raw = _strip_external_context_blocks(
        strip_reasoning(user_text or "")
    ).strip()

    if not raw:
        return ""

    explicit_memory_request = _contains_any(raw, EXPLICIT_MEMORY_MARKERS)
    explicit_correction = _contains_any(raw, CORRECTION_MARKERS)

    # Explicit remember/correction turns may use an imperative wrapper around the
    # fact, so preserve the full user text.
    if explicit_memory_request or explicit_correction:
        return raw[-MAX_DIALOGUE_CHARS:]

    kept = [
        segment
        for segment in _split_user_segments(raw)
        if not _is_query_or_command_segment(segment)
    ]

    durable_text = "\n".join(kept).strip()

    if len(durable_text) > MAX_DIALOGUE_CHARS:
        durable_text = durable_text[-MAX_DIALOGUE_CHARS:]

    return durable_text


def should_skip_memory_extraction(user_text: str) -> tuple[bool, str]:
    """
    Deterministic memory quality guard.

    It intentionally skips pure questions/commands. Questions can retrieve
    memory, but they do not create new durable memory.
    """

    raw = _strip_external_context_blocks(
        strip_reasoning(user_text or "")
    ).strip()
    q = raw.lower()

    if not q:
        return True, "empty_user_text"

    # OWUI internal prompts should already be filtered in routes.py, but keep a
    # second defensive line here.
    if raw.startswith("### Task:"):
        return True, "owui_internal_task"

    explicit_memory_request = _contains_any(q, EXPLICIT_MEMORY_MARKERS)
    explicit_correction = _contains_any(q, CORRECTION_MARKERS)

    if not explicit_memory_request and not explicit_correction:
        if "?" in q and any(marker in q for marker in MEMORY_QUESTION_MARKERS):
            return True, "memory_question_no_new_fact"

        if any(q.startswith(marker) for marker in MEMORY_QUESTION_MARKERS):
            return True, "memory_question_no_new_fact"

    for pattern in TECHNICAL_TEST_PATTERNS:
        if re.search(pattern, q, flags=re.IGNORECASE):
            return True, "technical_test_no_memory"

    durable_source = _extract_durable_source_text(raw)

    if not durable_source:
        return True, "question_or_command_no_new_fact"

    if len(durable_source.strip()) < 10:
        return True, "durable_source_too_short"

    return False, "extract"


def _normalize_content(content: str) -> str:
    """
    Нормализует одну строку памяти перед фильтрацией и сохранением.
    """

    content = (content or "").strip()

    content = re.sub(
        r'```.*?```',
        '',
        content,
        flags=re.DOTALL
    ).strip()

    content = re.sub(r'^[\-\*•\s]+', '', content).strip()
    content = re.sub(r'^\d+[\.)]\s*', '', content).strip()
    content = re.sub(r'\s+', ' ', content).strip()
    content = content.strip('"\'`“”«»')

    return content


def _source_has_explicit_preference(source_text: str) -> bool:
    return _contains_any(source_text, EXPLICIT_PREFERENCE_MARKERS)


def _is_blocked_memory(content: str, source_text: str = "") -> bool:
    """
    Rejects weak, inferred, conversation-meta and unsupported memories.
    """

    lowered = content.strip().lower()

    if not lowered:
        return True

    if len(lowered) < 8:
        return True

    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            return True

    # Any extracted preference is allowed only when the current user-authored
    # source explicitly states a preference. This catches variants such as
    # "Пользователь явно предпочитает ..." as well as plain "User prefers ...".
    preference_claim = re.search(
        r"^(the\s+)?user\s+(explicitly\s+|clearly\s+|strongly\s+)?prefers?\b"
        r"|^пользователь\s+(явно\s+|однозначно\s+|обычно\s+|в\s+целом\s+)?предпочитает\b"
        r"|\buser\s+preference\b|\bпредпочтени[ея]\s+пользователя\b",
        lowered,
        flags=re.IGNORECASE,
    )
    if preference_claim and not _source_has_explicit_preference(source_text):
        return True

    weak_phrases = (
        "next steps",
        "further steps",
        "development tasks",
        "complex problem solving",
        "chat history",
        "concise title",
        "emoji",
        "immediate response",
        "direct responses to technical queries",
        "hardware-specific information",
        "knowledge of server hardware",
        "access to server serial number",
        "document-based container code",
        "container code identification",
    )

    if any(phrase in lowered for phrase in weak_phrases):
        return True

    return False


def _similarity_tokens(text: str) -> set[str]:
    """
    Lightweight language-agnostic token set for within-response deduplication.
    """

    return {
        token
        for token in re.findall(r"[\w./:@+-]+", (text or "").lower(), re.UNICODE)
        if len(token) >= 3
    }


def _is_near_duplicate(content: str, accepted_contents: list[str]) -> bool:
    candidate = _similarity_tokens(content)

    if not candidate:
        return False

    for existing in accepted_contents:
        other = _similarity_tokens(existing)
        if not other:
            continue

        union = candidate | other
        if not union:
            continue

        jaccard = len(candidate & other) / len(union)
        containment = len(candidate & other) / min(len(candidate), len(other))

        if jaccard >= 0.72 or containment >= 0.85:
            return True

    return False


def _extract_source_anchors(text: str) -> set[str]:
    """
    Extracts identifiers/numbers/technical literals that must not be invented.
    """

    anchors = set()

    for token in re.findall(r"[A-Za-zА-Яа-я0-9][A-Za-zА-Яа-я0-9_.:/+-]*", text or ""):
        token = token.strip(".,;!?()[]{}<>\\\"'«»")
        if not token:
            continue

        normalized = token.lower()

        if (
            any(ch.isdigit() for ch in token)
            or "/" in token
            or ":" in token
            or "." in token
            or "_" in token
            or "-" in token
            or (len(token) >= 3 and token.isupper())
        ):
            anchors.add(normalized)

    return anchors


def _has_invented_anchors(content: str, source_text: str) -> bool:
    """
    Rejects an extracted item if it introduces identifiers, numbers, paths or
    versions absent from the user-authored source.
    """

    content_anchors = _extract_source_anchors(content)
    source_anchors = _extract_source_anchors(source_text)

    if not content_anchors:
        return False

    # Common grammatical/count words are not durable identifiers.
    harmless = {
        "one", "two", "three", "first", "second",
        "один", "два", "три", "первый", "второй",
    }

    unsupported = {
        anchor
        for anchor in content_anchors
        if anchor not in source_anchors and anchor not in harmless
    }

    return bool(unsupported)


LEXICAL_STOPWORDS = {
    # Russian function words and extraction wrappers.
    "это", "этот", "эта", "эти", "того", "для", "или", "как", "что",
    "при", "без", "над", "под", "про", "из", "от", "до", "по", "на",
    "в", "во", "и", "а", "но", "же", "бы", "ли", "у", "к", "с",
    "со", "не", "да", "нет", "запомни", "запомнить", "зафиксируй",
    "пользователь", "явно", "факт", "memory",
    # English function words and extraction wrappers.
    "the", "a", "an", "is", "are", "was", "were", "be", "to", "of",
    "for", "in", "on", "at", "by", "with", "and", "or", "but", "not",
    "this", "that", "these", "those", "remember", "record", "store",
    "user", "explicitly", "fact",
}


def _rough_stem_token(token: str) -> str:
    """Small deterministic stemmer for grounding checks, not search/ranking."""
    value = (token or "").lower().strip(".,;!?()[]{}<>\\\"'«»`")
    if len(value) < 4:
        return value

    # Preserve technical identifiers exactly.
    if any(ch.isdigit() for ch in value) or any(ch in value for ch in "_./:+-"):
        return value

    russian_suffixes = (
        "иями", "ями", "ами", "ого", "ему", "ому", "ыми", "ими",
        "иях", "ах", "ях", "ий", "ый", "ая", "ое", "ые", "ых",
        "ую", "юю", "ом", "ем", "ов", "ев", "ам", "ям",
        "а", "я", "ы", "и", "у", "ю", "е", "о",
    )
    english_suffixes = ("ingly", "edly", "ing", "ed", "es", "s")

    for suffix in russian_suffixes + english_suffixes:
        if value.endswith(suffix) and len(value) - len(suffix) >= 4:
            return value[:-len(suffix)]

    return value


def _lexical_grounding_tokens(text: str) -> set[str]:
    tokens = set()
    for raw in re.findall(r"[A-Za-zА-Яа-яЁё0-9][A-Za-zА-Яа-яЁё0-9_.:/+-]*", text or ""):
        lowered = raw.lower().strip(".,;!?()[]{}<>\\\"'«»`")
        if not lowered or lowered in LEXICAL_STOPWORDS:
            continue
        stem = _rough_stem_token(lowered)
        if len(stem) >= 3 and stem not in LEXICAL_STOPWORDS:
            tokens.add(stem)
    return tokens


def _is_lexically_grounded(content: str, source_text: str) -> bool:
    """
    Reject candidates whose meaningful vocabulary is mostly absent from the
    current user-authored source. This blocks plausible-sounding memories copied
    from prior context or invented by the extractor.
    """
    candidate = _lexical_grounding_tokens(content)
    source = _lexical_grounding_tokens(source_text)

    if not candidate or not source:
        return False

    overlap = len(candidate & source) / len(candidate)
    has_supported_anchor = bool(
        _extract_source_anchors(content) & _extract_source_anchors(source_text)
    )

    # Technical facts often contain a strong exact identifier plus a few harmless
    # paraphrased words. Pure natural-language claims require stronger overlap.
    required = 0.30 if has_supported_anchor else 0.50
    return overlap >= required


def _estimate_source_fact_limit(source_text: str) -> int:
    """Estimate how many independent memories the current source can justify."""
    text = (source_text or "").strip()
    if not text:
        return 0

    # Remove a leading remember/record wrapper before counting fact segments.
    text = re.sub(
        r"^\s*(?:запомни(?:ть)?|зафиксируй|зафиксировать|remember(?:\s+that)?|record(?:\s+this)?|store(?:\s+this)?)\s*[:,-]?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    segments = [
        part.strip(" -•\t")
        for part in re.split(r"(?<=[.!?])\s+|[;\n]+", text)
        if part and len(part.strip(" -•\t")) >= 8
    ]
    return max(1, min(MAX_EXTRACTED_ITEMS, len(segments) or 1))


def parse_memory_text(raw_text: str, source_text: str = "", max_items: int | None = None) -> list:
    """
    Extracts FACT:/MEMORY: lines and applies conservative quality filters.
    """

    items = []
    seen = set()
    accepted_contents = []
    effective_max_items = max(1, min(MAX_EXTRACTED_ITEMS, int(max_items or MAX_EXTRACTED_ITEMS)))

    pattern = re.compile(
        r'^(FACT|MEMORY):\s*(.+)$',
        re.MULTILINE | re.IGNORECASE
    )

    matches = pattern.findall(raw_text or "")

    if not matches:
        logger.info(
            "[WRITE_PATH] No valid FACT/MEMORY entries extracted."
        )
        return []

    rejected = 0

    for entry_type, content in matches:
        content = _normalize_content(content)

        if _is_blocked_memory(content, source_text=source_text):
            rejected += 1
            logger.debug(
                "[WRITE_PATH] Rejected weak/noisy memory: %s",
                content[:160]
            )
            continue

        if _has_invented_anchors(content, source_text):
            rejected += 1
            logger.info(
                "[WRITE_PATH] Rejected memory with unsupported anchors: %s",
                content[:160]
            )
            continue

        if not _is_lexically_grounded(content, source_text):
            rejected += 1
            logger.info(
                "[WRITE_PATH] Rejected memory not grounded in current source: %s",
                content[:160]
            )
            continue

        key = content.lower()
        if key in seen:
            rejected += 1
            continue

        if _is_near_duplicate(content, accepted_contents):
            rejected += 1
            logger.debug(
                "[WRITE_PATH] Rejected near-duplicate memory: %s",
                content[:160]
            )
            continue

        seen.add(key)
        accepted_contents.append(content)

        items.append({
            "content": content[:MAX_MEMORY_ITEM_CHARS],
            "kind": "episodic",
            "epistemic_type": (
                "Fact"
                if entry_type.upper() == "FACT"
                else "Observation"
            ),
            "source": "direct_user"
        })

        if len(items) >= effective_max_items:
            break

    logger.info(
        "[WRITE_PATH] Parsed memories: accepted=%s rejected=%s raw_matches=%s",
        len(items),
        rejected,
        len(matches)
    )

    return items


def _build_user_supplied_dialogue(messages: list) -> str:
    """
    Builds extraction input strictly from the last user-authored turn.

    Tool output, assistant replies, system prompts, OWUI memory_context and RAG
    source wrappers are never used as memory sources.
    """

    last_user_text = _extract_last_user_text(messages)
    durable_source = _extract_durable_source_text(last_user_text)

    if not durable_source:
        return ""

    return durable_source[-MAX_DIALOGUE_CHARS:]


async def process_episodic(
    messages: list,
    assistant_reply: str,
    active_state: dict
):
    """
    Main episodic-memory write pipeline.
    """

    if not assistant_reply:
        logger.info(
            "[WRITE_PATH] Empty assistant reply. Skipping."
        )
        return

    if len(assistant_reply.strip()) < 10:
        logger.info(
            "[WRITE_PATH] Assistant reply too short. Skipping."
        )
        return

    try:
        logger.info(
            "[WRITE_PATH] Task started. version=%s assistant_reply_length=%s",
            WRITE_PATH_VERSION,
            len(assistant_reply),
        )

        last_user_text = _extract_last_user_text(messages)
        skip, reason = should_skip_memory_extraction(last_user_text)

        if skip:
            logger.info(
                "[WRITE_PATH] Skipping extraction. reason=%s user_text=%r",
                reason,
                last_user_text[:240]
            )
            return

        full_dialogue = _build_user_supplied_dialogue(messages)

        if not full_dialogue:
            logger.info(
                "[WRITE_PATH] No durable user-supplied text available. Skipping."
            )
            return

        logger.info(
            "[WRITE_PATH] User-supplied extraction context length: %s reason=%s",
            len(full_dialogue),
            reason
        )

        prompt = EXTRACT_PROMPT.format(
            dialogue=full_dialogue
        )

        raw_response = await call_small_model(
            prompt,
            grammar=None,
            max_tokens=192
        )

        logger.info(
            "[WRITE_PATH] Raw Small Model response: %s",
            (raw_response or "")[:500]
        )

        source_fact_limit = _estimate_source_fact_limit(full_dialogue)
        logger.info(
            "[WRITE_PATH] Source-grounded item limit: %s",
            source_fact_limit,
        )

        items = parse_memory_text(
            raw_response,
            source_text=full_dialogue,
            max_items=source_fact_limit,
        )

        if not items:
            logger.info(
                "[WRITE_PATH] Nothing useful extracted."
            )
            return

        saved = 0

        for item in items:
            await insert_memory(
                content=item["content"],
                kind=item["kind"],
                importance=0.7,
                tags="[]",
                decay_rate=0.95,
                table_name="episodic_memory",
                epistemic_type=item["epistemic_type"],
                source=item["source"]
            )

            saved += 1

        logger.info(
            "[WRITE_PATH] Saved %s episodic memories.",
            saved
        )

        if saved:
            asyncio.create_task(consolidate())

    except Exception as e:
        logger.error(
            "[WRITE_PATH] Fatal Error: %s",
            e,
            exc_info=True
        )
