"""LLM call (structured tool-use) + offline fallback for the 4 sample transcripts."""

import json
import os
from datetime import date
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from prompts import SYSTEM_PROMPT, build_user_message
from schema import CaseAnalysis

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")

_EVIDENCED_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "source_text": {"type": ["string", "null"]},
    },
    "required": ["text", "source_text"],
}

CASE_ANALYSIS_TOOL = {
    "name": "record_case_analysis",
    "description": "Записать структурированный анализ одного транскрипта звонка по заданной схеме.",
    "input_schema": {
        "type": "object",
        "properties": {
            "outcome": {"type": "string", "description": "Итог разговора"},
            "next_step": {
                "type": "string",
                "description": "Согласованный следующий шаг, либо явное указание, что его нет",
            },
            "next_step_source_text": {"type": ["string", "null"]},
            "next_action_date": {
                "type": "object",
                "properties": {
                    "value": {"type": ["string", "null"], "description": "YYYY-MM-DD или null"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "ambiguity": {"type": ["string", "null"]},
                    "source_text": {"type": ["string", "null"]},
                },
                "required": ["value", "confidence", "ambiguity", "source_text"],
            },
            "customer_needs": {"type": "array", "items": _EVIDENCED_ITEM_SCHEMA},
            "risks": {"type": "array", "items": _EVIDENCED_ITEM_SCHEMA},
            "manager_mistakes": {"type": "array", "items": _EVIDENCED_ITEM_SCHEMA},
            "manager_attention": {"type": "array", "items": _EVIDENCED_ITEM_SCHEMA},
        },
        "required": [
            "outcome",
            "next_step",
            "next_step_source_text",
            "next_action_date",
            "customer_needs",
            "risks",
            "manager_mistakes",
            "manager_attention",
        ],
    },
}


class ExtractionError(RuntimeError):
    pass


def _get_api_key() -> Optional[str]:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    try:
        import streamlit as st

        return st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        return None


def analyze_transcript(
    transcript_text: str,
    call_date: date,
    stated_weekday: str,
    model: str = DEFAULT_MODEL,
) -> CaseAnalysis:
    api_key = _get_api_key()
    if not api_key:
        raise ExtractionError(
            "ANTHROPIC_API_KEY не задан. Установите переменную окружения, добавьте "
            "её в .env (см. .env.example) или в st.secrets перед запуском."
        )

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    user_message = build_user_message(transcript_text, call_date.isoformat(), stated_weekday)

    response = client.messages.create(
        model=model,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        tools=[CASE_ANALYSIS_TOOL],
        tool_choice={"type": "tool", "name": "record_case_analysis"},
        messages=[{"role": "user", "content": user_message}],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "record_case_analysis":
            return CaseAnalysis.model_validate(block.input)

    raise ExtractionError("Модель не вернула ожидаемый structured tool-call.")


def load_expected_result(transcript_id: int) -> CaseAnalysis:
    """Manually-authored ground truth, used as an offline demo fallback and as the
    comparison baseline for testing (see README 'Что было проверено')."""
    path = Path(__file__).parent / "data" / "expected_results.json"
    items = json.loads(path.read_text(encoding="utf-8"))
    for item in items:
        if item["transcript_id"] == transcript_id:
            return CaseAnalysis.model_validate(item["analysis"])
    raise ExtractionError(f"Нет эталонного результата для транскрипта {transcript_id}")
