"""LLM call (Gemini structured JSON output) + offline fallback for the 4 sample transcripts."""

import json
import os
from datetime import date
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from prompts import SYSTEM_PROMPT, build_user_message
from schema import CaseAnalysis

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


class ExtractionError(RuntimeError):
    pass


def _get_api_key() -> Optional[str]:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key:
        return key
    try:
        import streamlit as st

        return st.secrets.get("GEMINI_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
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
            "GEMINI_API_KEY не задан. Получите бесплатный ключ на "
            "https://aistudio.google.com/apikey и добавьте его в .env "
            "(см. .env.example) или в st.secrets перед запуском."
        )

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    user_message = build_user_message(transcript_text, call_date.isoformat(), stated_weekday)

    response = client.models.generate_content(
        model=model,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_json_schema=CaseAnalysis.model_json_schema(),
            temperature=0.2,
        ),
    )

    if not response.text:
        raise ExtractionError("Модель не вернула структурированный JSON-ответ.")

    try:
        data = json.loads(response.text)
    except json.JSONDecodeError as e:
        raise ExtractionError(f"Ответ модели не является валидным JSON: {e}") from e

    return CaseAnalysis.model_validate(data)


def load_expected_result(transcript_id: int) -> CaseAnalysis:
    """Manually-authored ground truth, used as an offline demo fallback and as the
    comparison baseline for testing (see README 'Что было проверено')."""
    path = Path(__file__).parent / "data" / "expected_results.json"
    items = json.loads(path.read_text(encoding="utf-8"))
    for item in items:
        if item["transcript_id"] == transcript_id:
            return CaseAnalysis.model_validate(item["analysis"])
    raise ExtractionError(f"Нет эталонного результата для транскрипта {transcript_id}")
