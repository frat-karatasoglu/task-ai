"""Streamlit MVP: анализ транскриптов клиентских звонков (Saby-интегратор)."""

import json
import pathlib
from datetime import date, datetime

import streamlit as st

from extractor import ExtractionError, analyze_transcript, load_expected_result
from schema import CaseAnalysis

st.set_page_config(page_title="Анализ транскриптов — Saby", layout="wide")

DATA_PATH = pathlib.Path(__file__).parent / "data" / "transcripts.json"
SAMPLES = json.loads(DATA_PATH.read_text(encoding="utf-8"))

CUSTOM_OPTION = "— вставить свой транскрипт —"

st.title("MVP: анализ клиентских звонков (Saby-интегратор)")
st.caption(
    "Итог, следующий согласованный шаг, дата следующего действия, потребности клиента, "
    "риски, возможные ошибки менеджера и то, что требует внимания руководителя — по одному транскрипту."
)

sample_labels = [CUSTOM_OPTION] + [f"Транскрипт {s['id']} — {s['client']}" for s in SAMPLES]
choice = st.selectbox("Тестовый транскрипт", sample_labels)

selected_sample = None
if choice == CUSTOM_OPTION:
    transcript_text = st.text_area(
        "Текст транскрипта", height=300, placeholder="Вставьте текст разговора..."
    )
    col1, col2 = st.columns(2)
    with col1:
        call_date: date = st.date_input("Дата разговора", value=date.today())
    with col2:
        stated_weekday = st.text_input("День недели по тексту транскрипта (если указан)", value="")
else:
    selected_sample = SAMPLES[sample_labels.index(choice) - 1]
    transcript_text = st.text_area("Текст транскрипта", value=selected_sample["text"], height=300)
    call_date = datetime.strptime(selected_sample["call_date"], "%Y-%m-%d").date()
    stated_weekday = selected_sample["stated_weekday"]
    st.caption(f"Дата разговора (из данных): {call_date.isoformat()} ({stated_weekday})")

analyze_clicked = st.button("Анализировать", type="primary")

if analyze_clicked:
    if not transcript_text.strip():
        st.warning("Вставьте текст транскрипта.")
    else:
        with st.spinner("Анализирую..."):
            result = None
            used_fallback = False
            try:
                result = analyze_transcript(transcript_text, call_date, stated_weekday or "не указан")
            except ExtractionError as e:
                if selected_sample is not None:
                    st.info(
                        f"⚠️ {e}\n\nПоказан заранее подготовленный вручную эталонный результат "
                        f"для транскрипта {selected_sample['id']} (демо-режим без живого API)."
                    )
                    result = load_expected_result(selected_sample["id"])
                    used_fallback = True
                else:
                    st.error(str(e))
            except Exception as e:  # неожиданная ошибка API/парсинга
                st.error(f"Не удалось получить структурированный ответ от модели: {e}")

        if result is not None:
            st.session_state["last_result"] = result.model_dump()
            st.session_state["used_fallback"] = used_fallback

if "last_result" in st.session_state:
    result = CaseAnalysis.model_validate(st.session_state["last_result"])
    if st.session_state.get("used_fallback"):
        st.warning("Показан эталонный (заранее подготовленный) результат, не live-вызов LLM.", icon="⚠️")

    st.divider()

    st.subheader("1. Итог разговора")
    st.write(result.outcome)

    st.subheader("2. Следующий согласованный шаг")
    st.write(result.next_step)
    if result.next_step_source_text:
        st.caption(f"Источник: «{result.next_step_source_text}»")

    st.subheader("3. Дата следующего действия")
    nad = result.next_action_date
    confidence_badge = {"high": "🟢 высокая", "medium": "🟡 средняя", "low": "🟠 низкая"}[nad.confidence]
    if nad.value:
        st.write(f"**{nad.value}** — уверенность: {confidence_badge}")
    else:
        st.write("**Дата не определена однозначно (null)**")
    if nad.ambiguity:
        st.caption(f"Пояснение: {nad.ambiguity}")
    if nad.source_text:
        st.caption(f"Источник: «{nad.source_text}»")

    def render_evidenced_list(title: str, items, empty_note: str) -> None:
        st.subheader(title)
        if not items:
            st.write(f"_{empty_note}_")
            return
        for it in items:
            st.markdown(f"- {it.text}")
            if it.source_text:
                st.caption(f"  «{it.source_text}»")

    render_evidenced_list("4. Потребности клиента", result.customer_needs, "Потребности явно не озвучены.")
    render_evidenced_list("5. Основные риски", result.risks, "Риски не выявлены из текста.")
    render_evidenced_list(
        "6. Возможные ошибки менеджера",
        result.manager_mistakes,
        "Оснований для ошибок менеджера в тексте не найдено.",
    )
    render_evidenced_list(
        "7. Что требует внимания руководителя",
        result.manager_attention,
        "Специальных поводов для внимания руководителя не выявлено.",
    )

    with st.expander("Исходный JSON"):
        st.json(result.model_dump())
else:
    st.info("Выберите тестовый транскрипт или вставьте свой и нажмите «Анализировать».")

with st.sidebar:
    st.markdown("### О режиме работы")
    st.markdown(
        "Ключ читается из `ANTHROPIC_API_KEY` (env / `.env` / `st.secrets`), "
        "никогда не хардкодится.\n\n"
        "Если ключ не задан — для 4 тестовых транскриптов показывается заранее "
        "подготовленный эталонный результат (демо-режим), UI при этом остаётся рабочим."
    )
