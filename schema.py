"""Pydantic schema for structured transcript analysis output."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class EvidencedItem(BaseModel):
    """A single conclusion plus the transcript fragment it is based on."""

    text: str
    source_text: Optional[str] = Field(
        None, description="Дословная (или близкая к дословной) цитата из транскрипта"
    )


class NextActionDate(BaseModel):
    value: Optional[str] = Field(
        None, description="YYYY-MM-DD, только если день определён однозначно; иначе null"
    )
    confidence: Literal["high", "medium", "low"]
    ambiguity: Optional[str] = Field(
        None, description="Объяснение неоднозначности/конфликта дат, или null"
    )
    source_text: Optional[str] = Field(
        None, description="Цитата, на основании которой определена дата"
    )


class CaseAnalysis(BaseModel):
    outcome: str
    next_step: str
    next_step_source_text: Optional[str] = None
    next_action_date: NextActionDate
    customer_needs: list[EvidencedItem] = Field(default_factory=list)
    risks: list[EvidencedItem] = Field(default_factory=list)
    manager_mistakes: list[EvidencedItem] = Field(default_factory=list)
    manager_attention: list[EvidencedItem] = Field(default_factory=list)
