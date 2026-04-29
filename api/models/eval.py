from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator


class EvalRequest(BaseModel):
    doc_id: int = Field(..., gt=0)
    questions: list[str] = Field(..., min_length=1, max_length=20)
    ground_truths: list[str] = Field(..., min_length=1, max_length=20)
    delay_s: float = Field(5.0, ge=1.0, le=60.0)

    @field_validator("questions", "ground_truths")
    @classmethod
    def validate_items_non_empty(cls, v: list[str]) -> list[str]:
        for item in v:
            if not item.strip():
                raise ValueError("Items cannot be empty strings")
            if len(item) > 2000:
                raise ValueError("Each item must be 2000 characters or fewer")
        return v

    @model_validator(mode="after")
    def validate_lengths_match(self) -> "EvalRequest":
        if len(self.questions) != len(self.ground_truths):
            raise ValueError("questions and ground_truths must have the same length")
        return self


class ReIngestRequest(BaseModel):
    user_feedback: str | None = Field(None, max_length=2000)
    workspace_id: int | None = Field(None, gt=0)


class RevertRequest(BaseModel):
    doc_id: int = Field(..., gt=0)
    version: int = Field(..., gt=0)


class NameStrategyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("name cannot be blank")
        return v
