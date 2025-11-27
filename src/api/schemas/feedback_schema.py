# src/api/schemas/feedback_schema.py

from pydantic import BaseModel, Field
from typing import Optional, Dict
from datetime import datetime

class FeedbackRequest(BaseModel):
    """피드백 요청 스키마"""
    user_id: str = Field(..., description="사용자 ID")
    original: str = Field(..., description="원문")
    corrected_text: str = Field(..., description="교정된 문장")
    selected_feature: str = Field(..., description="선택된 기능")
    feedback: str = Field(..., description="피드백 (만족/불만족)")
    context: Dict = Field(default_factory=dict, description="맥락 정보")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="타임스탬프"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_001",
                "original": "최근 연구 결과들에 따르면...",
                "corrected_text": "최근 연구에 따르면...",
                "selected_feature": "Paraphrase",
                "feedback": "만족",
                "context": {"tone": "normal", "genre": "informative"},
                "timestamp": "2025-11-24T10:00:00"
            }
        }


class FeedbackResponse(BaseModel):
    """피드백 응답 스키마"""
    success: bool = Field(..., description="성공 여부")
    message: str = Field(..., description="응답 메시지")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "소중한 의견 감사합니다. ☺️"
            }
        }