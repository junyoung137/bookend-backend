# src/api/routers/feedback.py 
# 피드백 관련 URL들을 ACE 시스템 함수들과 연결해주는 중개자

from fastapi import APIRouter, HTTPException
from src.api.schemas.feedback_schema import FeedbackRequest, FeedbackResponse
from src.ace.db.handler import (
    save_feedback_and_process,
    correct_with_personalization,
    has_user_feedback,
    init_db
)
from config.settings import get_settings
import logging

settings = get_settings()

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/feedback", tags=["feedback"])


@router.post("/submit", response_model=FeedbackResponse)
async def submit_feedback(feedback: FeedbackRequest):
    """
    피드백 제출 + ACE 파이프라인 실행
    """
    try:
        success = save_feedback_and_process(feedback.dict())

        return FeedbackResponse(
            success=success,
            message="소중한 의견 감사합니다. ☺️" if success else "피드백 저장 중 문제가 발생했습니다."
        )
    except Exception as e:
        logger.error(f"피드백 처리 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/correct")
async def correct_with_ace(request: dict):
    """
    개인화 교정 (ACE 적용)
    """
    try:
        result = correct_with_personalization(
            user_id=request['user_id'],
            text=request['text'],
            feature=request['feature'],
            tone=request.get('tone', 'normal'),
            genre=request.get('genre', 'informative'),
        )

        return result
    
    except Exception as e:
        logger.error(f"교정 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{user_id}")
async def get_feedback_status(user_id: str):
    """
    사용자 피드백 여부 확인
    """
    try:
        has_fb = has_user_feedback(user_id)
        return {
            "user_id": user_id,
            "has_feedback": has_fb,
            "personalization_enabled": has_fb
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))