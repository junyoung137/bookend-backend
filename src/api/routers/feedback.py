# src/api/routers/feedback.py
# 피드백 관련 API 엔드포인트

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
        # ✅ 명시적으로 bool 반환 보장
        success = save_feedback_and_process(feedback.dict())
        
        # None 체크
        if success is None:
            success = False
            
        return FeedbackResponse(
            success=bool(success),  # ✅ 명시적 bool 변환
            message="소중한 의견 감사합니다. ☺️" if success else "피드백 저장 중 문제가 발생했습니다."
        )
    except Exception as e:
        logger.error(f"피드백 처리 오류: {e}")
        # ✅ 에러 발생 시에도 valid response 반환
        return FeedbackResponse(
            success=False,
            message=f"오류 발생: {str(e)}"
        )

@router.post("/correct")
async def correct_with_ace(request: dict):
    """
    개인화 교정 (ACE 적용)
    
    - 피드백 있으면: Groq 개인화 교정 (백엔드)
    - 피드백 없으면: 프론트엔드에서 HuggingFace 처리
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
        # ✅ 에러 시에도 유효한 응답 반환
        return {
            'corrected': request['text'],  # 원문 그대로
            'method': 'error',
            'rules_applied': 0,
            'confidence': 0.0,
            'error': str(e)
        }

@router.get("/status/{user_id}")
async def get_feedback_status(user_id: str):
    """
    사용자 피드백 여부 확인
    """
    try:
        has_fb = has_user_feedback(user_id)
        return {
            "user_id": user_id,
            "has_feedback": bool(has_fb),  # ✅ 명시적 bool 변환
            "personalization_enabled": bool(has_fb)
        }
    except Exception as e:
        logger.error(f"상태 확인 오류: {e}")
        # ✅ 에러 시 기본값 반환
        return {
            "user_id": user_id,
            "has_feedback": False,
            "personalization_enabled": False,
            "error": str(e)
        }
