"""
Bookend Recommendation API

✅ 개선사항:
- RecommendationGenerator 통합
- content/type 자동 생성
- 에러 처리 강화
- One Source of Truth 원칙
- Single Responsibility 원칙
"""

import time
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.model_loader import get_model_loader
from src.api.recommendation_generator import get_recommendation_generator
from src.api.schemas import (
    RecommendRequest,
    HybridRecommendRequest,
    BatchRecommendRequest,
    RecommendResponse,
    BatchRecommendResponse,
    RecommendationItem,
    HealthResponse,
    ComponentHealth,
    ModelInfoResponse,
)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI App
app = FastAPI(
    title="Bookend Recommendation API",
    version="0.2.0",
    description="Hybrid v2 with Content Generation",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
model_loader = None
rec_generator = None
startup_time = None


# =========================================================
# Startup/Shutdown
# =========================================================

@app.on_event("startup")
async def startup_event():
    """앱 시작"""
    global model_loader, rec_generator, startup_time
    
    logger.info("🚀 Starting Bookend Recommendation API...")
    startup_time = time.time()
    
    try:
        # 모델 로더
        model_loader = get_model_loader()
        model_loader.load()
        logger.info("✅ Model loaded successfully")
        
        # 추천 제너레이터
        rec_generator = get_recommendation_generator()
        logger.info("✅ Recommendation generator loaded")
        
        logger.info("✅ API ready to serve requests")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """앱 종료"""
    logger.info("🛑 Shutting down Bookend Recommendation API...")
    logger.info("✅ Cleanup complete")


# =========================================================
# Helper Functions (Single Responsibility)
# =========================================================

def _enrich_recommendations(
    recommendations: List[Dict[str, Any]],
    context: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    추천에 content/type 추가
    
    Single Responsibility: content 생성만 담당
    Graceful Degradation: 실패 시 원본 반환
    
    Args:
        recommendations: 모델에서 받은 추천 리스트
        context: 선택적 컨텍스트 정보
    
    Returns:
        content/type이 추가된 추천 리스트
    """
    if not rec_generator:
        logger.warning("Generator not loaded, skipping enrichment")
        return recommendations
    
    try:
        enriched = rec_generator.generate_batch(recommendations, context)
        logger.debug(f"Enriched {len(enriched)} recommendations")
        return enriched
        
    except Exception as e:
        logger.error(f"Enrichment failed: {e}", exc_info=True)
        return recommendations  # Graceful degradation


def _format_response_items(
    recommendations: List[Dict[str, Any]],
    include_reasons: bool = True
) -> List[RecommendationItem]:
    """
    응답 아이템 포맷팅
    
    Single Responsibility: Pydantic 변환만 담당
    
    Args:
        recommendations: 추천 리스트
        include_reasons: 이유 포함 여부
    
    Returns:
        RecommendationItem 리스트
    """
    items = []
    
    for rank, rec in enumerate(recommendations, start=1):
        try:
            item = RecommendationItem(
                item_id=rec['item_id'],
                score=rec['score'],
                rank=rank,
                reasons=rec.get('reasons', []) if include_reasons else [],
                content=rec.get('content'),
                type=rec.get('type')
            )
            items.append(item)
            
        except Exception as e:
            logger.error(f"Failed to format item {rec.get('item_id')}: {e}")
            # 개별 실패해도 나머지 계속 처리
            continue
    
    return items


def _build_response(
    user_id: int,
    recommendations: List[Dict[str, Any]],
    is_cold_start: bool,
    latency_ms: float,
    metadata: Optional[Dict[str, Any]] = None
) -> RecommendResponse:
    """
    표준 응답 생성
    
    Single Responsibility: 응답 객체 생성만 담당
    
    Args:
        user_id: 사용자 ID
        recommendations: 추천 리스트
        is_cold_start: Cold start 여부
        latency_ms: 응답 시간 (ms)
        metadata: 추가 메타데이터
    
    Returns:
        RecommendResponse
    """
    items = _format_response_items(recommendations)
    
    return RecommendResponse(
        user_id=user_id,
        recommendations=items,
        total_count=len(items),
        model_name="Hybrid v2 Rebalanced",
        is_cold_start=is_cold_start,
        latency_ms=latency_ms,
        request_metadata=metadata,
        timestamp=datetime.now()
    )


# =========================================================
# Exception Handlers
# =========================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP 예외 처리"""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "path": str(request.url.path),
                "timestamp": datetime.now().isoformat()
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """일반 예외 처리"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": 500,
                "message": "Internal server error",
                "detail": str(exc),
                "path": str(request.url.path),
                "timestamp": datetime.now().isoformat()
            }
        }
    )


# =========================================================
# Health Check
# =========================================================

@app.get("/", tags=["Root"])
async def root():
    """루트 엔드포인트"""
    return {
        "service": "Bookend Recommendation API",
        "version": "0.2.0",
        "features": ["content_generation", "type_classification"],
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """헬스 체크 엔드포인트"""
    try:
        if model_loader is None:
            return HealthResponse(
                status="unhealthy",
                model_loaded=False,
                components={
                    "model": ComponentHealth(
                        status="unhealthy",
                        message="Model not loaded"
                    )
                },
                timestamp=datetime.now()
            )
        
        model_info = model_loader.get_model_info()
        uptime = time.time() - startup_time if startup_time else 0
        
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            components={
                "model": ComponentHealth(
                    status="healthy",
                    message="Model loaded and ready",
                    details=model_info
                ),
                "generator": ComponentHealth(
                    status="healthy" if rec_generator else "degraded",
                    message="Generator ready" if rec_generator else "Not loaded"
                )
            },
            model_info=model_info,
            uptime_seconds=uptime,
            timestamp=datetime.now()
        )
    
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            components={
                "model": ComponentHealth(
                    status="unhealthy",
                    message=str(e)
                )
            },
            timestamp=datetime.now()
        )


# =========================================================
# Recommendation Endpoints
# =========================================================

@app.post(
    "/api/v1/recommend",
    response_model=RecommendResponse,
    tags=["Recommendations"],
    summary="기본 추천 생성 (content/type 포함)"
)
async def recommend(request: RecommendRequest):
    """
    ✅ content/type이 포함된 기본 추천 생성
    
    Flow:
    1. 모델에서 추천 생성 (item_id, score, reasons)
    2. Generator로 content/type 추가
    3. 응답 포맷팅
    
    Returns:
        RecommendResponse with content/type fields
    """
    start_time = time.time()
    
    try:
        logger.info(
            f"Recommendation request: user_id={request.user_id}, "
            f"limit={request.limit}"
        )
        
        if model_loader is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please try again later."
            )
        
        # 1. Cold start 체크
        is_cold_start = not model_loader.is_user_known(request.user_id)
        
        # 2. 모델에서 추천 생성
        recommendations = model_loader.recommend(
            user_id=request.user_id,
            k=request.limit,
            exclude_interacted=request.exclude_interacted
        )
        
        # 3. content/type 추가
        context_dict = request.context.dict() if request.context else None
        enriched = _enrich_recommendations(recommendations, context_dict)
        
        # 4. 응답 생성
        latency_ms = (time.time() - start_time) * 1000
        
        response = _build_response(
            user_id=request.user_id,
            recommendations=enriched,
            is_cold_start=is_cold_start,
            latency_ms=latency_ms,
            metadata={
                "limit": request.limit,
                "exclude_interacted": request.exclude_interacted,
                "has_context": request.context is not None
            }
        )
        
        logger.info(
            f"✅ Generated {len(response.recommendations)} recommendations "
            f"for user {request.user_id} in {latency_ms:.2f}ms"
        )
        
        return response
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(
            f"Recommendation failed for user {request.user_id}: {e}",
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate recommendations: {str(e)}"
        )


@app.post(
    "/api/v1/recommend/hybrid",
    response_model=RecommendResponse,
    tags=["Recommendations"],
    summary="Hybrid 추천 생성 (고급 옵션, content/type 포함)"
)
async def recommend_hybrid(request: HybridRecommendRequest):
    """
    ✅ Hybrid 추천 생성 (content/type 포함)
    
    고급 기능:
    - min_score 필터링
    - exclude_items
    - diversity 옵션
    
    Flow:
    1. 모델에서 추천 생성 (여유있게 2배)
    2. 필터링 적용
    3. Generator로 content/type 추가
    4. 응답 포맷팅
    """
    start_time = time.time()
    
    try:
        logger.info(
            f"Hybrid recommendation request: user_id={request.user_id}, "
            f"limit={request.limit}, min_score={request.min_score}"
        )
        
        if model_loader is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded"
            )
        
        # 1. Cold start 체크
        is_cold_start = not model_loader.is_user_known(request.user_id)
        
        # 2. 모델에서 추천 생성 (여유있게)
        recommendations = model_loader.recommend(
            user_id=request.user_id,
            k=request.limit * 2,  # 필터링 대비
            exclude_interacted=request.exclude_interacted
        )
        
        # 3. 필터링 적용
        if request.min_score and request.min_score > 0:
            recommendations = [
                rec for rec in recommendations
                if rec['score'] >= request.min_score
            ]
            logger.debug(
                f"After min_score filter: {len(recommendations)} items"
            )
        
        if request.exclude_items:
            recommendations = [
                rec for rec in recommendations
                if rec['item_id'] not in request.exclude_items
            ]
            logger.debug(
                f"After exclude_items filter: {len(recommendations)} items"
            )
        
        # 4. Limit 적용
        recommendations = recommendations[:request.limit]
        
        # 5. content/type 추가
        enriched = _enrich_recommendations(recommendations, request.context)
        
        # 6. 응답 생성
        latency_ms = (time.time() - start_time) * 1000
        
        items = _format_response_items(enriched, request.include_reasons)
        
        response = RecommendResponse(
            user_id=request.user_id,
            recommendations=items,
            total_count=len(items),
            model_name="Hybrid v2 Rebalanced",
            is_cold_start=is_cold_start,
            latency_ms=latency_ms,
            request_metadata={
                "limit": request.limit,
                "min_score": request.min_score,
                "enable_diversity": request.enable_diversity,
                "exclude_items_count": len(request.exclude_items) if request.exclude_items else 0
            },
            timestamp=datetime.now()
        )
        
        logger.info(
            f"✅ Generated {len(items)} hybrid recommendations "
            f"for user {request.user_id} in {latency_ms:.2f}ms"
        )
        
        return response
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(
            f"Hybrid recommendation failed for user {request.user_id}: {e}",
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate hybrid recommendations: {str(e)}"
        )


@app.post(
    "/api/v1/recommend/batch",
    response_model=BatchRecommendResponse,
    tags=["Recommendations"],
    summary="배치 추천 생성 (content/type 포함)"
)
async def recommend_batch(request: BatchRecommendRequest):
    """
    ✅ 배치 추천 생성 (content/type 포함)
    
    여러 사용자에 대해 한 번에 추천 생성
    개별 실패해도 나머지 계속 처리
    """
    start_time = time.time()
    
    try:
        logger.info(
            f"Batch recommendation request: {len(request.user_ids)} users"
        )
        
        if model_loader is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded"
            )
        
        results = {}
        success_count = 0
        failure_count = 0
        
        for user_id in request.user_ids:
            try:
                # 1. 추천 생성
                recommendations = model_loader.recommend(
                    user_id=user_id,
                    k=request.limit,
                    exclude_interacted=request.exclude_interacted
                )
                
                # 2. content/type 추가
                enriched = _enrich_recommendations(recommendations)
                
                # 3. 응답 생성
                is_cold_start = not model_loader.is_user_known(user_id)
                
                response = _build_response(
                    user_id=user_id,
                    recommendations=enriched,
                    is_cold_start=is_cold_start,
                    latency_ms=0  # 개별 latency는 측정 안 함
                )
                
                results[user_id] = response
                success_count += 1
            
            except Exception as e:
                logger.error(
                    f"Batch failed for user {user_id}: {e}",
                    exc_info=True
                )
                failure_count += 1
                # 개별 실패해도 계속 진행
        
        processing_time = time.time() - start_time
        
        logger.info(
            f"✅ Batch completed: {success_count} success, "
            f"{failure_count} failures in {processing_time:.2f}s"
        )
        
        return BatchRecommendResponse(
            results=results,
            total_users=len(request.user_ids),
            success_count=success_count,
            failure_count=failure_count,
            processing_time_seconds=processing_time,
            timestamp=datetime.now()
        )
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Batch endpoint failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Batch recommendation failed: {str(e)}"
        )


# =========================================================
# Model Info
# =========================================================

@app.get(
    "/api/v1/model/info",
    response_model=ModelInfoResponse,
    tags=["Model"],
    summary="모델 정보 조회"
)
async def get_model_info():
    """
    현재 로드된 모델 정보 반환
    """
    try:
        if model_loader is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded"
            )
        
        info = model_loader.get_model_info()
        
        return ModelInfoResponse(
            model_type=info.get("model_type", "Unknown"),
            model_name="Hybrid v2 Rebalanced",
            num_users=info.get("num_users"),
            num_items=info.get("num_items"),
            weights=info.get("weights"),
            config=info,
            timestamp=datetime.now()
        )
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Failed to get model info: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve model information"
        )


@app.post(
    "/api/v1/model/reload",
    tags=["Model"],
    summary="모델 재로드 (관리자용)"
)
async def reload_model():
    """
    모델 재로드 (관리자용)
    
    ⚠️ Warning: 프로덕션에서는 인증 필요
    """
    try:
        logger.warning("⚠️ Admin action: Reloading model...")
        
        if model_loader is None:
            raise HTTPException(
                status_code=503,
                detail="Model loader not initialized"
            )
        
        model_loader.reload()
        
        logger.info("✅ Model reloaded successfully")
        
        return {
            "status": "success",
            "message": "Model reloaded successfully",
            "timestamp": datetime.now().isoformat()
        }
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Model reload failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reload model: {str(e)}"
        )


# =========================================================
# Run Server
# =========================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )