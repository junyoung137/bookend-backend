# src/api/routers/metrics.py
from typing import Dict, Any
import logging
from fastapi import APIRouter, Response, Depends
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

from src.api.dependencies import get_db_session
from src.monitoring.metrics import get_metrics_collector
from src.database.models import User, Interaction

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/metrics",
    response_class=PlainTextResponse,
    summary="Prometheus Metrics Endpoint",
    description="Returns metrics in Prometheus format for scraping"
)
async def metrics_endpoint() -> Response:
    """
    Prometheus 메트릭 엔드포인트
    - Prometheus가 이 엔드포인트를 주기적으로 스크래핑
    - text/plain 형식으로 메트릭 반환
    """
    try:
        collector = get_metrics_collector()
        metrics_data = collector.get_metrics()
        
        return Response(
            content=metrics_data,
            media_type="text/plain; version=0.0.4; charset=utf-8"
        )
    
    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}", exc_info=True)
        # 메트릭 생성 실패 시에도 빈 응답 반환 (Prometheus 에러 방지)
        return Response(
            content="# Metrics generation failed\n",
            media_type="text/plain",
            status_code=500
        )


@router.get(
    "/metrics/health",
    summary="Metrics System Health",
    response_model=Dict[str, Any]
)
async def metrics_health() -> Dict[str, Any]:
    """
    메트릭 시스템 헬스 체크
    - 메트릭 수집기 상태 확인
    - 수집된 메트릭 개수 확인
    """
    try:
        collector = get_metrics_collector()
        
        # 레지스트리에서 수집기 개수 확인
        collectors_count = len(list(collector.registry.collect()))
        
        return {
            "status": "healthy",
            "collectors_count": collectors_count,
            "timestamp": datetime.now().isoformat(),
            "registry": "REGISTRY" if collector.registry else "custom"
        }
    
    except Exception as e:
        logger.error(f"Metrics health check failed: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.post(
    "/metrics/update-active-users",
    summary="Update Active Users Metric",
    description="Manually update active users count (for scheduled tasks)"
)
async def update_active_users(
    db: Session = Depends(get_db_session)
) -> Dict[str, Any]:
    """
    활성 사용자 메트릭 업데이트
    - 주기적으로 호출되어야 함 (크론잡 또는 백그라운드 태스크)
    - 5분/1시간/24시간 활성 사용자 수 계산
    """
    try:
        collector = get_metrics_collector()
        now = datetime.now()
        
        # 5분 활성 사용자
        five_min_ago = now - timedelta(minutes=5)
        active_5m = db.query(User.id).join(Interaction).filter(
            Interaction.event_time >= five_min_ago
        ).distinct().count()
        
        # 1시간 활성 사용자
        one_hour_ago = now - timedelta(hours=1)
        active_1h = db.query(User.id).join(Interaction).filter(
            Interaction.event_time >= one_hour_ago
        ).distinct().count()
        
        # 24시간 활성 사용자
        one_day_ago = now - timedelta(days=1)
        active_24h = db.query(User.id).join(Interaction).filter(
            Interaction.event_time >= one_day_ago
        ).distinct().count()
        
        # 메트릭 업데이트
        collector.active_users_total.labels(time_window="5m").set(active_5m)
        collector.active_users_total.labels(time_window="1h").set(active_1h)
        collector.active_users_total.labels(time_window="24h").set(active_24h)
        
        logger.info(
            f"Active users updated: 5m={active_5m}, 1h={active_1h}, 24h={active_24h}"
        )
        
        return {
            "status": "success",
            "active_users": {
                "5m": active_5m,
                "1h": active_1h,
                "24h": active_24h
            },
            "timestamp": now.isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to update active users: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.get(
    "/metrics/summary",
    summary="Metrics Summary",
    response_model=Dict[str, Any]
)
async def metrics_summary() -> Dict[str, Any]:
    """
    메트릭 요약 정보
    - 주요 메트릭의 현재 값 확인
    - 디버깅 및 모니터링용
    """
    try:
        collector = get_metrics_collector()
        
        # 각 메트릭의 현재 샘플 수집
        summary = {
            "http_requests": {
                "total": _get_metric_value(
                    collector.http_requests_total,
                    "sum"
                ),
                "description": "Total HTTP requests"
            },
            "recommendations": {
                "total": _get_metric_value(
                    collector.recommendations_total,
                    "sum"
                ),
                "description": "Total recommendation requests"
            },
            "errors": {
                "total": _get_metric_value(
                    collector.errors_total,
                    "sum"
                ),
                "description": "Total errors"
            },
            "cache": {
                "hits": _get_metric_value(
                    collector.cache_hits_total,
                    "sum"
                ),
                "misses": _get_metric_value(
                    collector.cache_misses_total,
                    "sum"
                ),
                "description": "Cache hit/miss counts"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return summary
    
    except Exception as e:
        logger.error(f"Failed to generate metrics summary: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def _get_metric_value(metric, aggregation: str = "sum") -> float:
    """
    메트릭의 현재 값 추출 (헬퍼 함수)
    
    Args:
        metric: Prometheus 메트릭 객체
        aggregation: 집계 방식 ("sum", "count", "avg")
    
    Returns:
        메트릭 값 (float)
    """
    try:
        # Counter/Gauge의 경우
        if hasattr(metric, '_value'):
            return float(metric._value.get())
        
        # Labeled metric의 경우
        if hasattr(metric, '_metrics'):
            values = []
            for labeled_metric in metric._metrics.values():
                if hasattr(labeled_metric, '_value'):
                    values.append(labeled_metric._value.get())
            
            if not values:
                return 0.0
            
            if aggregation == "sum":
                return float(sum(values))
            elif aggregation == "avg":
                return float(sum(values) / len(values))
            elif aggregation == "count":
                return float(len(values))
        
        return 0.0
    
    except Exception as e:
        logger.debug(f"Failed to get metric value: {e}")
        return 0.0


@router.post(
    "/metrics/reset",
    summary="Reset Metrics (Debug Only)",
    description="Reset all metrics - use with caution!"
)
async def reset_metrics() -> Dict[str, Any]:
    """
    모든 메트릭 리셋 (디버깅용)
    ⚠️ 프로덕션 환경에서는 사용 금지
    """
    try:
        # 새로운 수집기 인스턴스 생성 (기존 메트릭 초기화)
        from src.monitoring.metrics import MetricsCollector
        global _metrics_collector
        
        # 싱글톤 리셋
        import src.monitoring.metrics as metrics_module
        metrics_module._metrics_collector = MetricsCollector()
        
        logger.warning("Metrics reset performed (debug action)")
        
        return {
            "status": "success",
            "message": "All metrics have been reset",
            "timestamp": datetime.now().isoformat(),
            "warning": "This is a debug action - do not use in production"
        }
    
    except Exception as e:
        logger.error(f"Failed to reset metrics: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


if __name__ == "__main__":
    from config.logging_config import setup_logging
    setup_logging()
    
    print("\n" + "=" * 70)
    print("METRICS ROUTER TEST")
    print("=" * 70)
    
    # 메트릭 수집기 초기화 테스트
    collector = get_metrics_collector()
    print(f"\nMetrics collector initialized: {type(collector).__name__}")
    
    # 샘플 메트릭 생성
    collector.track_http_request("GET", "/test", 200, 0.1)
    collector.track_recommendation("ambient", "success", 0.5, 5, 0.8)
    
    print("Sample metrics tracked")
    
    # 메트릭 출력 테스트
    metrics_output = collector.get_metrics()
    print(f"\nMetrics output length: {len(metrics_output)} bytes")
    print(f"First 500 chars:\n{metrics_output[:500].decode('utf-8')}")
    
    print("\n" + "=" * 70)
    print("Metrics router test complete")