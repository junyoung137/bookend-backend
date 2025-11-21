from typing import Optional, Dict, Any, Callable
from functools import wraps
import time
import logging
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Summary,
    Info,
    CollectorRegistry,
    REGISTRY,
    generate_latest,
)

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Prometheus 메트릭 수집기"""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or REGISTRY
        self._init_metrics()
        logger.info("Metrics collector initialized")
    
    def _init_metrics(self) -> None:
        """메트릭 초기화"""
        
        # ============================================================
        # HTTP 요청 메트릭
        # ============================================================
        self.http_requests_total = Counter(
            name='bookend_http_requests_total',
            documentation='Total HTTP requests',
            labelnames=['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.http_request_duration_seconds = Histogram(
            name='bookend_http_request_duration_seconds',
            documentation='HTTP request latency in seconds',
            labelnames=['method', 'endpoint'],
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self.registry
        )
        
        self.http_request_size_bytes = Summary(
            name='bookend_http_request_size_bytes',
            documentation='HTTP request size in bytes',
            labelnames=['method', 'endpoint'],
            registry=self.registry
        )
        
        self.http_response_size_bytes = Summary(
            name='bookend_http_response_size_bytes',
            documentation='HTTP response size in bytes',
            labelnames=['method', 'endpoint'],
            registry=self.registry
        )
        
        # ============================================================
        # 추천 시스템 메트릭
        # ============================================================
        self.recommendations_total = Counter(
            name='bookend_recommendations_total',
            documentation='Total recommendation requests',
            labelnames=['model_type', 'status'],
            registry=self.registry
        )
        
        self.recommendation_latency_seconds = Histogram(
            name='bookend_recommendation_latency_seconds',
            documentation='Recommendation generation latency',
            labelnames=['model_type'],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
            registry=self.registry
        )
        
        self.recommendation_items_count = Histogram(
            name='bookend_recommendation_items_count',
            documentation='Number of items in recommendation response',
            labelnames=['model_type'],
            buckets=(1, 3, 5, 10, 20, 50),
            registry=self.registry
        )
        
        self.recommendation_score_distribution = Histogram(
            name='bookend_recommendation_score_distribution',
            documentation='Distribution of recommendation scores',
            labelnames=['model_type'],
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
            registry=self.registry
        )
        
        # ============================================================
        # 모델 성능 메트릭
        # ============================================================
        self.model_fit_duration_seconds = Histogram(
            name='bookend_model_fit_duration_seconds',
            documentation='Model fitting duration',
            labelnames=['model_type'],
            buckets=(1, 5, 10, 30, 60, 120, 300),
            registry=self.registry
        )
        
        self.model_predict_duration_seconds = Histogram(
            name='bookend_model_predict_duration_seconds',
            documentation='Model prediction duration',
            labelnames=['model_type'],
            buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
            registry=self.registry
        )
        
        self.model_is_fitted = Gauge(
            name='bookend_model_is_fitted',
            documentation='Model fitted status (1=fitted, 0=not fitted)',
            labelnames=['model_type'],
            registry=self.registry
        )
        
        # ============================================================
        # 데이터베이스 메트릭
        # ============================================================
        self.db_connections_active = Gauge(
            name='bookend_db_connections_active',
            documentation='Number of active database connections',
            registry=self.registry
        )
        
        self.db_query_duration_seconds = Histogram(
            name='bookend_db_query_duration_seconds',
            documentation='Database query duration',
            labelnames=['query_type'],
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
            registry=self.registry
        )
        
        self.db_errors_total = Counter(
            name='bookend_db_errors_total',
            documentation='Total database errors',
            labelnames=['error_type'],
            registry=self.registry
        )
        
        # ============================================================
        # 캐시 메트릭
        # ============================================================
        self.cache_hits_total = Counter(
            name='bookend_cache_hits_total',
            documentation='Total cache hits',
            labelnames=['cache_type'],
            registry=self.registry
        )
        
        self.cache_misses_total = Counter(
            name='bookend_cache_misses_total',
            documentation='Total cache misses',
            labelnames=['cache_type'],
            registry=self.registry
        )
        
        self.cache_size_bytes = Gauge(
            name='bookend_cache_size_bytes',
            documentation='Cache size in bytes',
            labelnames=['cache_type'],
            registry=self.registry
        )
        
        # ============================================================
        # 사용자 활동 메트릭
        # ============================================================
        self.active_users_total = Gauge(
            name='bookend_active_users_total',
            documentation='Number of active users',
            labelnames=['time_window'],
            registry=self.registry
        )
        
        self.user_interactions_total = Counter(
            name='bookend_user_interactions_total',
            documentation='Total user interactions',
            labelnames=['event_type'],
            registry=self.registry
        )
        
        # ============================================================
        # 에러 메트릭
        # ============================================================
        self.errors_total = Counter(
            name='bookend_errors_total',
            documentation='Total errors',
            labelnames=['error_type', 'severity'],
            registry=self.registry
        )
        
        self.exceptions_total = Counter(
            name='bookend_exceptions_total',
            documentation='Total exceptions',
            labelnames=['exception_type', 'module'],
            registry=self.registry
        )
        
        # ============================================================
        # 시스템 정보 메트릭
        # ============================================================
        self.app_info = Info(
            name='bookend_app',
            documentation='Application information',
            registry=self.registry
        )
        
        self.app_info.info({
            'version': '0.2.0',
            'environment': 'production',
            'service': 'bookend-api'
        })
    
    def track_http_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration: float,
        request_size: Optional[int] = None,
        response_size: Optional[int] = None
    ) -> None:
        """HTTP 요청 메트릭 기록"""
        try:
            self.http_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code
            ).inc()
            
            self.http_request_duration_seconds.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
            if request_size is not None:
                self.http_request_size_bytes.labels(
                    method=method,
                    endpoint=endpoint
                ).observe(request_size)
            
            if response_size is not None:
                self.http_response_size_bytes.labels(
                    method=method,
                    endpoint=endpoint
                ).observe(response_size)
        except Exception as e:
            logger.error(f"Failed to track HTTP request metrics: {e}")
    
    def track_recommendation(
        self,
        model_type: str,
        status: str,
        latency: float,
        items_count: int,
        avg_score: Optional[float] = None
    ) -> None:
        """추천 메트릭 기록"""
        try:
            self.recommendations_total.labels(
                model_type=model_type,
                status=status
            ).inc()
            
            self.recommendation_latency_seconds.labels(
                model_type=model_type
            ).observe(latency)
            
            self.recommendation_items_count.labels(
                model_type=model_type
            ).observe(items_count)
            
            if avg_score is not None:
                self.recommendation_score_distribution.labels(
                    model_type=model_type
                ).observe(avg_score)
        except Exception as e:
            logger.error(f"Failed to track recommendation metrics: {e}")
    
    def track_model_fit(self, model_type: str, duration: float) -> None:
        """모델 학습 메트릭 기록"""
        try:
            self.model_fit_duration_seconds.labels(
                model_type=model_type
            ).observe(duration)
            
            self.model_is_fitted.labels(
                model_type=model_type
            ).set(1)
        except Exception as e:
            logger.error(f"Failed to track model fit metrics: {e}")
    
    def track_model_predict(self, model_type: str, duration: float) -> None:
        """모델 예측 메트릭 기록"""
        try:
            self.model_predict_duration_seconds.labels(
                model_type=model_type
            ).observe(duration)
        except Exception as e:
            logger.error(f"Failed to track model predict metrics: {e}")
    
    def track_error(
        self,
        error_type: str,
        severity: str = "error",
        exception_type: Optional[str] = None,
        module: Optional[str] = None
    ) -> None:
        """에러 메트릭 기록"""
        try:
            self.errors_total.labels(
                error_type=error_type,
                severity=severity
            ).inc()
            
            if exception_type and module:
                self.exceptions_total.labels(
                    exception_type=exception_type,
                    module=module
                ).inc()
        except Exception as e:
            logger.error(f"Failed to track error metrics: {e}")
    
    def track_cache(
        self,
        cache_type: str,
        hit: bool,
        size_bytes: Optional[int] = None
    ) -> None:
        """캐시 메트릭 기록"""
        try:
            if hit:
                self.cache_hits_total.labels(cache_type=cache_type).inc()
            else:
                self.cache_misses_total.labels(cache_type=cache_type).inc()
            
            if size_bytes is not None:
                self.cache_size_bytes.labels(cache_type=cache_type).set(size_bytes)
        except Exception as e:
            logger.error(f"Failed to track cache metrics: {e}")
    
    def get_metrics(self) -> bytes:
        """현재 메트릭 반환"""
        return generate_latest(self.registry)


# ============================================================
# 싱글톤 인스턴스
# ============================================================
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_registry() -> MetricsCollector:
    """메트릭 수집기 싱글톤 반환"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


# ============================================================
# 데코레이터 헬퍼 함수
# ============================================================

def track_request_duration(endpoint: str):
    """HTTP 요청 duration 트래킹 데코레이터"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            metrics = get_metrics_registry()
            start_time = time.time()
            try:
                response = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                # 상태 코드 추출
                status_code = getattr(response, 'status_code', 200)
                
                metrics.track_http_request(
                    method='GET',  # 실제로는 request에서 추출
                    endpoint=endpoint,
                    status_code=status_code,
                    duration=duration
                )
                
                return response
            except Exception as e:
                duration = time.time() - start_time
                metrics.track_http_request(
                    method='GET',
                    endpoint=endpoint,
                    status_code=500,
                    duration=duration
                )
                raise
        return wrapper
    return decorator


def track_recommendation_latency(model_type: str):
    """추천 latency 트래킹 데코레이터"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            metrics = get_metrics_registry()
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                latency = time.time() - start_time
                
                # 결과에서 메트릭 추출
                items_count = len(result) if isinstance(result, list) else 0
                avg_score = None
                if items_count > 0 and hasattr(result[0], 'score'):
                    avg_score = sum(item.score for item in result) / items_count
                
                metrics.track_recommendation(
                    model_type=model_type,
                    status='success',
                    latency=latency,
                    items_count=items_count,
                    avg_score=avg_score
                )
                
                return result
            except Exception as e:
                latency = time.time() - start_time
                metrics.track_recommendation(
                    model_type=model_type,
                    status='error',
                    latency=latency,
                    items_count=0
                )
                raise
        return wrapper
    return decorator


def increment_recommendation_counter(model_type: str, status: str = 'success') -> None:
    """추천 카운터 증가"""
    metrics = get_metrics_registry()
    metrics.recommendations_total.labels(
        model_type=model_type,
        status=status
    ).inc()


def increment_error_counter(
    error_type: str,
    severity: str = 'error',
    exception_type: Optional[str] = None,
    module: Optional[str] = None
) -> None:
    """에러 카운터 증가"""
    metrics = get_metrics_registry()
    metrics.track_error(error_type, severity, exception_type, module)