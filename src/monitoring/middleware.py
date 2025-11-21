# src/monitoring/middleware.py
from typing import Callable, List, Optional
import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from fastapi import Request, Response
from src.monitoring.metrics import get_metrics_collector

logger = logging.getLogger(__name__)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """
    Prometheus 메트릭 수집 미들웨어
    - HTTP 요청/응답 추적
    - 레이턴시 측정
    - 상태 코드별 카운팅
    """

    def __init__(
        self,
        app: ASGIApp,
        exclude_paths: Optional[List[str]] = None
    ):
        super().__init__(app)
        self.metrics = get_metrics_collector()
        self.exclude_paths = exclude_paths or []
        logger.info(
            f"PrometheusMiddleware initialized with "
            f"{len(self.exclude_paths)} excluded paths"
        )

    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """HTTP 요청 처리 및 메트릭 수집"""
        
        # 제외 경로 체크
        if self._should_exclude(request.url.path):
            return await call_next(request)

        # 요청 시작 시간
        start_time = time.time()
        
        # 요청 크기 측정
        request_size = self._get_request_size(request)
        
        # 경로 정규화 (파라미터 제거)
        endpoint = self._normalize_endpoint(request.url.path)
        method = request.method

        try:
            # 요청 처리
            response = await call_next(request)
            
            # 응답 크기 측정
            response_size = self._get_response_size(response)
            
            # 레이턴시 계산
            duration = time.time() - start_time
            
            # 메트릭 기록
            self.metrics.track_http_request(
                method=method,
                endpoint=endpoint,
                status_code=response.status_code,
                duration=duration,
                request_size=request_size,
                response_size=response_size
            )
            
            return response

        except Exception as e:
            # 에러 발생 시에도 메트릭 기록
            duration = time.time() - start_time
            
            self.metrics.track_http_request(
                method=method,
                endpoint=endpoint,
                status_code=500,
                duration=duration,
                request_size=request_size
            )
            
            # 에러 메트릭 추가
            self.metrics.track_error(
                error_type="middleware_exception",
                severity="error",
                exception_type=type(e).__name__,
                module="prometheus_middleware"
            )
            
            logger.error(
                f"Middleware error: {e} [{method} {endpoint}]",
                exc_info=True
            )
            
            raise

    def _should_exclude(self, path: str) -> bool:
        """경로가 제외 대상인지 확인"""
        return any(path.startswith(excluded) for excluded in self.exclude_paths)

    def _normalize_endpoint(self, path: str) -> str:
        """
        엔드포인트 경로 정규화
        - /api/v1/users/123 -> /api/v1/users/{id}
        - /api/v1/items/456/details -> /api/v1/items/{id}/details
        """
        try:
            parts = path.split('/')
            normalized_parts = []
            
            for i, part in enumerate(parts):
                # 숫자로만 구성된 경로 파라미터를 {id}로 치환
                if part.isdigit():
                    normalized_parts.append('{id}')
                # UUID 패턴 감지 (간단한 버전)
                elif len(part) == 36 and part.count('-') == 4:
                    normalized_parts.append('{uuid}')
                else:
                    normalized_parts.append(part)
            
            return '/'.join(normalized_parts)
        
        except Exception as e:
            logger.warning(f"Failed to normalize endpoint {path}: {e}")
            return path

    def _get_request_size(self, request: Request) -> Optional[int]:
        """요청 크기 추정"""
        try:
            content_length = request.headers.get('content-length')
            if content_length:
                return int(content_length)
            return None
        except Exception as e:
            logger.debug(f"Failed to get request size: {e}")
            return None

    def _get_response_size(self, response: Response) -> Optional[int]:
        """응답 크기 추정"""
        try:
            # Response 객체에서 content-length 헤더 확인
            if hasattr(response, 'headers'):
                content_length = response.headers.get('content-length')
                if content_length:
                    return int(content_length)
            
            # body가 있는 경우 크기 계산
            if hasattr(response, 'body'):
                return len(response.body)
            
            return None
        
        except Exception as e:
            logger.debug(f"Failed to get response size: {e}")
            return None


class MetricsCollectionMiddleware(BaseHTTPMiddleware):
    """
    추가 메트릭 수집 미들웨어
    - 활성 사용자 추적
    - 세션 정보 수집
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.metrics = get_metrics_collector()
        self._active_requests = 0
        logger.info("MetricsCollectionMiddleware initialized")

    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """요청 처리 및 추가 메트릭 수집"""
        
        self._active_requests += 1
        
        try:
            # 사용자 정보 추출
            user_id = self._extract_user_id(request)
            if user_id:
                # 활성 사용자 카운트 업데이트는 별도 태스크에서 처리
                pass
            
            response = await call_next(request)
            return response
        
        finally:
            self._active_requests -= 1

    def _extract_user_id(self, request: Request) -> Optional[int]:
        """요청에서 사용자 ID 추출"""
        try:
            # Authorization 헤더에서 추출
            auth_header = request.headers.get('authorization')
            if auth_header:
                # JWT 토큰 파싱 로직 (간단한 버전)
                # 실제로는 JWT 디코딩 필요
                pass
            
            # 쿼리 파라미터에서 추출
            user_id = request.query_params.get('user_id')
            if user_id and user_id.isdigit():
                return int(user_id)
            
            # Request state에서 추출
            if hasattr(request.state, 'user'):
                user = request.state.user
                if isinstance(user, dict):
                    return user.get('user_id')
            
            return None
        
        except Exception as e:
            logger.debug(f"Failed to extract user_id: {e}")
            return None


def create_prometheus_middleware(
    app: ASGIApp,
    exclude_paths: Optional[List[str]] = None
) -> PrometheusMiddleware:
    """
    Prometheus 미들웨어 팩토리 함수
    
    Args:
        app: FastAPI 애플리케이션
        exclude_paths: 메트릭 수집에서 제외할 경로 리스트
    
    Returns:
        PrometheusMiddleware 인스턴스
    """
    default_excludes = [
        '/metrics',
        '/health/live',
        '/health/ready',
        '/docs',
        '/redoc',
        '/openapi.json',
        '/favicon.ico'
    ]
    
    if exclude_paths:
        default_excludes.extend(exclude_paths)
    
    return PrometheusMiddleware(app, exclude_paths=default_excludes)


if __name__ == "__main__":
    from config.logging_config import setup_logging
    setup_logging()
    
    print("\n" + "=" * 70)
    print("PROMETHEUS MIDDLEWARE TEST")
    print("=" * 70)
    
    # 테스트용 간단한 FastAPI 앱
    from fastapi import FastAPI
    
    app = FastAPI()
    
    # 미들웨어 추가
    middleware = create_prometheus_middleware(app)
    app.add_middleware(PrometheusMiddleware, exclude_paths=['/metrics'])
    
    @app.get("/test")
    async def test_endpoint():
        return {"message": "test"}
    
    print("\nMiddleware configured successfully")
    print("Excluded paths:", middleware.exclude_paths)
    print("\nPrometheus middleware test complete")