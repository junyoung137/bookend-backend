# src/api/middleware/logging.py
"""
Logging middleware for request/response tracking and debugging.

Features:
1. Request logging (method, path, headers, body)
2. Response logging (status, headers, body)
3. Performance timing (request duration)
4. Error tracking and correlation
5. User context tracking
6. Request ID generation for tracing

Principles:
- Single Responsibility: Only handles logging
- Error Handling: Never fails requests due to logging errors
- Performance: Async operations, minimal overhead
- Privacy: Masks sensitive data (passwords, tokens)
"""

from typing import Callable, Optional, Dict, Any, Set
from datetime import datetime
import logging
import time
import json
import uuid
import re

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from starlette.datastructures import Headers

logger = logging.getLogger(__name__)


# =========================================================
# Sensitive Data Masking
# =========================================================

class DataMasker:
    """
    Mask sensitive data in logs.
    
    Handles:
    - Password fields
    - API keys and tokens
    - Credit card numbers
    - Email addresses (partial masking)
    """
    
    # Sensitive field patterns
    SENSITIVE_FIELDS = {
        "password",
        "passwd",
        "pwd",
        "secret",
        "api_key",
        "apikey",
        "token",
        "access_token",
        "refresh_token",
        "authorization",
        "auth",
        "credit_card",
        "card_number",
        "cvv",
        "ssn",
    }
    
    # Sensitive header patterns
    SENSITIVE_HEADERS = {
        "authorization",
        "x-api-key",
        "cookie",
        "set-cookie",
    }
    
    @classmethod
    def mask_dict(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively mask sensitive fields in dictionary.
        
        Args:
            data: Dictionary to mask
        
        Returns:
            Masked dictionary copy
        """
        if not isinstance(data, dict):
            return data
        
        masked = {}
        
        for key, value in data.items():
            key_lower = key.lower()
            
            # Check if key is sensitive
            if any(sensitive in key_lower for sensitive in cls.SENSITIVE_FIELDS):
                masked[key] = "***MASKED***"
            
            # Recursively mask nested dicts
            elif isinstance(value, dict):
                masked[key] = cls.mask_dict(value)
            
            # Recursively mask lists
            elif isinstance(value, list):
                masked[key] = [
                    cls.mask_dict(item) if isinstance(item, dict) else item
                    for item in value
                ]
            
            # Keep non-sensitive values
            else:
                masked[key] = value
        
        return masked
    
    @classmethod
    def mask_headers(cls, headers: Headers) -> Dict[str, str]:
        """
        Mask sensitive headers.
        
        Args:
            headers: Request/response headers
        
        Returns:
            Masked headers dictionary
        """
        masked = {}
        
        for key, value in headers.items():
            key_lower = key.lower()
            
            if key_lower in cls.SENSITIVE_HEADERS:
                # Partial masking for authorization (show scheme only)
                if key_lower == "authorization" and value.startswith("Bearer "):
                    masked[key] = f"Bearer ***{value[-8:]}"
                else:
                    masked[key] = "***MASKED***"
            else:
                masked[key] = value
        
        return masked
    
    @classmethod
    def mask_email(cls, email: str) -> str:
        """
        Partially mask email address.
        
        Args:
            email: Email address
        
        Returns:
            Masked email (e.g., u***r@example.com)
        """
        if not email or "@" not in email:
            return email
        
        local, domain = email.split("@", 1)
        
        if len(local) <= 2:
            masked_local = "*" * len(local)
        else:
            masked_local = local[0] + "*" * (len(local) - 2) + local[-1]
        
        return f"{masked_local}@{domain}"


# =========================================================
# Request Logger
# =========================================================

class RequestLogger:
    """
    Structured request/response logger.
    
    Logs:
    - Request details (method, path, headers, body)
    - Response details (status, headers, body)
    - Timing information (duration)
    - User context (if authenticated)
    - Errors and exceptions
    """
    
    def __init__(
        self,
        log_request_body: bool = True,
        log_response_body: bool = False,
        max_body_length: int = 1000,
        exclude_paths: Optional[Set[str]] = None
    ):
        """
        Initialize request logger.
        
        Args:
            log_request_body: Whether to log request body
            log_response_body: Whether to log response body
            max_body_length: Max body length to log (truncate if longer)
            exclude_paths: Paths to exclude from logging
        """
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.max_body_length = max_body_length
        
        # Exclude health checks and static assets
        self.exclude_paths = exclude_paths or {
            "/health",
            "/health/live",
            "/health/ready",
            "/metrics",
        }
    
    def should_log(self, path: str) -> bool:
        """Check if path should be logged."""
        return not any(path.startswith(excluded) for excluded in self.exclude_paths)
    
    async def log_request(
        self,
        request: Request,
        request_id: str,
        body: Optional[bytes] = None
    ) -> None:
        """
        Log incoming request.
        
        Args:
            request: FastAPI request
            request_id: Unique request ID
            body: Request body bytes (optional)
        """
        try:
            # Build log entry
            log_data = {
                "type": "request",
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
                "method": request.method,
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "headers": DataMasker.mask_headers(request.headers),
                "client": {
                    "host": request.client.host if request.client else None,
                    "port": request.client.port if request.client else None,
                },
            }
            
            # Add user context if authenticated
            user = getattr(request.state, "user", None)
            if user:
                log_data["user"] = {
                    "user_id": user.get("user_id"),
                    "email": DataMasker.mask_email(user.get("email", "")),
                    "role": user.get("role"),
                }
            
            # Add request body if enabled
            if self.log_request_body and body:
                try:
                    # Try to decode as JSON
                    body_str = body.decode("utf-8")
                    
                    # Truncate if too long
                    if len(body_str) > self.max_body_length:
                        body_str = body_str[:self.max_body_length] + "...[truncated]"
                    
                    # Try to parse as JSON for masking
                    try:
                        body_json = json.loads(body_str)
                        log_data["body"] = DataMasker.mask_dict(body_json)
                    except json.JSONDecodeError:
                        log_data["body"] = body_str
                
                except UnicodeDecodeError:
                    log_data["body"] = f"<binary data: {len(body)} bytes>"
            
            # Log as JSON
            logger.info(
                f"API Request: {request.method} {request.url.path}",
                extra={"json_data": log_data}
            )
        
        except Exception as e:
            logger.error(f"Failed to log request: {e}", exc_info=True)
    
    async def log_response(
        self,
        request: Request,
        response: Response,
        request_id: str,
        duration_ms: float,
        error: Optional[Exception] = None
    ) -> None:
        """
        Log outgoing response.
        
        Args:
            request: FastAPI request
            response: Response object
            request_id: Unique request ID
            duration_ms: Request processing duration
            error: Exception if error occurred
        """
        try:
            # Build log entry
            log_data = {
                "type": "response",
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
                "headers": DataMasker.mask_headers(response.headers),
            }
            
            # Add error info if present
            if error:
                log_data["error"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
            
            # Log level based on status code
            if response.status_code >= 500:
                log_level = logging.ERROR
            elif response.status_code >= 400:
                log_level = logging.WARNING
            else:
                log_level = logging.INFO
            
            # Log message
            message = (
                f"API Response: {request.method} {request.url.path} "
                f"-> {response.status_code} ({duration_ms:.2f}ms)"
            )
            
            logger.log(
                log_level,
                message,
                extra={"json_data": log_data}
            )
        
        except Exception as e:
            logger.error(f"Failed to log response: {e}", exc_info=True)


# =========================================================
# Logging Middleware
# =========================================================

class LoggingMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for comprehensive request/response logging.
    
    Features:
    - Request/response logging with masking
    - Request ID generation and propagation
    - Performance timing
    - Error tracking
    - Structured logging (JSON)
    """
    
    def __init__(
        self,
        app: ASGIApp,
        request_logger: Optional[RequestLogger] = None,
        exclude_paths: Optional[Set[str]] = None
    ):
        """
        Initialize logging middleware.
        
        Args:
            app: ASGI application
            request_logger: RequestLogger instance
            exclude_paths: Paths to exclude from logging
        """
        super().__init__(app)
        
        self.request_logger = request_logger or RequestLogger(
            log_request_body=True,
            log_response_body=False,
            max_body_length=1000
        )
        
        self.exclude_paths = exclude_paths or {
            "/health",
            "/health/live",
            "/health/ready",
            "/metrics",
        }
        
        logger.info("Logging middleware initialized")
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """
        Process request with logging.
        
        Args:
            request: FastAPI request
            call_next: Next middleware/handler
        
        Returns:
            Response
        """
        # Generate unique request ID
        request_id = self._generate_request_id(request)
        
        # Attach request ID to request state
        request.state.request_id = request_id
        
        # Skip logging for excluded paths
        if not self._should_log(request):
            return await call_next(request)
        
        # Record start time
        start_time = time.time()
        
        # Read request body (if needed for logging)
        body = None
        if self.request_logger.log_request_body:
            try:
                body = await request.body()
                # Important: Store body in request state for route handlers to access
                request.state.body = body
            except Exception as e:
                logger.warning(f"Failed to read request body: {e}")
        
        # Log request
        try:
            await self.request_logger.log_request(
                request,
                request_id,
                body
            )
        except Exception as e:
            logger.error(f"Request logging failed: {e}", exc_info=True)
        
        # Process request
        error = None
        try:
            response = await call_next(request)
        except Exception as e:
            error = e
            # Create error response
            from fastapi.responses import JSONResponse
            response = JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code": 500,
                        "message": "Internal server error",
                        "request_id": request_id,
                    }
                }
            )
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        # Add timing header
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        
        # Log response
        try:
            await self.request_logger.log_response(
                request,
                response,
                request_id,
                duration_ms,
                error
            )
        except Exception as e:
            logger.error(f"Response logging failed: {e}", exc_info=True)
        
        return response
    
    def _generate_request_id(self, request: Request) -> str:
        """
        Generate unique request ID.
        
        Priority:
        1. X-Request-ID header (if provided by proxy/client)
        2. Generate new UUID
        
        Args:
            request: FastAPI request
        
        Returns:
            Request ID string
        """
        # Check if request ID provided by client/proxy
        existing_id = request.headers.get("X-Request-ID")
        if existing_id:
            return existing_id
        
        # Generate new UUID
        return str(uuid.uuid4())
    
    def _should_log(self, request: Request) -> bool:
        """Check if request should be logged."""
        path = request.url.path
        return self.request_logger.should_log(path)


# =========================================================
# Request Context Logger (for use in route handlers)
# =========================================================

class RequestContextLogger:
    """
    Logger that automatically includes request context.
    
    Usage in route handlers:
        >>> from src.api.middleware.logging import get_request_logger
        >>> 
        >>> @app.get("/users")
        >>> async def get_users(request: Request):
        ...     logger = get_request_logger(request)
        ...     logger.info("Fetching users")
        ...     # Logs will include request_id automatically
    """
    
    def __init__(self, request: Request, base_logger: logging.Logger = None):
        """
        Initialize request context logger.
        
        Args:
            request: FastAPI request
            base_logger: Base logger to wrap
        """
        self.request = request
        self.base_logger = base_logger or logger
        self.request_id = getattr(request.state, "request_id", None)
        self.user = getattr(request.state, "user", None)
    
    def _add_context(self, msg: str) -> tuple[str, dict]:
        """Add request context to log message."""
        extra = {
            "request_id": self.request_id,
        }
        
        if self.user:
            extra["user_id"] = self.user.get("user_id")
        
        return msg, {"extra": extra}
    
    def debug(self, msg: str, *args, **kwargs):
        """Log debug message with context."""
        msg, context = self._add_context(msg)
        kwargs.update(context)
        self.base_logger.debug(msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        """Log info message with context."""
        msg, context = self._add_context(msg)
        kwargs.update(context)
        self.base_logger.info(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        """Log warning message with context."""
        msg, context = self._add_context(msg)
        kwargs.update(context)
        self.base_logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        """Log error message with context."""
        msg, context = self._add_context(msg)
        kwargs.update(context)
        self.base_logger.error(msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        """Log critical message with context."""
        msg, context = self._add_context(msg)
        kwargs.update(context)
        self.base_logger.critical(msg, *args, **kwargs)


def get_request_logger(request: Request) -> RequestContextLogger:
    """
    Get logger with request context.
    
    Args:
        request: FastAPI request
    
    Returns:
        RequestContextLogger instance
    
    Usage:
        >>> @app.get("/users")
        >>> async def get_users(request: Request):
        ...     logger = get_request_logger(request)
        ...     logger.info("Processing request")
    """
    return RequestContextLogger(request)


# =========================================================
# Performance Monitoring
# =========================================================

class PerformanceMonitor:
    """
    Track and log performance metrics.
    
    Features:
    - Endpoint response time tracking
    - Slow query detection
    - Performance degradation alerts
    """
    
    def __init__(self, slow_threshold_ms: float = 1000):
        """
        Initialize performance monitor.
        
        Args:
            slow_threshold_ms: Threshold for slow request warning
        """
        self.slow_threshold_ms = slow_threshold_ms
        
        # Track endpoint performance: {endpoint: [durations]}
        self._performance_data: Dict[str, list] = {}
    
    def record_request(
        self,
        method: str,
        path: str,
        duration_ms: float,
        status_code: int
    ) -> None:
        """
        Record request performance.
        
        Args:
            method: HTTP method
            path: Request path
            duration_ms: Request duration
            status_code: Response status code
        """
        try:
            # Create endpoint key
            endpoint = f"{method} {path}"
            
            # Initialize list if needed
            if endpoint not in self._performance_data:
                self._performance_data[endpoint] = []
            
            # Store duration
            self._performance_data[endpoint].append(duration_ms)
            
            # Keep only last 100 requests per endpoint
            if len(self._performance_data[endpoint]) > 100:
                self._performance_data[endpoint] = self._performance_data[endpoint][-100:]
            
            # Check for slow requests
            if duration_ms > self.slow_threshold_ms:
                logger.warning(
                    f"Slow request detected: {endpoint} took {duration_ms:.2f}ms "
                    f"(threshold: {self.slow_threshold_ms}ms)",
                    extra={
                        "endpoint": endpoint,
                        "duration_ms": duration_ms,
                        "status_code": status_code,
                    }
                )
        
        except Exception as e:
            logger.error(f"Performance recording failed: {e}", exc_info=True)
    
    def get_endpoint_stats(self, endpoint: str) -> Optional[Dict[str, float]]:
        """
        Get performance statistics for endpoint.
        
        Args:
            endpoint: Endpoint string (e.g., "GET /users")
        
        Returns:
            Stats dict with avg, min, max, p95
        """
        try:
            durations = self._performance_data.get(endpoint)
            
            if not durations:
                return None
            
            sorted_durations = sorted(durations)
            count = len(sorted_durations)
            
            return {
                "count": count,
                "avg": sum(sorted_durations) / count,
                "min": sorted_durations[0],
                "max": sorted_durations[-1],
                "p50": sorted_durations[int(count * 0.5)],
                "p95": sorted_durations[int(count * 0.95)],
                "p99": sorted_durations[int(count * 0.99)],
            }
        
        except Exception as e:
            logger.error(f"Stats calculation failed: {e}", exc_info=True)
            return None


# =========================================================
# Singleton Instances
# =========================================================

_performance_monitor_instance: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get singleton performance monitor instance."""
    global _performance_monitor_instance
    if _performance_monitor_instance is None:
        _performance_monitor_instance = PerformanceMonitor(slow_threshold_ms=1000)
    return _performance_monitor_instance


# =========================================================
# Testing Utilities
# =========================================================

if __name__ == "__main__":
    from config.logging_config import setup_logging
    
    setup_logging()
    
    print("\n" + "=" * 70)
    print("📝 LOGGING MIDDLEWARE TEST")
    print("=" * 70)
    
    # Test DataMasker
    print("\n🔒 Testing DataMasker...")
    
    test_data = {
        "username": "testuser",
        "password": "secret123",
        "email": "user@example.com",
        "api_key": "abc123xyz",
        "metadata": {
            "token": "bearer_token_here",
            "preferences": "dark_mode"
        }
    }
    
    masked = DataMasker.mask_dict(test_data)
    print(f"Original: {test_data}")
    print(f"Masked: {masked}")
    
    # Test email masking
    print("\n📧 Testing email masking...")
    emails = [
        "user@example.com",
        "ab@test.com",
        "verylongemail@example.com"
    ]
    
    for email in emails:
        masked_email = DataMasker.mask_email(email)
        print(f"  {email} -> {masked_email}")
    
    # Test PerformanceMonitor
    print("\n⏱️ Testing PerformanceMonitor...")
    monitor = PerformanceMonitor(slow_threshold_ms=100)
    
    # Record some requests
    monitor.record_request("GET", "/users", 50, 200)
    monitor.record_request("GET", "/users", 150, 200)  # Slow
    monitor.record_request("GET", "/users", 75, 200)
    
    # Get stats
    stats = monitor.get_endpoint_stats("GET /users")
    if stats:
        print(f"  Endpoint stats:")
        print(f"    Count: {stats['count']}")
        print(f"    Avg: {stats['avg']:.2f}ms")
        print(f"    P95: {stats['p95']:.2f}ms")
    
    print("\n" + "=" * 70)
    print("✅ Logging middleware test complete")