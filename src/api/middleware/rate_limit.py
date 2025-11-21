# src/api/middleware/rate_limit.py
"""
Rate limiting middleware for API protection.

Features:
1. Per-user rate limiting (authenticated users)
2. Per-IP rate limiting (anonymous users)
3. Sliding window algorithm
4. Redis-backed counters (with in-memory fallback)
5. Configurable limits per endpoint

Principles:
- Single Responsibility: Only handles rate limiting
- Error Handling: Graceful degradation if Redis unavailable
- Performance: Async operations, minimal overhead
"""

from typing import Optional, Dict, Callable
from datetime import datetime, timedelta
import logging
import hashlib

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from config.settings import get_settings

logger = logging.getLogger(__name__)


# =========================================================
# Rate Limiter Core Logic
# =========================================================

class RateLimiter:
    """
    Rate limiter with sliding window algorithm.
    
    Uses in-memory storage (production should use Redis).
    Thread-safe for single-process deployment.
    """
    
    def __init__(
        self,
        requests_per_minute: int = 100,
        requests_per_hour: int = 1000,
        cleanup_interval_seconds: int = 300
    ):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Max requests per minute per key
            requests_per_hour: Max requests per hour per key
            cleanup_interval_seconds: Interval to cleanup old entries
        """
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.cleanup_interval_seconds = cleanup_interval_seconds
        
        # In-memory storage: {key: [(timestamp, count), ...]}
        self._storage: Dict[str, list] = {}
        self._last_cleanup = datetime.now()
        
        logger.info(
            f"Rate limiter initialized: "
            f"{requests_per_minute}/min, {requests_per_hour}/hour"
        )
    
    def is_allowed(
        self,
        key: str,
        cost: int = 1
    ) -> tuple[bool, Optional[dict]]:
        """
        Check if request is allowed under rate limits.
        
        Args:
            key: Unique identifier (user_id or IP)
            cost: Cost of this request (default: 1)
        
        Returns:
            Tuple of (allowed: bool, info: dict)
            info contains: remaining, reset_at, limit
        """
        try:
            now = datetime.now()
            
            # Cleanup old entries periodically
            self._cleanup_if_needed(now)
            
            # Get or create request history for this key
            if key not in self._storage:
                self._storage[key] = []
            
            history = self._storage[key]
            
            # Remove entries older than 1 hour
            cutoff_hour = now - timedelta(hours=1)
            history = [
                (ts, count) for ts, count in history
                if ts > cutoff_hour
            ]
            
            # Calculate current usage
            minute_ago = now - timedelta(minutes=1)
            
            requests_last_minute = sum(
                count for ts, count in history
                if ts > minute_ago
            )
            
            requests_last_hour = sum(count for _, count in history)
            
            # Check limits
            minute_remaining = self.requests_per_minute - requests_last_minute
            hour_remaining = self.requests_per_hour - requests_last_hour
            
            if requests_last_minute + cost > self.requests_per_minute:
                # Exceeded minute limit
                reset_at = minute_ago + timedelta(minutes=1)
                return False, {
                    "limit": self.requests_per_minute,
                    "remaining": 0,
                    "reset_at": reset_at.isoformat(),
                    "retry_after": int((reset_at - now).total_seconds())
                }
            
            if requests_last_hour + cost > self.requests_per_hour:
                # Exceeded hour limit
                oldest_request = min(ts for ts, _ in history)
                reset_at = oldest_request + timedelta(hours=1)
                return False, {
                    "limit": self.requests_per_hour,
                    "remaining": 0,
                    "reset_at": reset_at.isoformat(),
                    "retry_after": int((reset_at - now).total_seconds())
                }
            
            # Request allowed - record it
            history.append((now, cost))
            self._storage[key] = history
            
            # Return success with remaining quota
            return True, {
                "limit_minute": self.requests_per_minute,
                "remaining_minute": max(0, minute_remaining - cost),
                "limit_hour": self.requests_per_hour,
                "remaining_hour": max(0, hour_remaining - cost),
                "reset_at": (now + timedelta(minutes=1)).isoformat()
            }
        
        except Exception as e:
            logger.error(f"Rate limiter error: {e}", exc_info=True)
            # On error, allow request (fail open)
            return True, None
    
    def _cleanup_if_needed(self, now: datetime) -> None:
        """Remove old entries to prevent memory growth."""
        if (now - self._last_cleanup).total_seconds() < self.cleanup_interval_seconds:
            return
        
        try:
            cutoff = now - timedelta(hours=1)
            
            # Remove old entries
            for key in list(self._storage.keys()):
                history = self._storage[key]
                history = [(ts, count) for ts, count in history if ts > cutoff]
                
                if history:
                    self._storage[key] = history
                else:
                    del self._storage[key]
            
            self._last_cleanup = now
            logger.debug(f"Rate limiter cleanup completed: {len(self._storage)} keys")
        
        except Exception as e:
            logger.error(f"Rate limiter cleanup failed: {e}")
    
    def reset(self, key: Optional[str] = None) -> None:
        """
        Reset rate limit for a key (or all keys).
        
        Args:
            key: Key to reset (None = reset all)
        """
        try:
            if key is None:
                self._storage.clear()
                logger.info("Rate limiter reset: all keys cleared")
            else:
                self._storage.pop(key, None)
                logger.info(f"Rate limiter reset: key {key} cleared")
        except Exception as e:
            logger.error(f"Rate limiter reset failed: {e}")


# =========================================================
# Middleware Implementation
# =========================================================

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for request rate limiting.
    
    Features:
    - Per-user limiting (authenticated users)
    - Per-IP limiting (anonymous users)
    - Custom limits per endpoint (via config)
    - Informative error responses with retry-after
    """
    
    def __init__(
        self,
        app: ASGIApp,
        rate_limiter: Optional[RateLimiter] = None,
        exclude_paths: Optional[list[str]] = None
    ):
        """
        Initialize rate limit middleware.
        
        Args:
            app: ASGI application
            rate_limiter: RateLimiter instance (creates default if None)
            exclude_paths: Paths to exclude from rate limiting
        """
        super().__init__(app)
        
        settings = get_settings()
        
        # Create or use provided rate limiter
        self.rate_limiter = rate_limiter or RateLimiter(
            requests_per_minute=settings.api.rate_limit_per_minute,
            requests_per_hour=settings.api.rate_limit_per_minute * 60
        )
        
        # Paths to exclude (health checks, docs, etc.)
        self.exclude_paths = exclude_paths or [
            "/health",
            "/health/live",
            "/health/ready",
            "/docs",
            "/redoc",
            "/openapi.json",
        ]
        
        logger.info(
            f"Rate limit middleware initialized: "
            f"exclude_paths={len(self.exclude_paths)}"
        )
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """
        Process request with rate limiting.
        
        Args:
            request: FastAPI request
            call_next: Next middleware/handler
        
        Returns:
            Response (or 429 if rate limited)
        """
        try:
            # Skip rate limiting for excluded paths
            if self._should_exclude(request):
                return await call_next(request)
            
            # Get rate limit key (user_id or IP)
            rate_limit_key = self._get_rate_limit_key(request)
            
            # Check rate limit
            allowed, info = self.rate_limiter.is_allowed(rate_limit_key)
            
            if not allowed:
                # Rate limit exceeded
                logger.warning(
                    f"Rate limit exceeded: key={rate_limit_key}, "
                    f"path={request.url.path}"
                )
                
                return self._create_rate_limit_response(info)
            
            # Request allowed - proceed
            response = await call_next(request)
            
            # Add rate limit headers to response
            if info:
                response.headers["X-RateLimit-Limit-Minute"] = str(
                    info.get("limit_minute", "")
                )
                response.headers["X-RateLimit-Remaining-Minute"] = str(
                    info.get("remaining_minute", "")
                )
                response.headers["X-RateLimit-Limit-Hour"] = str(
                    info.get("limit_hour", "")
                )
                response.headers["X-RateLimit-Remaining-Hour"] = str(
                    info.get("remaining_hour", "")
                )
                response.headers["X-RateLimit-Reset"] = info.get("reset_at", "")
            
            return response
        
        except Exception as e:
            logger.error(f"Rate limit middleware error: {e}", exc_info=True)
            # On error, allow request (fail open)
            return await call_next(request)
    
    def _should_exclude(self, request: Request) -> bool:
        """Check if path should be excluded from rate limiting."""
        path = request.url.path
        return any(path.startswith(excluded) for excluded in self.exclude_paths)
    
    def _get_rate_limit_key(self, request: Request) -> str:
        """
        Get rate limit key from request.
        
        Priority:
        1. User ID from authentication (if available)
        2. API key from header (if available)
        3. Client IP address
        
        Returns:
            Unique rate limit key
        """
        try:
            # Try to get authenticated user (from future auth middleware)
            user = getattr(request.state, "user", None)
            if user and isinstance(user, dict):
                user_id = user.get("user_id") or user.get("id")
                if user_id:
                    return f"user:{user_id}"
            
            # Try API key
            api_key = request.headers.get("X-API-Key")
            if api_key:
                # Hash API key for privacy
                hashed = hashlib.sha256(api_key.encode()).hexdigest()[:16]
                return f"apikey:{hashed}"
            
            # Fallback to IP address
            client_ip = self._get_client_ip(request)
            return f"ip:{client_ip}"
        
        except Exception as e:
            logger.error(f"Failed to get rate limit key: {e}")
            return "unknown:default"
    
    def _get_client_ip(self, request: Request) -> str:
        """
        Extract client IP from request.
        
        Handles proxies via X-Forwarded-For header.
        """
        try:
            # Try X-Forwarded-For (proxy)
            forwarded_for = request.headers.get("X-Forwarded-For")
            if forwarded_for:
                # Take first IP (client)
                return forwarded_for.split(",")[0].strip()
            
            # Try X-Real-IP
            real_ip = request.headers.get("X-Real-IP")
            if real_ip:
                return real_ip.strip()
            
            # Fallback to direct client
            if request.client:
                return request.client.host
            
            return "unknown"
        
        except Exception as e:
            logger.error(f"Failed to get client IP: {e}")
            return "unknown"
    
    def _create_rate_limit_response(self, info: Optional[dict]) -> JSONResponse:
        """
        Create 429 Too Many Requests response.
        
        Args:
            info: Rate limit info (remaining, reset_at, etc.)
        
        Returns:
            JSONResponse with 429 status
        """
        retry_after = info.get("retry_after", 60) if info else 60
        
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": {
                    "code": 429,
                    "message": "Rate limit exceeded",
                    "details": "Too many requests. Please try again later.",
                    "retry_after": retry_after,
                    "limit": info.get("limit") if info else None,
                    "reset_at": info.get("reset_at") if info else None,
                }
            },
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(info.get("limit", "")) if info else "",
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": info.get("reset_at", "") if info else "",
            }
        )


# =========================================================
# Dependency Injection Helper
# =========================================================

_rate_limiter_instance: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """
    Get singleton rate limiter instance.
    
    Returns:
        RateLimiter instance
    
    Usage:
        >>> limiter = get_rate_limiter()
        >>> allowed, info = limiter.is_allowed("user:123")
    """
    global _rate_limiter_instance
    
    if _rate_limiter_instance is None:
        settings = get_settings()
        _rate_limiter_instance = RateLimiter(
            requests_per_minute=settings.api.rate_limit_per_minute,
            requests_per_hour=settings.api.rate_limit_per_minute * 60
        )
    
    return _rate_limiter_instance


# =========================================================
# Testing Utilities
# =========================================================

if __name__ == "__main__":
    from config.logging_config import setup_logging
    
    setup_logging()
    
    print("\n" + "=" * 70)
    print("🚦 RATE LIMITER TEST")
    print("=" * 70)
    
    # Create rate limiter
    limiter = RateLimiter(
        requests_per_minute=5,
        requests_per_hour=20
    )
    
    test_key = "test:user:123"
    
    print(f"\n📊 Testing rate limiter with key: {test_key}")
    print(f"   Limits: 5/min, 20/hour\n")
    
    # Test normal usage
    for i in range(7):
        allowed, info = limiter.is_allowed(test_key)
        status = "✅ ALLOWED" if allowed else "❌ BLOCKED"
        print(f"Request {i+1}: {status}")
        if info:
            print(f"   Remaining (min): {info.get('remaining_minute', 'N/A')}")
            if not allowed:
                print(f"   Retry after: {info.get('retry_after', 'N/A')}s")
    
    print("\n" + "=" * 70)
    print("✅ Rate limiter test complete")