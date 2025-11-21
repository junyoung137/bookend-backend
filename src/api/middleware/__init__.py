# package middleware
# src/api/middleware/__init__.py
"""
API Middleware package for Bookend Recommendation System.

Provides:
- Rate limiting (per-user and per-IP)
- Authentication (JWT validation)
- Request/response logging
- Error tracking
"""

from .rate_limit import RateLimitMiddleware, get_rate_limiter
from .auth import AuthenticationMiddleware, get_current_user, require_auth
from .logging import LoggingMiddleware

__all__ = [
    "RateLimitMiddleware",
    "get_rate_limiter",
    "AuthenticationMiddleware",
    "get_current_user",
    "require_auth",
    "LoggingMiddleware",
]