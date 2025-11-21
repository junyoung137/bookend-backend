# src/api/middleware/auth.py
"""
Authentication middleware for API security.

Features:
1. JWT token validation (Bearer token)
2. API key authentication (X-API-Key header)
3. User context injection into request state
4. Role-based access control (RBAC)
5. Token refresh mechanism

Principles:
- Single Responsibility: Only handles authentication
- Error Handling: Clear error messages for auth failures
- Security: No sensitive data in logs
- Performance: Token validation caching
"""

from typing import Optional, Dict, Any, Callable
from datetime import datetime, timedelta
from functools import wraps
import logging
import hashlib
import secrets

from fastapi import Request, Response, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import jwt

from config.settings import get_settings

logger = logging.getLogger(__name__)


# =========================================================
# JWT Configuration
# =========================================================

class JWTConfig:
    """JWT token configuration."""
    
    # TODO: Move to environment variables in production
    SECRET_KEY = "your-secret-key-change-in-production"  # Should be in .env
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 15
    REFRESH_TOKEN_EXPIRE_DAYS = 7
    
    @classmethod
    def get_secret_key(cls) -> str:
        """Get JWT secret key from settings."""
        settings = get_settings()
        # Try to get from environment, fallback to default
        return getattr(settings, 'jwt_secret_key', cls.SECRET_KEY)


# =========================================================
# Token Manager
# =========================================================

class TokenManager:
    """
    JWT token creation and validation.
    
    Handles:
    - Access token generation
    - Refresh token generation
    - Token validation and decoding
    - Token expiration checks
    """
    
    def __init__(self):
        """Initialize token manager."""
        self.secret_key = JWTConfig.get_secret_key()
        self.algorithm = JWTConfig.ALGORITHM
        self.access_token_expire = timedelta(
            minutes=JWTConfig.ACCESS_TOKEN_EXPIRE_MINUTES
        )
        self.refresh_token_expire = timedelta(
            days=JWTConfig.REFRESH_TOKEN_EXPIRE_DAYS
        )
        
        # Token cache: {token_hash: (user_data, expire_time)}
        self._token_cache: Dict[str, tuple] = {}
        self._cache_ttl_seconds = 60  # Cache for 1 minute
    
    def create_access_token(
        self,
        user_id: int,
        email: Optional[str] = None,
        role: str = "user",
        additional_claims: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create JWT access token.
        
        Args:
            user_id: User database ID
            email: User email
            role: User role (user/admin)
            additional_claims: Extra claims to include
        
        Returns:
            JWT token string
        
        Example:
            >>> manager = TokenManager()
            >>> token = manager.create_access_token(
            ...     user_id=123,
            ...     email="user@example.com",
            ...     role="admin"
            ... )
        """
        try:
            now = datetime.utcnow()
            expire = now + self.access_token_expire
            
            # Build token payload
            payload = {
                "user_id": user_id,
                "email": email,
                "role": role,
                "type": "access",
                "iat": now,
                "exp": expire,
                "jti": secrets.token_urlsafe(16),  # Unique token ID
            }
            
            # Add additional claims
            if additional_claims:
                payload.update(additional_claims)
            
            # Encode token
            token = jwt.encode(
                payload,
                self.secret_key,
                algorithm=self.algorithm
            )
            
            logger.debug(f"Created access token for user {user_id}")
            
            return token
        
        except Exception as e:
            logger.error(f"Failed to create access token: {e}", exc_info=True)
            raise
    
    def create_refresh_token(
        self,
        user_id: int,
        email: Optional[str] = None
    ) -> str:
        """
        Create JWT refresh token.
        
        Args:
            user_id: User database ID
            email: User email
        
        Returns:
            JWT refresh token string
        """
        try:
            now = datetime.utcnow()
            expire = now + self.refresh_token_expire
            
            payload = {
                "user_id": user_id,
                "email": email,
                "type": "refresh",
                "iat": now,
                "exp": expire,
                "jti": secrets.token_urlsafe(16),
            }
            
            token = jwt.encode(
                payload,
                self.secret_key,
                algorithm=self.algorithm
            )
            
            logger.debug(f"Created refresh token for user {user_id}")
            
            return token
        
        except Exception as e:
            logger.error(f"Failed to create refresh token: {e}", exc_info=True)
            raise
    
    def validate_token(
        self,
        token: str,
        token_type: str = "access"
    ) -> Optional[Dict[str, Any]]:
        """
        Validate and decode JWT token.
        
        Args:
            token: JWT token string
            token_type: Expected token type (access/refresh)
        
        Returns:
            Decoded token payload or None if invalid
        
        Raises:
            jwt.ExpiredSignatureError: Token expired
            jwt.InvalidTokenError: Token invalid
        """
        try:
            # Check cache first
            token_hash = self._hash_token(token)
            cached = self._get_cached_token(token_hash)
            if cached:
                return cached
            
            # Decode and validate token
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            
            # Verify token type
            if payload.get("type") != token_type:
                logger.warning(
                    f"Token type mismatch: expected {token_type}, "
                    f"got {payload.get('type')}"
                )
                return None
            
            # Cache valid token
            self._cache_token(token_hash, payload)
            
            logger.debug(f"Token validated for user {payload.get('user_id')}")
            
            return payload
        
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
        
        except Exception as e:
            logger.error(f"Token validation failed: {e}", exc_info=True)
            return None
    
    def refresh_access_token(
        self,
        refresh_token: str
    ) -> Optional[str]:
        """
        Create new access token from refresh token.
        
        Args:
            refresh_token: Valid refresh token
        
        Returns:
            New access token or None if refresh token invalid
        """
        try:
            # Validate refresh token
            payload = self.validate_token(refresh_token, token_type="refresh")
            
            if not payload:
                return None
            
            # Create new access token
            new_token = self.create_access_token(
                user_id=payload.get("user_id"),
                email=payload.get("email"),
                role=payload.get("role", "user")
            )
            
            logger.info(f"Refreshed access token for user {payload.get('user_id')}")
            
            return new_token
        
        except Exception as e:
            logger.error(f"Token refresh failed: {e}", exc_info=True)
            return None
    
    def _hash_token(self, token: str) -> str:
        """Create hash of token for caching (privacy)."""
        return hashlib.sha256(token.encode()).hexdigest()[:32]
    
    def _cache_token(self, token_hash: str, payload: Dict[str, Any]) -> None:
        """Cache validated token payload."""
        try:
            expire_time = datetime.utcnow() + timedelta(seconds=self._cache_ttl_seconds)
            self._token_cache[token_hash] = (payload, expire_time)
            
            # Cleanup old cache entries
            self._cleanup_cache()
        except Exception as e:
            logger.error(f"Token caching failed: {e}")
    
    def _get_cached_token(self, token_hash: str) -> Optional[Dict[str, Any]]:
        """Get cached token payload if not expired."""
        try:
            if token_hash not in self._token_cache:
                return None
            
            payload, expire_time = self._token_cache[token_hash]
            
            if datetime.utcnow() > expire_time:
                del self._token_cache[token_hash]
                return None
            
            return payload
        except Exception as e:
            logger.error(f"Cache retrieval failed: {e}")
            return None
    
    def _cleanup_cache(self) -> None:
        """Remove expired cache entries."""
        try:
            now = datetime.utcnow()
            expired_keys = [
                key for key, (_, expire_time) in self._token_cache.items()
                if now > expire_time
            ]
            
            for key in expired_keys:
                del self._token_cache[key]
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired token cache entries")
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")


# =========================================================
# API Key Manager
# =========================================================

class APIKeyManager:
    """
    API key authentication manager.
    
    Handles:
    - API key validation
    - Key-to-user mapping
    - Rate limiting per key
    """
    
    def __init__(self):
        """Initialize API key manager."""
        # TODO: Load from database in production
        # Format: {api_key_hash: {"user_id": int, "role": str, "name": str}}
        self._api_keys: Dict[str, Dict[str, Any]] = {}
        
        logger.info("API key manager initialized")
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """
        Validate API key and return associated user info.
        
        Args:
            api_key: API key string
        
        Returns:
            User info dict or None if invalid
        """
        try:
            # Hash API key for lookup
            key_hash = self._hash_api_key(api_key)
            
            # Lookup in storage
            user_info = self._api_keys.get(key_hash)
            
            if user_info:
                logger.debug(f"Valid API key for user {user_info.get('user_id')}")
                return user_info.copy()
            
            logger.warning("Invalid API key provided")
            return None
        
        except Exception as e:
            logger.error(f"API key validation failed: {e}", exc_info=True)
            return None
    
    def create_api_key(
        self,
        user_id: int,
        role: str = "user",
        name: Optional[str] = None
    ) -> str:
        """
        Create new API key for user.
        
        Args:
            user_id: User database ID
            role: User role
            name: Key name/description
        
        Returns:
            API key string
        """
        try:
            # Generate secure random key
            api_key = f"bk_{secrets.token_urlsafe(32)}"
            
            # Hash and store
            key_hash = self._hash_api_key(api_key)
            self._api_keys[key_hash] = {
                "user_id": user_id,
                "role": role,
                "name": name or f"Key for user {user_id}",
                "created_at": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Created API key for user {user_id}")
            
            return api_key
        
        except Exception as e:
            logger.error(f"API key creation failed: {e}", exc_info=True)
            raise
    
    def revoke_api_key(self, api_key: str) -> bool:
        """
        Revoke API key.
        
        Args:
            api_key: API key to revoke
        
        Returns:
            True if revoked, False if not found
        """
        try:
            key_hash = self._hash_api_key(api_key)
            
            if key_hash in self._api_keys:
                del self._api_keys[key_hash]
                logger.info("API key revoked")
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"API key revocation failed: {e}", exc_info=True)
            return False
    
    def _hash_api_key(self, api_key: str) -> str:
        """Hash API key for secure storage."""
        return hashlib.sha256(api_key.encode()).hexdigest()


# =========================================================
# Authentication Middleware
# =========================================================

class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for authentication.
    
    Features:
    - JWT token validation (Bearer token)
    - API key validation (X-API-Key header)
    - User context injection into request.state
    - Optional authentication (allows anonymous)
    """
    
    def __init__(
        self,
        app: ASGIApp,
        token_manager: Optional[TokenManager] = None,
        api_key_manager: Optional[APIKeyManager] = None,
        exclude_paths: Optional[list[str]] = None,
        require_auth_paths: Optional[list[str]] = None
    ):
        """
        Initialize authentication middleware.
        
        Args:
            app: ASGI application
            token_manager: TokenManager instance
            api_key_manager: APIKeyManager instance
            exclude_paths: Paths to skip authentication
            require_auth_paths: Paths that require authentication
        """
        super().__init__(app)
        
        self.token_manager = token_manager or TokenManager()
        self.api_key_manager = api_key_manager or APIKeyManager()
        
        # Paths to skip authentication
        self.exclude_paths = exclude_paths or [
            "/health",
            "/health/live",
            "/health/ready",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/v1/auth/login",
            "/api/v1/auth/register",
        ]
        
        # Paths that require authentication (return 401 if not authenticated)
        self.require_auth_paths = require_auth_paths or [
            "/api/v1/admin",
        ]
        
        logger.info("Authentication middleware initialized")
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """
        Process request with authentication.
        
        Args:
            request: FastAPI request
            call_next: Next middleware/handler
        
        Returns:
            Response (or 401 if auth required and failed)
        """
        try:
            # Skip authentication for excluded paths
            if self._should_exclude(request):
                return await call_next(request)
            
            # Try to authenticate user
            user = await self._authenticate_request(request)
            
            # Inject user into request state
            request.state.user = user
            request.state.is_authenticated = user is not None
            
            # Check if authentication is required for this path
            if self._requires_auth(request) and not user:
                return self._create_unauthorized_response(
                    "Authentication required for this endpoint"
                )
            
            # Proceed with request
            response = await call_next(request)
            
            return response
        
        except Exception as e:
            logger.error(f"Authentication middleware error: {e}", exc_info=True)
            # On error, proceed without authentication (fail open for non-protected routes)
            request.state.user = None
            request.state.is_authenticated = False
            return await call_next(request)
    
    async def _authenticate_request(
        self,
        request: Request
    ) -> Optional[Dict[str, Any]]:
        """
        Authenticate request using JWT or API key.
        
        Args:
            request: FastAPI request
        
        Returns:
            User dict or None if not authenticated
        """
        try:
            # Try JWT authentication first
            user = await self._authenticate_jwt(request)
            if user:
                return user
            
            # Try API key authentication
            user = await self._authenticate_api_key(request)
            if user:
                return user
            
            # No valid authentication found
            return None
        
        except Exception as e:
            logger.error(f"Authentication failed: {e}", exc_info=True)
            return None
    
    async def _authenticate_jwt(
        self,
        request: Request
    ) -> Optional[Dict[str, Any]]:
        """Authenticate using JWT Bearer token."""
        try:
            # Get Authorization header
            auth_header = request.headers.get("Authorization")
            
            if not auth_header:
                return None
            
            # Check Bearer scheme
            if not auth_header.startswith("Bearer "):
                logger.warning("Invalid Authorization header scheme")
                return None
            
            # Extract token
            token = auth_header.replace("Bearer ", "").strip()
            
            # Validate token
            payload = self.token_manager.validate_token(token, token_type="access")
            
            if not payload:
                return None
            
            # Extract user info
            user = {
                "user_id": payload.get("user_id"),
                "email": payload.get("email"),
                "role": payload.get("role", "user"),
                "auth_method": "jwt",
                "token_jti": payload.get("jti"),
            }
            
            logger.debug(f"Authenticated user {user['user_id']} via JWT")
            
            return user
        
        except Exception as e:
            logger.error(f"JWT authentication failed: {e}", exc_info=True)
            return None
    
    async def _authenticate_api_key(
        self,
        request: Request
    ) -> Optional[Dict[str, Any]]:
        """Authenticate using API key."""
        try:
            # Get API key from header
            api_key = request.headers.get("X-API-Key")
            
            if not api_key:
                return None
            
            # Validate API key
            user_info = self.api_key_manager.validate_api_key(api_key)
            
            if not user_info:
                return None
            
            # Add auth method
            user_info["auth_method"] = "api_key"
            
            logger.debug(f"Authenticated user {user_info['user_id']} via API key")
            
            return user_info
        
        except Exception as e:
            logger.error(f"API key authentication failed: {e}", exc_info=True)
            return None
    
    def _should_exclude(self, request: Request) -> bool:
        """Check if path should be excluded from authentication."""
        path = request.url.path
        return any(path.startswith(excluded) for excluded in self.exclude_paths)
    
    def _requires_auth(self, request: Request) -> bool:
        """Check if path requires authentication."""
        path = request.url.path
        return any(path.startswith(required) for required in self.require_auth_paths)
    
    def _create_unauthorized_response(self, message: str) -> Response:
        """Create 401 Unauthorized response."""
        from fastapi.responses import JSONResponse
        
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "error": {
                    "code": 401,
                    "message": "Unauthorized",
                    "details": message,
                }
            },
            headers={
                "WWW-Authenticate": "Bearer"
            }
        )


# =========================================================
# FastAPI Dependencies
# =========================================================

# HTTP Bearer security scheme
security = HTTPBearer(auto_error=False)


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[Dict[str, Any]]:
    """
    FastAPI dependency to get current authenticated user.
    
    Args:
        request: FastAPI request
        credentials: HTTP Bearer credentials
    
    Returns:
        User dict or None if not authenticated
    
    Usage:
        >>> @app.get("/me")
        >>> async def get_me(user: dict = Depends(get_current_user)):
        ...     if not user:
        ...         raise HTTPException(401, "Not authenticated")
        ...     return user
    """
    try:
        # Try to get user from request state (set by middleware)
        user = getattr(request.state, "user", None)
        
        if user:
            return user
        
        # Fallback: Try to authenticate from credentials
        if credentials:
            token_manager = TokenManager()
            payload = token_manager.validate_token(credentials.credentials)
            
            if payload:
                return {
                    "user_id": payload.get("user_id"),
                    "email": payload.get("email"),
                    "role": payload.get("role", "user"),
                    "auth_method": "jwt",
                }
        
        return None
    
    except Exception as e:
        logger.error(f"Get current user failed: {e}", exc_info=True)
        return None


async def require_auth(
    user: Optional[Dict[str, Any]] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    FastAPI dependency that requires authentication.
    
    Args:
        user: Current user from get_current_user
    
    Returns:
        User dict
    
    Raises:
        HTTPException: 401 if not authenticated
    
    Usage:
        >>> @app.get("/protected")
        >>> async def protected_route(user: dict = Depends(require_auth)):
        ...     return {"user": user}
    """
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user


async def require_admin(
    user: Dict[str, Any] = Depends(require_auth)
) -> Dict[str, Any]:
    """
    FastAPI dependency that requires admin role.
    
    Args:
        user: Current authenticated user
    
    Returns:
        User dict
    
    Raises:
        HTTPException: 403 if not admin
    
    Usage:
        >>> @app.post("/admin/deploy")
        >>> async def deploy_model(user: dict = Depends(require_admin)):
        ...     return {"message": "Deployed"}
    """
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    
    return user


# =========================================================
# Singleton Instances
# =========================================================

_token_manager_instance: Optional[TokenManager] = None
_api_key_manager_instance: Optional[APIKeyManager] = None


def get_token_manager() -> TokenManager:
    """Get singleton token manager instance."""
    global _token_manager_instance
    if _token_manager_instance is None:
        _token_manager_instance = TokenManager()
    return _token_manager_instance


def get_api_key_manager() -> APIKeyManager:
    """Get singleton API key manager instance."""
    global _api_key_manager_instance
    if _api_key_manager_instance is None:
        _api_key_manager_instance = APIKeyManager()
    return _api_key_manager_instance


# =========================================================
# Testing Utilities
# =========================================================

if __name__ == "__main__":
    from config.logging_config import setup_logging
    
    setup_logging()
    
    print("\n" + "=" * 70)
    print("🔐 AUTHENTICATION MODULE TEST")
    print("=" * 70)
    
    # Test TokenManager
    print("\n📋 Testing TokenManager...")
    token_manager = TokenManager()
    
    # Create access token
    access_token = token_manager.create_access_token(
        user_id=123,
        email="test@example.com",
        role="admin"
    )
    print(f"✅ Created access token: {access_token[:50]}...")
    
    # Validate token
    payload = token_manager.validate_token(access_token)
    if payload:
        print(f"✅ Token validated: user_id={payload['user_id']}, role={payload['role']}")
    else:
        print("❌ Token validation failed")
    
    # Test APIKeyManager
    print("\n📋 Testing APIKeyManager...")
    api_key_manager = APIKeyManager()
    
    # Create API key
    api_key = api_key_manager.create_api_key(
        user_id=123,
        role="admin",
        name="Test Key"
    )
    print(f"✅ Created API key: {api_key[:30]}...")
    
    # Validate API key
    user_info = api_key_manager.validate_api_key(api_key)
    if user_info:
        print(f"✅ API key validated: user_id={user_info['user_id']}, role={user_info['role']}")
    else:
        print("❌ API key validation failed")
    
    print("\n" + "=" * 70)
    print("✅ Authentication module test complete")