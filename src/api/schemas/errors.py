# src/api/schemas/errors.py
"""
Error schemas for Bookend Recommendation API.

Provides Pydantic models for:
- Standard error responses
- Validation errors
- Service errors
- Custom exceptions

Principles:
- Consistent Format: All errors follow same structure
- Rich Context: Include details for debugging
- Client-Friendly: Clear messages for end users
"""

from typing import List, Dict, Any, Optional
from datetime import datetime

from pydantic import BaseModel, Field, ConfigDict
from fastapi import HTTPException, status


# =========================================================
# Base Error Response Schema
# =========================================================

class ErrorDetail(BaseModel):
    """Individual error detail."""
    
    field: Optional[str] = Field(
        default=None,
        description="Field name if validation error"
    )
    
    message: str = Field(
        ...,
        description="Error message"
    )
    
    error_type: Optional[str] = Field(
        default=None,
        description="Error type classification"
    )


class ErrorResponse(BaseModel):
    """
    Standard error response schema.
    
    Used for all error responses across the API.
    """
    
    error: Dict[str, Any] = Field(
        ...,
        description="Error information"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": {
                    "code": 404,
                    "message": "User not found",
                    "details": [
                        {
                            "field": "user_id",
                            "message": "User with ID 999 does not exist",
                            "error_type": "not_found"
                        }
                    ],
                    "path": "/api/v1/ambient/recommend",
                    "timestamp": "2025-01-15T10:30:00Z"
                }
            }
        }
    )


# =========================================================
# Custom Exception Classes
# =========================================================

class APIError(HTTPException):
    """
    Base API exception class.
    
    All custom exceptions inherit from this.
    """
    
    def __init__(
        self,
        status_code: int,
        message: str,
        details: Optional[List[ErrorDetail]] = None,
        headers: Optional[Dict[str, str]] = None
    ):
        super().__init__(
            status_code=status_code,
            detail={
                "code": status_code,
                "message": message,
                "details": [d.model_dump() for d in (details or [])],
                "timestamp": datetime.now().isoformat()
            },
            headers=headers
        )


class ValidationError(APIError):
    """
    Validation error (422).
    
    Raised when request validation fails.
    """
    
    def __init__(
        self,
        message: str = "Request validation failed",
        details: Optional[List[ErrorDetail]] = None
    ):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            message=message,
            details=details
        )


class NotFoundError(APIError):
    """
    Not found error (404).
    
    Raised when requested resource doesn't exist.
    """
    
    def __init__(
        self,
        resource: str = "Resource",
        resource_id: Any = None,
        details: Optional[List[ErrorDetail]] = None
    ):
        message = f"{resource} not found"
        if resource_id is not None:
            message = f"{resource} with ID {resource_id} not found"
        
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            message=message,
            details=details
        )


class ServiceUnavailableError(APIError):
    """
    Service unavailable error (503).
    
    Raised when a service dependency is unavailable.
    """
    
    def __init__(
        self,
        service: str = "Service",
        message: Optional[str] = None,
        details: Optional[List[ErrorDetail]] = None
    ):
        if message is None:
            message = f"{service} is temporarily unavailable"
        
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            message=message,
            details=details,
            headers={"Retry-After": "60"}  # Suggest retry after 60 seconds
        )


class InternalServerError(APIError):
    """
    Internal server error (500).
    
    Raised for unexpected errors.
    """
    
    def __init__(
        self,
        message: str = "Internal server error",
        details: Optional[List[ErrorDetail]] = None
    ):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=message,
            details=details
        )


class RateLimitError(APIError):
    """
    Rate limit exceeded error (429).
    
    Raised when rate limit is exceeded.
    """
    
    def __init__(
        self,
        limit: int,
        window: str = "minute",
        retry_after: int = 60,
        details: Optional[List[ErrorDetail]] = None
    ):
        message = f"Rate limit exceeded: {limit} requests per {window}"
        
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            message=message,
            details=details,
            headers={"Retry-After": str(retry_after)}
        )


class AuthenticationError(APIError):
    """
    Authentication error (401).
    
    Raised when authentication fails.
    """
    
    def __init__(
        self,
        message: str = "Authentication required",
        details: Optional[List[ErrorDetail]] = None
    ):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            message=message,
            details=details,
            headers={"WWW-Authenticate": "Bearer"}
        )


class AuthorizationError(APIError):
    """
    Authorization error (403).
    
    Raised when user lacks permissions.
    """
    
    def __init__(
        self,
        message: str = "Insufficient permissions",
        required_role: Optional[str] = None,
        details: Optional[List[ErrorDetail]] = None
    ):
        if required_role:
            message = f"Requires role: {required_role}"
        
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            message=message,
            details=details
        )


# =========================================================
# Error Response Builder Utility
# =========================================================

def build_error_response(
    code: int,
    message: str,
    path: Optional[str] = None,
    details: Optional[List[ErrorDetail]] = None
) -> Dict[str, Any]:
    """
    Build standardized error response dictionary.
    
    Args:
        code: HTTP status code
        message: Error message
        path: Request path
        details: Error details list
    
    Returns:
        Error response dictionary
    """
    return {
        "error": {
            "code": code,
            "message": message,
            "details": [d.model_dump() for d in (details or [])],
            "path": path,
            "timestamp": datetime.now().isoformat()
        }
    }


# =========================================================
# Testing Utilities
# =========================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🔧 ERROR SCHEMAS TEST")
    print("="*70)
    
    # Test error detail
    print("\n📋 Error Detail:")
    detail = ErrorDetail(
        field="user_id",
        message="User ID must be positive",
        error_type="validation_error"
    )
    print(detail.model_dump_json(indent=2))
    
    # Test error response
    print("\n📋 Error Response:")
    error_resp = build_error_response(
        code=422,
        message="Validation failed",
        path="/api/v1/ambient/recommend",
        details=[detail]
    )
    import json
    print(json.dumps(error_resp, indent=2))
    
    print("\n✅ Error schemas test complete")