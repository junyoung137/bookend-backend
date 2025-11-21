# package config
"""
Configuration package for Bookend Recommendation System.

This package manages:
- Environment-specific settings (DEV/PROD)
- Database connections (PostgreSQL)
- Storage configurations (MinIO, Redis)
- Logging setup
- Model configurations

Usage:
    >>> from config import get_settings, get_storage, setup_logging
    >>> 
    >>> # Setup logging
    >>> setup_logging()
    >>> 
    >>> # Get settings
    >>> settings = get_settings()
    >>> print(settings.environment)
    >>> 
    >>> # Get storage manager
    >>> storage = get_storage()
    >>> storage.save_model(model, "ambient", "v1.0.0")
"""

# Core settings
from .settings import (
    Settings,
    get_settings,
    validate_settings,
    Environment
)

# Logging
from .logging_config import (
    setup_logging,
    get_logger
)

# Database
from .database import (
    get_db,
    get_session,
    DatabaseConnection
)

# Storage backends
from .minio_config import (
    get_minio,
    minio_client,
    MinIOConnection
)

from .redis_config import (
    get_redis,
    redis_pipeline,
    RedisConnection
)

# Unified storage interface
from .storage import (
    get_storage,
    StorageManager
)

__version__ = "0.1.0"

__all__ = [
    # Settings
    "Settings",
    "get_settings",
    "validate_settings",
    "Environment",
    
    # Logging
    "setup_logging",
    "get_logger",
    
    # Database
    "get_db",
    "get_session",
    "DatabaseConnection",
    
    # Storage
    "get_minio",
    "minio_client",
    "MinIOConnection",
    "get_redis",
    "redis_pipeline",
    "RedisConnection",
    "get_storage",
    "StorageManager",
]