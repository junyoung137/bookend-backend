# package config
"""
Configuration package for Bookend Recommendation System.
...
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
    DatabaseConnection,
    session_scope,   
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
    "session_scope",   
    
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
