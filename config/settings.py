import os
from enum import Enum
from typing import Optional
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# .env 파일 명시적으로 로드
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded .env file from: {env_path}")
else:
    print(f"⚠️  .env file not found at: {env_path}")


class Environment(str, Enum):
    """Valid environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class DatabaseSettings(BaseSettings):
    """PostgreSQL database configuration."""
    
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, ge=1, le=65535, description="Database port")
    name: str = Field(default="bookend", description="Database name")
    user: str = Field(default="postgres", description="Database user")
    password: str = Field(default="", description="Database password")
    pool_size: int = Field(default=5, ge=1, le=50, description="Connection pool size")
    max_overflow: int = Field(default=10, ge=0, le=50, description="Max overflow connections")
    pool_timeout: int = Field(default=30, ge=1, description="Pool timeout in seconds")
    echo: bool = Field(default=False, description="Echo SQL queries")
    
    @property
    def url(self) -> str:
        """Generate PostgreSQL connection URL."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
    
    @property
    def async_url(self) -> str:
        """Generate async PostgreSQL connection URL."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
    
    model_config = SettingsConfigDict(
        env_prefix="DB_",
        case_sensitive=False,
        extra="ignore"
    )


class RedisSettings(BaseSettings):
    """Redis cache configuration."""
    
    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, ge=1, le=65535, description="Redis port")
    db: int = Field(default=0, ge=0, le=15, description="Redis database number")
    password: Optional[str] = Field(default=None, description="Redis password")
    ttl: int = Field(default=3600, ge=60, description="Default TTL in seconds")
    max_connections: int = Field(default=10, ge=1, le=100, description="Max connections")
    
    @property
    def url(self) -> str:
        """Generate Redis connection URL."""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"
    
    model_config = SettingsConfigDict(
        env_prefix="REDIS_",
        case_sensitive=False,
        extra="ignore"
    )


class MinIOSettings(BaseSettings):
    """MinIO object storage configuration."""
    
    endpoint: str = Field(default="localhost:9000", description="MinIO endpoint")
    access_key: str = Field(default="minioadmin", description="Access key")
    secret_key: str = Field(default="minioadmin", description="Secret key")
    secure: bool = Field(default=False, description="Use HTTPS")
    region: str = Field(default="us-east-1", description="Region")
    
    # Bucket names
    features_bucket: str = Field(default="bookend-features", description="Features bucket")
    models_bucket: str = Field(default="bookend-models", description="Models bucket")
    interim_bucket: str = Field(default="bookend-interim", description="Interim data bucket")
    
    model_config = SettingsConfigDict(
        env_prefix="MINIO_",
        case_sensitive=False,
        extra="ignore"
    )


class APISettings(BaseSettings):
    """FastAPI application configuration."""
    
    title: str = Field(default="Bookend Recommendation API", description="API title")
    version: str = Field(default="0.1.0", description="API version")
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, ge=1, le=65535, description="API port")
    workers: int = Field(default=1, ge=1, le=16, description="Number of workers")
    reload: bool = Field(default=True, description="Auto-reload on code changes")
    
    # Rate limiting
    rate_limit_per_minute: int = Field(default=100, ge=1, description="Requests per minute per user")
    rate_limit_per_ip: int = Field(default=1000, ge=1, description="Requests per minute per IP")
    
    # CORS (환경변수에서는 쉼표로 구분된 문자열로 받음)
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:8501",
        description="Allowed CORS origins (comma-separated)"
    )
    
    def get_cors_origins_list(self) -> list[str]:
        """CORS origins를 리스트로 반환"""
        if isinstance(self.cors_origins, str):
            return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]
        return self.cors_origins
    
    model_config = SettingsConfigDict(
        env_prefix="API_",
        case_sensitive=False,
        extra="ignore"
    )


class Settings(BaseSettings):
    """
    Main settings class combining all configuration groups.
    
    Usage:
        settings = get_settings()
        db_url = settings.database.url
    """
    
    # Environment
    environment: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Current environment"
    )
    debug: bool = Field(default=True, description="Debug mode")
    
    # Sub-configurations (수정: 환경변수에서 직접 로드)
    database: DatabaseSettings = Field(default_factory=lambda: DatabaseSettings())
    redis: RedisSettings = Field(default_factory=lambda: RedisSettings())
    minio: MinIOSettings = Field(default_factory=lambda: MinIOSettings())
    api: APISettings = Field(default_factory=lambda: APISettings())
    
    # Project paths
    project_root: str = Field(default=".", description="Project root directory")
    data_dir: str = Field(default="data", description="Data directory")
    models_dir: str = Field(default="data/models", description="Local models directory")
    logs_dir: str = Field(default="logs", description="Logs directory")
    
    # Model configuration
    model_config_path: str = Field(
        default="config/model_config.yaml",
        description="Model configuration file path"
    )
    
    @field_validator("environment", mode="before")
    @classmethod
    def parse_environment(cls, v: str | Environment) -> Environment:
        """Parse environment string to enum."""
        if isinstance(v, Environment):
            return v
        try:
            return Environment(v.lower())
        except ValueError:
            raise ValueError(
                f"Invalid environment: {v}. Must be one of: "
                f"{', '.join(e.value for e in Environment)}"
            )
    
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == Environment.DEVELOPMENT
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    This ensures only one Settings object is created per application lifecycle.
    Uses lru_cache to implement the Singleton pattern.
    
    Returns:
        Settings: The application settings
    
    Example:
        >>> settings = get_settings()
        >>> db_url = settings.database.url
    """
    settings = Settings()
    
    # 디버그 정보 출력 (개발 환경에서만)
    if settings.debug:
        print(f"\n📋 Settings Loaded:")
        print(f"  Environment: {settings.environment.value}")
        print(f"  Database: {settings.database.user}@{settings.database.host}:{settings.database.port}/{settings.database.name}")
        print(f"  Debug Mode: {settings.debug}\n")
    
    return settings


def validate_settings() -> dict[str, bool]:
    """
    Validate all settings and return status.
    
    Returns:
        dict: Validation results for each setting group
    
    Example:
        >>> results = validate_settings()
        >>> if not all(results.values()):
        ...     print("Configuration issues detected!")
    """
    settings = get_settings()
    results = {
        "database": True,
        "redis": True,
        "minio": True,
        "api": True,
        "paths": True,
    }
    
    # Validate database
    try:
        assert settings.database.host, "Database host is required"
        assert settings.database.port > 0, "Database port must be positive"
        assert settings.database.name, "Database name is required"
        assert settings.database.user, "Database user is required"
        assert settings.database.password, "Database password is required"
        print("✅ Database settings validated")
    except AssertionError as e:
        results["database"] = False
        print(f"❌ Database validation failed: {e}")
    
    # Validate Redis
    try:
        assert settings.redis.host, "Redis host is required"
        assert settings.redis.port > 0, "Redis port must be positive"
        print("✅ Redis settings validated")
    except AssertionError as e:
        results["redis"] = False
        print(f"⚠️  Redis validation failed: {e}")
    
    # Validate MinIO
    try:
        assert settings.minio.endpoint, "MinIO endpoint is required"
        assert settings.minio.access_key, "MinIO access key is required"
        assert settings.minio.secret_key, "MinIO secret key is required"
        print("✅ MinIO settings validated")
    except AssertionError as e:
        results["minio"] = False
        print(f"⚠️  MinIO validation failed: {e}")
    
    # Validate paths
    try:
        import pathlib
        for path_attr in ["data_dir", "models_dir", "logs_dir"]:
            path = pathlib.Path(getattr(settings, path_attr))
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
        print("✅ All paths validated/created")
    except Exception as e:
        results["paths"] = False
        print(f"❌ Path validation failed: {e}")
    
    return results


def print_current_config():
    """현재 설정을 출력합니다 (디버깅용)."""
    settings = get_settings()
    
    print("\n" + "="*60)
    print("🔧 Current Configuration")
    print("="*60)
    
    print(f"\n📌 Environment:")
    print(f"  Mode: {settings.environment.value}")
    print(f"  Debug: {settings.debug}")
    
    print(f"\n🗄️  Database:")
    print(f"  Host: {settings.database.host}")
    print(f"  Port: {settings.database.port}")
    print(f"  Name: {settings.database.name}")
    print(f"  User: {settings.database.user}")
    print(f"  Password: {'*' * len(settings.database.password) if settings.database.password else '(not set)'}")
    print(f"  URL: {settings.database.url.replace(settings.database.password, '***')}")
    
    print(f"\n🔴 Redis:")
    print(f"  Host: {settings.redis.host}")
    print(f"  Port: {settings.redis.port}")
    print(f"  DB: {settings.redis.db}")
    
    print(f"\n📦 MinIO:")
    print(f"  Endpoint: {settings.minio.endpoint}")
    print(f"  Access Key: {settings.minio.access_key}")
    
    print(f"\n🌐 API:")
    print(f"  Host: {settings.api.host}")
    print(f"  Port: {settings.api.port}")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    # Test settings loading
    print_current_config()
    
    # Validate all settings
    print("🔍 Validating settings...\n")
    validation_results = validate_settings()
    
    print(f"\n📊 Validation Results:")
    for component, is_valid in validation_results.items():
        status = "✅" if is_valid else "❌"
        print(f"  {status} {component}: {'Valid' if is_valid else 'Invalid'}")
    
    print(f"\n{'✅ All Valid!' if all(validation_results.values()) else '❌ Some validations failed'}\n")