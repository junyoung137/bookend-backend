# package routers
# src/api/routers/__init__.py
"""
API Routers package for Bookend Recommendation System.

Provides modular routers for different recommendation strategies:
- ambient: Layout-aware recommendations
- temporal: Time-pattern based recommendations
- hybrid: Multi-strategy blended recommendations
- features: Feature extraction and analysis
- health: Health checks and monitoring
- admin: Administrative operations

Usage:
    from src.api.routers import ambient, temporal, hybrid
    
    app.include_router(ambient.router, prefix="/api/v1/ambient")
    app.include_router(temporal.router, prefix="/api/v1/temporal")
    app.include_router(hybrid.router, prefix="/api/v1/hybrid")
"""

from . import (
    ambient,
    temporal,
    hybrid,
    features,
    health,
    admin
)

__all__ = [
    "ambient",
    "temporal",
    "hybrid",
    "features",
    "health",
    "admin"
]