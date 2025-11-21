from datetime import datetime
from typing import Any, Dict

from sqlalchemy import Column, Integer, DateTime, func
from sqlalchemy.orm import declarative_base, declared_attr


class TimestampMixin:
    """Mixin providing automatic timestamp columns."""

    @declared_attr
    def created_at(cls):
        return Column(DateTime(timezone=True), default=func.now(), nullable=False, index=True)

    @declared_attr
    def updated_at(cls):
        return Column(DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False)


class BaseModel:
    """Base model class with utility helpers."""

    @declared_attr
    def __tablename__(cls) -> str:
        """Auto-generate table name in snake_case plural form."""
        name = cls.__name__
        return ''.join(['_' + c.lower() if c.isupper() else c for c in name]).lstrip('_') + 's'

    id = Column(Integer, primary_key=True, autoincrement=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert SQLAlchemy model instance to dictionary."""
        return {column.name: getattr(self, column.name) for column in self.__table__.columns}

    def update_from_dict(self, data: Dict[str, Any]) -> None:
        """Update model attributes from a dictionary."""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def __repr__(self) -> str:
        """Readable string representation for debugging."""
        attrs = ', '.join(f"{k}={v!r}" for k, v in self.to_dict().items() if k != 'id')
        return f"<{self.__class__.__name__}(id={self.id}, {attrs})>"


# Declarative Base combining BaseModel and TimestampMixin
Base = declarative_base(cls=(BaseModel, TimestampMixin))