from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, JSON, ForeignKey,
    Index, CheckConstraint, UniqueConstraint, func
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB

from .base import Base

class User(Base):
    """User model storing profile and metadata."""
    
    distinct_id = Column(String(255), unique=True, nullable=False, index=True)
    user_id = Column(String(255), unique=True, nullable=True, index=True)
    email = Column(String(255), unique=True, nullable=True, index=True)
    name = Column(String(255), nullable=True)

    # Device & Location
    browser = Column(String(200), nullable=True)
    browser_version = Column(String(100), nullable=True)
    os = Column(String(200), nullable=True)
    device = Column(String(200), nullable=True)
    city = Column(String(200), nullable=True)
    region = Column(String(200), nullable=True)
    country_code = Column(String(10), nullable=True, index=True)
    timezone = Column(String(100), nullable=True)

    # Referrer
    initial_referrer = Column(Text, nullable=True)
    initial_referring_domain = Column(String(500), nullable=True)

    # UTM
    initial_utm_source = Column(String(200), nullable=True)
    initial_utm_medium = Column(String(200), nullable=True)
    initial_utm_campaign = Column(String(500), nullable=True)

    # Account
    is_logged_in = Column(Boolean, default=False)
    account_type = Column(String(50), nullable=True)
    role = Column(Integer, default=1)

    # Activity
    last_seen = Column(DateTime(timezone=True), nullable=True, index=True)
    total_sessions = Column(Integer, default=0)

    # Metadata
    user_data = Column(JSONB, nullable=True)

    # Relationships
    interactions = relationship("Interaction", back_populates="user", cascade="all, delete-orphan")
    features = relationship("UserFeature", back_populates="user", uselist=False, cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_user_location', 'country_code', 'city'),
        Index('idx_user_activity', 'last_seen', 'is_logged_in'),
    )


class Item(Base):
    """Item model representing features/tools."""

    item_code = Column(String(100), unique=True, nullable=False, index=True)
    item_name = Column(String(255), nullable=False)
    item_type = Column(String(50), nullable=False, index=True)

    category = Column(String(100), nullable=True, index=True)
    subcategory = Column(String(100), nullable=True)
    description = Column(Text, nullable=True)

    total_usage_count = Column(Integer, default=0, index=True)
    unique_user_count = Column(Integer, default=0)
    avg_rating = Column(Float, nullable=True)

    is_active = Column(Boolean, default=True, index=True)
    is_premium = Column(Boolean, default=False)

    tags = Column(JSONB, nullable=True)
    item_data = Column(JSONB, nullable=True)

    interactions = relationship("Interaction", back_populates="item", cascade="all, delete-orphan")
    features = relationship("ItemFeature", back_populates="item", uselist=False, cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_item_popularity', 'total_usage_count', 'unique_user_count'),
        Index('idx_item_category', 'category', 'item_type'),
        CheckConstraint('total_usage_count >= 0', name='check_usage_positive'),
    )


class Interaction(Base):
    """Interaction model storing user-item events."""

    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    item_id = Column(Integer, ForeignKey('items.id', ondelete='CASCADE'), nullable=True, index=True)

    event_name = Column(String(200), nullable=False, index=True)
    event_time = Column(DateTime(timezone=True), nullable=False, index=True)
    insert_id = Column(String(200), unique=True, nullable=False)

    device_id = Column(String(255), nullable=True, index=True)
    browser = Column(String(200), nullable=True)
    os = Column(String(200), nullable=True)
    current_url = Column(Text, nullable=True)

    field = Column(String(100), nullable=True)
    tone = Column(String(50), nullable=True)
    maintenance = Column(String(50), nullable=True)
    target_language = Column(String(50), nullable=True)

    response_time_ms = Column(Integer, nullable=True)
    input_sentence_length = Column(Integer, nullable=True)

    llm_provider = Column(String(50), nullable=True)
    llm_name = Column(String(200), nullable=True)
    llm_version = Column(String(100), nullable=True)

    position = Column(String(50), nullable=True)
    trigger = Column(String(100), nullable=True)
    properties = Column(JSONB, nullable=True)

    user = relationship("User", back_populates="interactions")
    item = relationship("Item", back_populates="interactions")

    __table_args__ = (
        Index('idx_interaction_user_time', 'user_id', 'event_time'),
        Index('idx_interaction_event', 'event_name', 'event_time'),
        Index('idx_interaction_device', 'user_id', 'device_id'),
    )


class UserFeature(Base):
    """Computed user features for recommendation."""

    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), unique=True, nullable=False, index=True)

    total_paraphrases = Column(Integer, default=0)
    last_7d_count = Column(Integer, default=0)
    last_30d_count = Column(Integer, default=0)
    avg_session_length_minutes = Column(Float, nullable=True)

    preferred_tone = Column(String(50), nullable=True)
    preferred_maintenance = Column(String(50), nullable=True)
    preferred_language = Column(String(50), nullable=True)

    most_active_hour = Column(Integer, nullable=True)
    most_active_day_of_week = Column(Integer, nullable=True)
    avg_response_time_ms = Column(Float, nullable=True)

    primary_browser = Column(String(200), nullable=True)
    primary_os = Column(String(200), nullable=True)
    primary_device = Column(String(200), nullable=True)

    avg_input_length = Column(Float, nullable=True)
    vocabulary_diversity = Column(Float, nullable=True)
    avg_selections_per_session = Column(Float, nullable=True)
    copy_rate = Column(Float, nullable=True)

    days_since_first_interaction = Column(Integer, nullable=True)
    days_since_last_interaction = Column(Integer, nullable=True)

    engagement_score = Column(Float, nullable=True)
    exploration_score = Column(Float, nullable=True)

    feature_vector = Column(JSONB, nullable=True)
    computed_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)

    user = relationship("User", back_populates="features")

    __table_args__ = (
        Index('idx_userfeature_engagement', 'engagement_score', 'last_7d_count'),
        Index('idx_userfeature_recency', 'days_since_last_interaction'),
    )


class ItemFeature(Base):
    """Computed item features for recommendation."""

    item_id = Column(Integer, ForeignKey('items.id', ondelete='CASCADE'), unique=True, nullable=False, index=True)

    popularity_score = Column(Float, nullable=True)
    trending_score = Column(Float, nullable=True)
    total_runs = Column(Integer, default=0)
    total_selections = Column(Integer, default=0)
    total_copies = Column(Integer, default=0)

    avg_response_time_ms = Column(Float, nullable=True)
    success_rate = Column(Float, nullable=True)

    unique_users_all_time = Column(Integer, default=0)
    unique_users_7d = Column(Integer, default=0)
    unique_users_30d = Column(Integer, default=0)
    user_retention_rate = Column(Float, nullable=True)

    peak_usage_hour = Column(Integer, nullable=True)
    peak_usage_day = Column(Integer, nullable=True)

    category_rank = Column(Integer, nullable=True)
    category_percentile = Column(Float, nullable=True)

    quality_score = Column(Float, nullable=True)
    freshness_score = Column(Float, nullable=True)
    frequently_used_with = Column(JSONB, nullable=True)

    feature_vector = Column(JSONB, nullable=True)
    computed_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)

    item = relationship("Item", back_populates="features")

    __table_args__ = (
        Index('idx_itemfeature_popularity', 'popularity_score', 'trending_score'),
        Index('idx_itemfeature_quality', 'quality_score', 'success_rate'),
    )


class InteractionFeature(Base):
    """
    Interaction sequence features for advanced recommendation.
    
    세션별 행동 패턴을 저장하여:
    - Soft Loop: 반복 패턴 감지
    - Ghost Preview: 탐색 패턴 분석
    - Temporal Flow: 시간대별 사용 패턴
    """
    
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    session_id = Column(String(100), unique=True, nullable=False, index=True)
    
    # Session Basic Info
    session_start = Column(DateTime(timezone=True), nullable=False, index=True)
    session_end = Column(DateTime(timezone=True), nullable=False)
    session_duration_minutes = Column(Float, nullable=False)
    
    # Event Counts
    total_events = Column(Integer, default=0)
    unique_items = Column(Integer, default=0)
    run_count = Column(Integer, default=0)
    select_count = Column(Integer, default=0)
    copy_count = Column(Integer, default=0)
    
    # Temporal Patterns (Temporal Flow 지원)
    hour_of_day = Column(Integer, nullable=True)
    time_of_day = Column(String(20), nullable=True)
    is_peak_hours = Column(Integer, default=0)
    
    # Context Switches (Soft Loop 지원)
    device_changes = Column(Integer, default=0)
    tone_changes = Column(Integer, default=0)
    maintenance_changes = Column(Integer, default=0)
    
    # Quality Metrics
    avg_response_time_ms = Column(Float, nullable=True)
    events_per_minute = Column(Float, nullable=True)
    
    # Ghost Preview Metrics
    preview_to_select_ratio = Column(Float, nullable=True)
    exploration_score = Column(Float, nullable=True)
    
    # Soft Loop Metrics
    repeat_pattern_score = Column(Float, nullable=True)
    sequence_diversity = Column(Float, nullable=True)
    
    # Temporal Gap Statistics
    avg_gap_minutes = Column(Float, nullable=True)
    median_gap_minutes = Column(Float, nullable=True)
    max_gap_minutes = Column(Float, nullable=True)
    
    # Advanced Features
    event_type_distribution = Column(JSONB, nullable=True)
    common_sequences = Column(JSONB, nullable=True)
    tone_diversity = Column(Integer, default=0)
    
    computed_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    
    __table_args__ = (
        Index('idx_interaction_feature_user_time', 'user_id', 'session_start'),
        Index('idx_interaction_feature_session', 'session_id'),
    )