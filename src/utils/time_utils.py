"""
Time utilities for Bookend Recommendation System.

Provides time-related helper functions:
1. Duration formatting (seconds to human-readable)
2. Datetime parsing (flexible input formats)
3. Timestamp generation (UTC standardized)
4. Time difference calculation

Principles:
- Single Responsibility: Each function handles one time operation
- UTC Standard: All timestamps in UTC
- Error Handling: Graceful fallback for invalid inputs
- Type Safety: Clear input/output types
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Union

logger = logging.getLogger(__name__)


# =========================================================
# Duration Formatting
# =========================================================

def format_duration(seconds: float, precision: int = 2) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        precision: Decimal precision for sub-second durations
    
    Returns:
        Formatted duration string
    
    Example:
        >>> format_duration(0.123)
        '123.00ms'
        >>> format_duration(65.5)
        '1m 5.50s'
        >>> format_duration(3661)
        '1h 1m 1s'
    """
    try:
        if seconds < 0:
            return "0s"
        
        # Milliseconds (< 1s)
        if seconds < 1:
            ms = seconds * 1000
            return f"{ms:.{precision}f}ms"
        
        # Seconds only (< 1m)
        if seconds < 60:
            return f"{seconds:.{precision}f}s"
        
        # Minutes and seconds (< 1h)
        if seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m {secs:.{precision}f}s"
        
        # Hours, minutes, and seconds (< 1d)
        if seconds < 86400:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours}h {minutes}m {secs:.0f}s"
        
        # Days and hours (>= 1d)
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        return f"{days}d {hours}h"
    
    except Exception as e:
        logger.warning(f"Failed to format duration: {e}")
        return f"{seconds:.2f}s"


# =========================================================
# Datetime Parsing
# =========================================================

def parse_datetime(
    value: Union[str, int, float, datetime],
    default_timezone: timezone = timezone.utc
) -> Optional[datetime]:
    """
    Parse various datetime formats to datetime object.
    
    Supports:
    - ISO 8601 strings: "2025-01-01T12:00:00Z"
    - Unix timestamps: 1704110400
    - Datetime objects: datetime(2025, 1, 1)
    
    Args:
        value: Datetime value in various formats
        default_timezone: Timezone to use if none specified
    
    Returns:
        Timezone-aware datetime object, or None if parsing fails
    
    Example:
        >>> parse_datetime("2025-01-01T12:00:00Z")
        datetime.datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
        >>> parse_datetime(1704110400)
        datetime.datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    """
    try:
        # Already a datetime
        if isinstance(value, datetime):
            # Ensure timezone-aware
            if value.tzinfo is None:
                return value.replace(tzinfo=default_timezone)
            return value
        
        # Unix timestamp (int or float)
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=default_timezone)
        
        # String (ISO 8601)
        if isinstance(value, str):
            # Try ISO format with timezone
            try:
                return datetime.fromisoformat(value.replace('Z', '+00:00'))
            except ValueError:
                pass
            
            # Try ISO format without timezone
            try:
                dt = datetime.fromisoformat(value)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=default_timezone)
                return dt
            except ValueError:
                pass
            
            # Try common formats
            formats = [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
                "%Y/%m/%d %H:%M:%S",
                "%Y/%m/%d",
            ]
            
            for fmt in formats:
                try:
                    dt = datetime.strptime(value, fmt)
                    return dt.replace(tzinfo=default_timezone)
                except ValueError:
                    continue
        
        logger.warning(f"Unable to parse datetime: {value}")
        return None
    
    except Exception as e:
        logger.error(f"Error parsing datetime: {e}")
        return None


# =========================================================
# Timestamp Generation
# =========================================================

def get_current_timestamp(as_string: bool = False) -> Union[datetime, str]:
    """
    Get current UTC timestamp.
    
    Args:
        as_string: Whether to return as ISO 8601 string
    
    Returns:
        Current UTC datetime or ISO string
    
    Example:
        >>> now = get_current_timestamp()
        >>> print(now.isoformat())
        '2025-01-01T12:00:00+00:00'
        >>> now_str = get_current_timestamp(as_string=True)
        >>> print(now_str)
        '2025-01-01T12:00:00+00:00'
    """
    try:
        now = datetime.now(tz=timezone.utc)
        
        if as_string:
            return now.isoformat()
        
        return now
    
    except Exception as e:
        logger.error(f"Error getting current timestamp: {e}")
        # Fallback
        now = datetime.now(tz=timezone.utc)
        return now.isoformat() if as_string else now


# =========================================================
# Time Difference Calculation
# =========================================================

def calculate_time_diff(
    start: Union[datetime, str],
    end: Optional[Union[datetime, str]] = None,
    unit: str = "seconds"
) -> Optional[float]:
    """
    Calculate time difference between two datetimes.
    
    Args:
        start: Start datetime
        end: End datetime (defaults to now)
        unit: Output unit ('seconds', 'minutes', 'hours', 'days')
    
    Returns:
        Time difference in specified unit, or None if calculation fails
    
    Example:
        >>> start = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        >>> end = datetime(2025, 1, 1, 13, 30, 0, tzinfo=timezone.utc)
        >>> calculate_time_diff(start, end, unit='minutes')
        90.0
    """
    try:
        # Parse start
        start_dt = parse_datetime(start)
        if not start_dt:
            logger.warning("Invalid start datetime")
            return None
        
        # Parse end (or use now)
        if end is None:
            end_dt = get_current_timestamp()
        else:
            end_dt = parse_datetime(end)
            if not end_dt:
                logger.warning("Invalid end datetime")
                return None
        
        # Calculate difference
        diff = end_dt - start_dt
        total_seconds = diff.total_seconds()
        
        # Convert to requested unit
        conversions = {
            "seconds": 1,
            "minutes": 60,
            "hours": 3600,
            "days": 86400
        }
        
        if unit not in conversions:
            logger.warning(f"Unknown unit '{unit}', using seconds")
            unit = "seconds"
        
        return total_seconds / conversions[unit]
    
    except Exception as e:
        logger.error(f"Error calculating time difference: {e}")
        return None


# =========================================================
# Relative Time Description
# =========================================================

def get_relative_time_description(dt: Union[datetime, str]) -> str:
    """
    Get human-readable relative time description.
    
    Args:
        dt: Datetime to describe
    
    Returns:
        Relative time description (e.g., "2 hours ago", "in 3 days")
    
    Example:
        >>> from datetime import timedelta
        >>> past = datetime.now(timezone.utc) - timedelta(hours=2)
        >>> get_relative_time_description(past)
        '2 hours ago'
    """
    try:
        target_dt = parse_datetime(dt)
        if not target_dt:
            return "unknown time"
        
        now = get_current_timestamp()
        diff = now - target_dt
        seconds = diff.total_seconds()
        
        # Future time
        if seconds < 0:
            seconds = abs(seconds)
            suffix = "from now"
        else:
            suffix = "ago"
        
        # Choose appropriate unit
        if seconds < 60:
            return f"{int(seconds)} seconds {suffix}"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return f"{minutes} minute{'s' if minutes != 1 else ''} {suffix}"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return f"{hours} hour{'s' if hours != 1 else ''} {suffix}"
        elif seconds < 2592000:  # 30 days
            days = int(seconds / 86400)
            return f"{days} day{'s' if days != 1 else ''} {suffix}"
        elif seconds < 31536000:  # 365 days
            months = int(seconds / 2592000)
            return f"{months} month{'s' if months != 1 else ''} {suffix}"
        else:
            years = int(seconds / 31536000)
            return f"{years} year{'s' if years != 1 else ''} {suffix}"
    
    except Exception as e:
        logger.error(f"Error getting relative time: {e}")
        return "unknown time"


# =========================================================
# Time Range Generator
# =========================================================

def generate_time_range(
    start: Union[datetime, str],
    end: Union[datetime, str],
    step: timedelta
) -> list[datetime]:
    """
    Generate list of datetimes in range.
    
    Args:
        start: Start datetime
        end: End datetime
        step: Time step between datetimes
    
    Returns:
        List of datetimes
    
    Example:
        >>> start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        >>> end = datetime(2025, 1, 2, tzinfo=timezone.utc)
        >>> step = timedelta(hours=6)
        >>> times = generate_time_range(start, end, step)
        >>> len(times)
        5  # 00:00, 06:00, 12:00, 18:00, 24:00
    """
    try:
        start_dt = parse_datetime(start)
        end_dt = parse_datetime(end)
        
        if not start_dt or not end_dt:
            logger.warning("Invalid datetime range")
            return []
        
        if start_dt >= end_dt:
            logger.warning("Start time must be before end time")
            return []
        
        times = []
        current = start_dt
        
        while current <= end_dt:
            times.append(current)
            current += step
        
        return times
    
    except Exception as e:
        logger.error(f"Error generating time range: {e}")
        return []


if __name__ == "__main__":
    from config.logging_config import setup_logging
    
    setup_logging(environment="development", debug=True)
    
    print("=" * 70)
    print("TIME UTILITIES DEMO")
    print("=" * 70)
    
    # Duration formatting
    print("\n1️⃣ Duration Formatting:")
    durations = [0.123, 5.5, 65.5, 3661, 90000]
    for d in durations:
        print(f"   {d}s → {format_duration(d)}")
    
    # Datetime parsing
    print("\n2️⃣ Datetime Parsing:")
    test_values = [
        "2025-01-01T12:00:00Z",
        1704110400,
        "2025-01-01",
        datetime.now()
    ]
    for val in test_values:
        parsed = parse_datetime(val)
        print(f"   {val} → {parsed}")
    
    # Current timestamp
    print("\n3️⃣ Current Timestamp:")
    now = get_current_timestamp()
    now_str = get_current_timestamp(as_string=True)
    print(f"   Datetime: {now}")
    print(f"   String: {now_str}")
    
    # Time difference
    print("\n4️⃣ Time Difference:")
    start = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    end = datetime(2025, 1, 1, 13, 30, 0, tzinfo=timezone.utc)
    diff_min = calculate_time_diff(start, end, unit='minutes')
    print(f"   Difference: {diff_min} minutes")
    
    # Relative time
    print("\n5️⃣ Relative Time:")
    past = datetime.now(timezone.utc) - timedelta(hours=2, minutes=30)
    rel_time = get_relative_time_description(past)
    print(f"   {past} → {rel_time}")
    
    # Time range
    print("\n6️⃣ Time Range Generation:")
    range_start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    range_end = datetime(2025, 1, 2, tzinfo=timezone.utc)
    time_range = generate_time_range(range_start, range_end, timedelta(hours=6))
    print(f"   Generated {len(time_range)} timestamps:")
    for t in time_range:
        print(f"      {t.strftime('%Y-%m-%d %H:%M')}")
    
    print("\n✅ Time utilities demo completed!")