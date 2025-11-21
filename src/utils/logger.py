"""
Logger utilities for Bookend Recommendation System.

Provides enhanced logging capabilities:
1. Structured logger retrieval (wraps config.logging_config)
2. Context-aware logging (user_id, request_id)
3. Execution time tracking decorator
4. Error logging with stack traces

Principles:
- Single Source of Truth: Uses config.logging_config
- Context Injection: Automatic metadata addition
- Performance Tracking: Built-in timing decorator
- Error Handling: Graceful degradation
"""

import logging
import functools
import time
from typing import Any, Callable, Dict, Optional
from contextlib import contextmanager

# Import from existing logging_config (Single Source of Truth)
from config.logging_config import get_logger as _get_base_logger, LoggerAdapter

# Module logger
logger = logging.getLogger(__name__)


# =========================================================
# Enhanced Logger Retrieval
# =========================================================

def get_logger(name: str, **default_context) -> logging.Logger:
    """
    Get a logger instance with optional default context.
    
    This is a convenience wrapper around config.logging_config.get_logger
    that allows adding default context fields.
    
    Args:
        name: Logger name (typically __name__)
        **default_context: Default context fields to include in all logs
    
    Returns:
        logging.Logger or LoggerAdapter: Logger instance
    
    Example:
        >>> logger = get_logger(__name__, service="api")
        >>> logger.info("Request processed")
        # Logs with service="api" automatically included
    """
    try:
        base_logger = _get_base_logger(name)
        
        # If context provided, wrap in adapter
        if default_context:
            return LoggerAdapter(base_logger, **default_context)
        
        return base_logger
    
    except Exception as e:
        # Fallback to basic logger if config fails
        fallback_logger = logging.getLogger(name)
        fallback_logger.warning(f"Failed to get configured logger, using fallback: {e}")
        return fallback_logger


# =========================================================
# Context Manager for Temporary Context
# =========================================================

@contextmanager
def LogContext(logger: logging.Logger, **context):
    """
    Context manager for temporary logging context.
    
    Wraps logger in LoggerAdapter for the duration of the context,
    then returns to original logger.
    
    Args:
        logger: Base logger instance
        **context: Context fields to add temporarily
    
    Yields:
        LoggerAdapter: Logger with added context
    
    Example:
        >>> logger = get_logger(__name__)
        >>> with LogContext(logger, request_id="req_123", user_id="user_456"):
        ...     logger.info("Processing request")
        ...     # Logs include request_id and user_id
        >>> logger.info("Outside context")
        # Logs without request_id and user_id
    """
    try:
        # Create temporary adapter with context
        context_logger = LoggerAdapter(logger, **context)
        yield context_logger
    
    except Exception as e:
        logger.error(f"Error in logging context: {e}", exc_info=True)
        # Yield original logger on error
        yield logger


# =========================================================
# Execution Time Tracking Decorator
# =========================================================

def log_execution_time(
    logger: Optional[logging.Logger] = None,
    level: int = logging.INFO,
    message: Optional[str] = None
):
    """
    Decorator to log function execution time.
    
    Args:
        logger: Logger instance (uses function's module logger if None)
        level: Log level to use
        message: Custom message template (default: "Function {name} executed in {duration:.2f}s")
    
    Returns:
        Decorator function
    
    Example:
        >>> @log_execution_time()
        ... def process_data(data):
        ...     time.sleep(1)
        ...     return data
        >>> 
        >>> result = process_data([1, 2, 3])
        # Logs: "Function process_data executed in 1.00s"
        
        >>> @log_execution_time(level=logging.DEBUG, message="{name} took {duration:.4f}s")
        ... def fast_function():
        ...     pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Determine logger
            func_logger = logger or logging.getLogger(func.__module__)
            
            # Start timing
            start_time = time.time()
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Calculate duration
                duration = time.time() - start_time
                
                # Format message
                msg = message or "Function {name} executed in {duration:.2f}s"
                log_msg = msg.format(
                    name=func.__name__,
                    duration=duration,
                    args=args,
                    kwargs=kwargs
                )
                
                # Log execution time
                func_logger.log(level, log_msg)
                
                return result
            
            except Exception as e:
                # Log error with execution time
                duration = time.time() - start_time
                func_logger.error(
                    f"Function {func.__name__} failed after {duration:.2f}s: {e}",
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator


# =========================================================
# Structured Error Logging
# =========================================================

def log_error(
    logger: logging.Logger,
    message: str,
    error: Optional[Exception] = None,
    **context
) -> None:
    """
    Log error with structured context and optional exception info.
    
    Args:
        logger: Logger instance
        message: Error message
        error: Exception object (includes stack trace if provided)
        **context: Additional context fields
    
    Example:
        >>> try:
        ...     risky_operation()
        ... except ValueError as e:
        ...     log_error(
        ...         logger,
        ...         "Failed to process data",
        ...         error=e,
        ...         user_id="user_123",
        ...         data_size=1000
        ...     )
    """
    try:
        # Build error context
        error_context = {
            "error_type": type(error).__name__ if error else "Unknown",
            "error_message": str(error) if error else message,
            **context
        }
        
        # Log with context
        if error:
            logger.error(
                message,
                exc_info=True,
                extra=error_context
            )
        else:
            logger.error(
                message,
                extra=error_context
            )
    
    except Exception as log_err:
        # Fallback to basic logging if structured logging fails
        logger.error(f"{message} (logging error: {log_err})")


# =========================================================
# Performance Tracking Context Manager
# =========================================================

@contextmanager
def track_performance(
    logger: logging.Logger,
    operation: str,
    level: int = logging.INFO,
    **context
):
    """
    Context manager for tracking operation performance.
    
    Args:
        logger: Logger instance
        operation: Operation name/description
        level: Log level to use
        **context: Additional context fields
    
    Yields:
        Dict: Context dictionary that can be updated during operation
    
    Example:
        >>> with track_performance(logger, "data_processing", user_id="user_123") as ctx:
        ...     process_data()
        ...     ctx['records_processed'] = 1000
        # Logs: "Operation data_processing completed in X.XXs (records_processed=1000)"
    """
    start_time = time.time()
    operation_context = dict(context)
    
    try:
        logger.log(level, f"Starting operation: {operation}", extra=operation_context)
        
        # Yield mutable context
        yield operation_context
        
        # Calculate duration
        duration = time.time() - start_time
        operation_context['duration_seconds'] = round(duration, 3)
        
        # Log completion
        logger.log(
            level,
            f"Operation {operation} completed in {duration:.2f}s",
            extra=operation_context
        )
    
    except Exception as e:
        # Log failure
        duration = time.time() - start_time
        operation_context['duration_seconds'] = round(duration, 3)
        operation_context['error'] = str(e)
        
        logger.error(
            f"Operation {operation} failed after {duration:.2f}s: {e}",
            exc_info=True,
            extra=operation_context
        )
        raise


# =========================================================
# Conditional Logging Helper
# =========================================================

def log_if(
    condition: bool,
    logger: logging.Logger,
    level: int,
    message: str,
    **kwargs
) -> None:
    """
    Log message only if condition is True.
    
    Args:
        condition: Whether to log
        logger: Logger instance
        level: Log level
        message: Log message
        **kwargs: Additional logging arguments
    
    Example:
        >>> log_if(
        ...     user.is_premium,
        ...     logger,
        ...     logging.INFO,
        ...     "Premium user action",
        ...     user_id=user.id
        ... )
    """
    if condition:
        logger.log(level, message, **kwargs)


# =========================================================
# Batch Logging for Collections
# =========================================================

def log_collection(
    logger: logging.Logger,
    level: int,
    items: list,
    message_template: str,
    max_items: int = 10
) -> None:
    """
    Log information about a collection of items.
    
    Args:
        logger: Logger instance
        level: Log level
        items: Collection to log
        message_template: Message template with {count} and {items} placeholders
        max_items: Maximum number of items to include in log
    
    Example:
        >>> users = ["user1", "user2", "user3"]
        >>> log_collection(
        ...     logger,
        ...     logging.INFO,
        ...     users,
        ...     "Processing {count} users: {items}",
        ...     max_items=2
        ... )
        # Logs: "Processing 3 users: ['user1', 'user2', '...']"
    """
    try:
        count = len(items)
        
        # Truncate items if too many
        if count > max_items:
            items_preview = list(items[:max_items]) + ['...']
        else:
            items_preview = list(items)
        
        message = message_template.format(
            count=count,
            items=items_preview
        )
        
        logger.log(level, message, extra={'item_count': count})
    
    except Exception as e:
        logger.warning(f"Failed to log collection: {e}")


# =========================================================
# Deprecation Warning Logger
# =========================================================

def log_deprecation(
    logger: logging.Logger,
    old_name: str,
    new_name: str,
    version: str
) -> None:
    """
    Log deprecation warning for old code/function.
    
    Args:
        logger: Logger instance
        old_name: Deprecated item name
        new_name: Replacement item name
        version: Version when deprecated
    
    Example:
        >>> log_deprecation(
        ...     logger,
        ...     "old_function",
        ...     "new_function",
        ...     "v2.0.0"
        ... )
        # Logs warning about deprecation
    """
    logger.warning(
        f"DEPRECATION: '{old_name}' is deprecated since {version}. "
        f"Use '{new_name}' instead.",
        extra={
            'deprecated_item': old_name,
            'replacement': new_name,
            'deprecation_version': version
        }
    )


if __name__ == "__main__":
    # Setup logging first
    from config.logging_config import setup_logging
    setup_logging(environment="development", debug=True)
    
    print("=" * 70)
    print("LOGGER UTILITIES DEMO")
    print("=" * 70)
    
    # Basic logger
    print("\n1️⃣ Basic Logger:")
    logger = get_logger(__name__)
    logger.info("This is a basic log message")
    
    # Logger with default context
    print("\n2️⃣ Logger with Default Context:")
    api_logger = get_logger(__name__, service="api", version="v1")
    api_logger.info("API request received")
    
    # Context manager
    print("\n3️⃣ Temporary Context:")
    with LogContext(logger, request_id="req_123", user_id="user_456"):
        logger.info("Processing request")
    logger.info("Outside context")
    
    # Execution time decorator
    print("\n4️⃣ Execution Time Tracking:")
    
    @log_execution_time(logger=logger)
    def slow_function():
        time.sleep(0.1)
        return "done"
    
    result = slow_function()
    
    # Performance tracking
    print("\n5️⃣ Performance Tracking:")
    with track_performance(logger, "data_processing", records=1000) as ctx:
        time.sleep(0.05)
        ctx['records_processed'] = 1000
        ctx['errors'] = 0
    
    # Collection logging
    print("\n6️⃣ Collection Logging:")
    items = [f"item_{i}" for i in range(15)]
    log_collection(
        logger,
        logging.INFO,
        items,
        "Processing {count} items: {items}",
        max_items=5
    )
    
    # Error logging
    print("\n7️⃣ Error Logging:")
    try:
        raise ValueError("Example error")
    except ValueError as e:
        log_error(logger, "Operation failed", error=e, user_id="user_123")
    
    print("\n✅ Logger utilities demo completed!")