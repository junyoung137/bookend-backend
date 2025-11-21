
import sys
import logging
import logging.handlers
from pathlib import Path
from typing import Optional
from datetime import datetime

from pythonjsonlogger import jsonlogger


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """
    Custom JSON formatter with additional metadata.
    
    Adds timestamp, level, module, and function name to each log record.
    """
    
    def add_fields(self, log_record: dict, record: logging.LogRecord, message_dict: dict) -> None:
        """Add custom fields to log record."""
        super().add_fields(log_record, record, message_dict)
        
        # Add timestamp
        log_record["timestamp"] = datetime.utcnow().isoformat()
        
        # Add level name
        log_record["level"] = record.levelname
        
        # Add location info
        log_record["module"] = record.module
        log_record["function"] = record.funcName
        log_record["line"] = record.lineno
        
        # Add process info
        log_record["process_id"] = record.process
        log_record["thread_id"] = record.thread


def _create_console_handler(level: int, use_json: bool = False) -> logging.Handler:
    """
    Create console (stdout) handler.
    
    Args:
        level: Logging level
        use_json: Whether to use JSON formatting
    
    Returns:
        logging.Handler: Configured console handler
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    if use_json:
        formatter = CustomJsonFormatter(
            "%(timestamp)s %(level)s %(name)s %(message)s"
        )
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    handler.setFormatter(formatter)
    return handler


def _create_file_handler(
    log_file: Path,
    level: int,
    use_json: bool = False,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Handler:
    """
    Create rotating file handler.
    
    Args:
        log_file: Path to log file
        level: Logging level
        use_json: Whether to use JSON formatting
        max_bytes: Maximum file size before rotation
        backup_count: Number of backup files to keep
    
    Returns:
        logging.Handler: Configured file handler
    """
    # Ensure log directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    handler = logging.handlers.RotatingFileHandler(
        filename=str(log_file),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8"
    )
    handler.setLevel(level)
    
    if use_json:
        formatter = CustomJsonFormatter(
            "%(timestamp)s %(level)s %(name)s %(message)s"
        )
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    handler.setFormatter(formatter)
    return handler


def _get_log_level(environment: str, debug: bool) -> int:
    """
    Determine log level based on environment and debug flag.
    
    Args:
        environment: Current environment (development/staging/production)
        debug: Debug mode flag
    
    Returns:
        int: Logging level constant
    """
    if debug:
        return logging.DEBUG
    
    level_map = {
        "development": logging.DEBUG,
        "staging": logging.INFO,
        "production": logging.WARNING
    }
    
    return level_map.get(environment.lower(), logging.INFO)


def setup_logging(
    environment: str = "development",
    debug: bool = True,
    logs_dir: str = "logs",
    app_name: str = "bookend"
) -> None:
    """
    Setup centralized logging configuration.
    
    Creates separate log files for:
    - app.log: General application logs
    - error.log: Error-level logs only
    - api.log: API request/response logs
    - pipeline.log: Data pipeline logs
    
    Args:
        environment: Current environment
        debug: Enable debug mode
        logs_dir: Directory for log files
        app_name: Application name for log files
    
    Example:
        >>> setup_logging(environment="production", debug=False)
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started")
    """
    logs_path = Path(logs_dir)
    logs_path.mkdir(parents=True, exist_ok=True)
    
    # Determine settings
    log_level = _get_log_level(environment, debug)
    use_json = environment.lower() == "production"
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all levels
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Add console handler
    console_handler = _create_console_handler(log_level, use_json)
    root_logger.addHandler(console_handler)
    
    # Add general application log file
    app_log_file = logs_path / f"{app_name}.log"
    app_file_handler = _create_file_handler(app_log_file, log_level, use_json)
    root_logger.addHandler(app_file_handler)
    
    # Add error-only log file
    error_log_file = logs_path / f"{app_name}_error.log"
    error_file_handler = _create_file_handler(error_log_file, logging.ERROR, use_json)
    root_logger.addHandler(error_file_handler)
    
    # Configure specific loggers
    _configure_component_loggers(logs_path, log_level, use_json, app_name)
    
    # Suppress noisy third-party loggers
    _suppress_third_party_loggers()
    
    # Log setup completion
    logger = logging.getLogger(__name__)
    logger.info(
        f"Logging configured: environment={environment}, level={logging.getLevelName(log_level)}, "
        f"json={use_json}, logs_dir={logs_dir}"
    )


def _configure_component_loggers(
    logs_path: Path,
    log_level: int,
    use_json: bool,
    app_name: str
) -> None:
    """
    Configure loggers for specific components.
    
    Args:
        logs_path: Path to logs directory
        log_level: Logging level
        use_json: Whether to use JSON formatting
        app_name: Application name
    """
    components = {
        "src.api": f"{app_name}_api.log",
        "src.pipeline": f"{app_name}_pipeline.log",
        "src.models": f"{app_name}_models.log",
    }
    
    for logger_name, log_file_name in components.items():
        logger = logging.getLogger(logger_name)
        log_file = logs_path / log_file_name
        file_handler = _create_file_handler(log_file, log_level, use_json)
        logger.addHandler(file_handler)


def _suppress_third_party_loggers() -> None:
    """Suppress noisy third-party library loggers."""
    noisy_loggers = [
        "urllib3",
        "requests",
        "botocore",
        "boto3",
        "minio",
        "sqlalchemy.engine",
        "asyncio",
    ]
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    Args:
        name: Module name (typically __name__)
    
    Returns:
        logging.Logger: Configured logger instance
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
        >>> logger.error("An error occurred", exc_info=True)
    """
    return logging.getLogger(name)


class LoggerAdapter:
    """
    Context-aware logger adapter for adding request/user context.
    
    Example:
        >>> adapter = LoggerAdapter(logger, user_id="user123", request_id="req456")
        >>> adapter.info("User action completed")
    """
    
    def __init__(self, logger: logging.Logger, **context):
        """
        Initialize adapter with context.
        
        Args:
            logger: Base logger instance
            **context: Additional context fields (user_id, request_id, etc.)
        """
        self.logger = logger
        self.context = context
    
    def _log(self, level: int, msg: str, *args, **kwargs) -> None:
        """Internal log method with context injection."""
        extra = kwargs.get("extra", {})
        extra.update(self.context)
        kwargs["extra"] = extra
        self.logger.log(level, msg, *args, **kwargs)
    
    def debug(self, msg: str, *args, **kwargs) -> None:
        """Log debug message with context."""
        self._log(logging.DEBUG, msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs) -> None:
        """Log info message with context."""
        self._log(logging.INFO, msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs) -> None:
        """Log warning message with context."""
        self._log(logging.WARNING, msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs) -> None:
        """Log error message with context."""
        self._log(logging.ERROR, msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs) -> None:
        """Log critical message with context."""
        self._log(logging.CRITICAL, msg, *args, **kwargs)


if __name__ == "__main__":
    # Test logging setup
    setup_logging(environment="development", debug=True)
    
    logger = get_logger(__name__)
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test context adapter
    adapter = LoggerAdapter(logger, user_id="test_user", request_id="req_123")
    adapter.info("User action completed successfully")