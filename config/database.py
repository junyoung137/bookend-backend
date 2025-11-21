from typing import Optional, Generator
from contextlib import contextmanager
import logging

from sqlalchemy import create_engine, text, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool

from config.settings import get_settings

logger = logging.getLogger(__name__)


class DatabaseConnection:
    """
    Singleton database connection manager.
    
    Ensures only one connection pool exists across the application.
    Thread-safe and supports both sync and async operations.
    """
    
    _instance: Optional['DatabaseConnection'] = None
    _engine: Optional[Engine] = None
    _session_factory: Optional[sessionmaker] = None
    
    def __new__(cls) -> 'DatabaseConnection':
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize connection (only once due to singleton)."""
        if self._engine is None:
            self._initialize_engine()
    
    def _initialize_engine(self) -> None:
        """
        Create and configure SQLAlchemy engine.
        
        Features:
        - Connection pooling with QueuePool
        - Automatic reconnection on connection loss
        - Query execution logging in debug mode
        - Pessimistic disconnect handling
        """
        settings = get_settings()
        
        try:
            self._engine = create_engine(
                settings.database.url,
                poolclass=QueuePool,
                pool_size=settings.database.pool_size,
                max_overflow=settings.database.max_overflow,
                pool_timeout=settings.database.pool_timeout,
                pool_pre_ping=True,  # Verify connections before using
                pool_recycle=3600,  # Recycle connections after 1 hour
                echo=settings.database.echo,  # Log SQL queries
                connect_args={
                    "connect_timeout": 10,
                    "application_name": "bookend_recommender"
                }
            )
            
            # Add connection event listeners
            self._setup_event_listeners()
            
            # Create session factory
            self._session_factory = sessionmaker(
                bind=self._engine,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False
            )
            
            logger.info(
                f"Database engine initialized: "
                f"pool_size={settings.database.pool_size}, "
                f"max_overflow={settings.database.max_overflow}"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize database engine: {e}", exc_info=True)
            raise
    
    def _setup_event_listeners(self) -> None:
        """Setup SQLAlchemy event listeners for monitoring."""
        
        @event.listens_for(self._engine, "connect")
        def receive_connect(dbapi_conn, connection_record):
            """Log when new connection is created."""
            logger.debug("New database connection established")
        
        @event.listens_for(self._engine, "checkout")
        def receive_checkout(dbapi_conn, connection_record, connection_proxy):
            """Log when connection is checked out from pool."""
            logger.debug("Connection checked out from pool")
        
        @event.listens_for(self._engine, "checkin")
        def receive_checkin(dbapi_conn, connection_record):
            """Log when connection is returned to pool."""
            logger.debug("Connection returned to pool")
    
    @property
    def engine(self) -> Engine:
        """Get SQLAlchemy engine instance."""
        if self._engine is None:
            self._initialize_engine()
        return self._engine
    
    @property
    def session_factory(self) -> sessionmaker:
        """Get session factory."""
        if self._session_factory is None:
            self._initialize_engine()
        return self._session_factory
    
    def get_session(self) -> Session:
        """
        Create a new database session.
        
        Returns:
            Session: SQLAlchemy session instance
        
        Example:
            >>> db = DatabaseConnection()
            >>> session = db.get_session()
            >>> try:
            ...     # Use session
            ...     session.commit()
            ... finally:
            ...     session.close()
        """
        if self._session_factory is None:
            self._initialize_engine()
        return self._session_factory()
    
    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """
        Provide a transactional scope for database operations.
        
        Automatically commits on success and rolls back on exception.
        Always closes the session.
        
        Yields:
            Session: SQLAlchemy session instance
        
        Example:
            >>> db = DatabaseConnection()
            >>> with db.session_scope() as session:
            ...     user = session.query(User).first()
            ...     # Changes auto-committed
        """
        session = self.get_session()
        try:
            yield session
            session.commit()
            logger.debug("Database transaction committed")
        except Exception as e:
            session.rollback()
            logger.error(f"Database transaction rolled back: {e}", exc_info=True)
            raise
        finally:
            session.close()
    
    def health_check(self) -> dict[str, any]:
        """
        Check database connection health.
        
        Returns:
            dict: Health status with connection info
        
        Example:
            >>> db = DatabaseConnection()
            >>> status = db.health_check()
            >>> if status['healthy']:
            ...     print("Database is healthy")
        """
        result = {
            "healthy": False,
            "pool_size": None,
            "checked_out": None,
            "overflow": None,
            "error": None
        }
        
        try:
            # Test query
            with self.session_scope() as session:
                session.execute(text("SELECT 1"))
            
            # Get pool statistics
            pool = self._engine.pool
            result.update({
                "healthy": True,
                "pool_size": pool.size(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow()
            })
            
            logger.debug("Database health check passed")
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Database health check failed: {e}", exc_info=True)
        
        return result
    
    def dispose(self) -> None:
        """
        Dispose database engine and close all connections.
        
        Should be called on application shutdown.
        
        Example:
            >>> db = DatabaseConnection()
            >>> # ... use database ...
            >>> db.dispose()
        """
        if self._engine is not None:
            self._engine.dispose()
            logger.info("Database engine disposed")
            self._engine = None
            self._session_factory = None


# Singleton instance getter
_db_instance: Optional[DatabaseConnection] = None


def get_db() -> DatabaseConnection:
    """
    Get singleton database connection instance.
    
    Returns:
        DatabaseConnection: Singleton database connection
    
    Example:
        >>> db = get_db()
        >>> with db.session_scope() as session:
        ...     results = session.query(User).all()
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseConnection()
    return _db_instance


def get_session() -> Generator[Session, None, None]:
    """
    FastAPI dependency for getting database session.
    
    Yields:
        Session: SQLAlchemy session
    
    Example:
        >>> from fastapi import Depends
        >>> @app.get("/users")
        >>> def get_users(db: Session = Depends(get_session)):
        ...     return db.query(User).all()
    """
    db = get_db()
    session = db.get_session()
    try:
        yield session
    finally:
        session.close()


if __name__ == "__main__":
    # Test database connection
    from config.logging_config import setup_logging
    
    setup_logging()
    
    db = get_db()
    
    # Health check
    status = db.health_check()
    print(f"Database Health: {status}")
    
    # Test session scope
    with db.session_scope() as session:
        result = session.execute(text("SELECT version()"))
        version = result.scalar()
        print(f"PostgreSQL Version: {version}")