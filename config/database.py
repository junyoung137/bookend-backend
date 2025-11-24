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
        """
        settings = get_settings()
        
        try:
            self._engine = create_engine(
                settings.database.url,
                poolclass=QueuePool,
                pool_size=settings.database.pool_size,
                max_overflow=settings.database.max_overflow,
                pool_timeout=settings.database.pool_timeout,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=settings.database.echo,
                connect_args={
                    "connect_timeout": 10,
                    "application_name": "bookend_recommender"
                }
            )
            
            # Event listeners
            self._setup_event_listeners()
            
            # Session factory
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
            logger.debug("New database connection established")
        
        @event.listens_for(self._engine, "checkout")
        def receive_checkout(dbapi_conn, connection_record, connection_proxy):
            logger.debug("Connection checked out from pool")
        
        @event.listens_for(self._engine, "checkin")
        def receive_checkin(dbapi_conn, connection_record):
            logger.debug("Connection returned to pool")
    
    @property
    def engine(self) -> Engine:
        if self._engine is None:
            self._initialize_engine()
        return self._engine
    
    @property
    def session_factory(self) -> sessionmaker:
        if self._session_factory is None:
            self._initialize_engine()
        return self._session_factory
    
    def get_session(self) -> Session:
        if self._session_factory is None:
            self._initialize_engine()
        return self._session_factory()
    
    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """
        Transaction scope wrapper.
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
            logger.debug("Database session closed")
    
    def health_check(self) -> dict[str, any]:
        """
        Check database connection health.
        """
        result = {
            "healthy": False,
            "pool_size": None,
            "checked_out": None,
            "overflow": None,
            "error": None
        }
        
        try:
            with self.session_scope() as session:
                session.execute(text("SELECT 1"))
            
            pool = self._engine.pool
            result.update({
                "healthy": True,
                "pool_size": pool.size(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow()
            })
            
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def dispose(self) -> None:
        """
        Dispose engine.
        """
        if self._engine:
            self._engine.dispose()
            logger.info("Database engine disposed")
            self._engine = None
            self._session_factory = None


# Singleton instance
_db_instance: Optional[DatabaseConnection] = None


def get_db() -> DatabaseConnection:
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseConnection()
    return _db_instance


def get_session() -> Generator[Session, None, None]:
    """
    FastAPI dependency — provides one session per request.
    """
    db = get_db()
    session = db.get_session()
    try:
        yield session
    finally:
        session.close()


if __name__ == "__main__":
    from config.logging_config import setup_logging
    setup_logging()
    
    db = get_db()
    
    print("Health:", db.health_check())
    
    with db.session_scope() as session:
        version = session.execute(text("SELECT version()")).scalar()
        print("PostgreSQL Version:", version)
