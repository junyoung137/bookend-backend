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
    """

    _instance: Optional['DatabaseConnection'] = None
    _engine: Optional[Engine] = None
    _session_factory: Optional[sessionmaker] = None

    def __new__(cls) -> 'DatabaseConnection':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._engine is None:
            self._initialize_engine()

    def _initialize_engine(self) -> None:
        """Initialize SQLAlchemy engine."""
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
                    "application_name": "bookend_recommender",
                },
            )

            self._setup_event_listeners()

            self._session_factory = sessionmaker(
                bind=self._engine,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False,
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
        """SQLAlchemy engine event listeners."""

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
        Transactional scope for SQLAlchemy operations.
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
        """Check database connectivity and pool status."""
        result = {
            "healthy": False,
            "pool_size": None,
            "checked_out": None,
            "overflow": None,
            "error": None,
        }

        try:
            with self.session_scope() as session:
                session.execute(text("SELECT 1"))

            pool = self._engine.pool
            result.update(
                {
                    "healthy": True,
                    "pool_size": pool.size(),
                    "checked_out": pool.checkedout(),
                    "overflow": pool.overflow(),
                }
            )

        except Exception as e:
            result["error"] = str(e)

        return result

    def dispose(self) -> None:
        """Dispose database connections."""
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
    FastAPI dependency for request-scoped DB session.
    """
    db = get_db()
    session = db.get_session()

    try:
        yield session
    finally:
        session.close()


@contextmanager
def session_scope():
    """
    Module-level session_scope for compatibility.
    Internally uses the DatabaseConnection singleton.

    Allows:
        from config.database import session_scope
    """
    db = get_db()
    with db.session_scope() as session:
        yield session


if __name__ == "__main__":
    from config.logging_config import setup_logging

    setup_logging()

    db = get_db()
    print("Health:", db.health_check())

    with db.session_scope() as s:
        version = s.execute(text("SELECT version()")).scalar()
        print("PostgreSQL Version:", version)
