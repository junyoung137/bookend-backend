from typing import Optional, List, Dict, Any, Type, TypeVar
from contextlib import contextmanager
import logging

from sqlalchemy import create_engine, text, insert, update
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError, OperationalError

from config.settings import get_settings
from src.database.models import Base

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Base)


class PostgresSingleton:
    """
    Thread-safe PostgreSQL singleton with transaction management.
    Provides a unified interface for safe database operations.
    """

    _instance: Optional["PostgresSingleton"] = None

    def __new__(cls) -> "PostgresSingleton":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        settings = get_settings()
        self.engine = create_engine(
            settings.database.url,
            pool_size=settings.database.pool_size,
            max_overflow=settings.database.max_overflow,
            pool_pre_ping=True,
            echo=settings.database.echo,
            future=True,
        )
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)
        logger.info("✅ PostgreSQL engine and session factory initialized")

    # =========================================================
    # Transaction Context (수정됨)
    # =========================================================
    @contextmanager
    def transaction(self):
        """
        Safe transaction context manager.
        
        Usage:
            with get_postgres().transaction() as session:
                session.add(user)
                # 자동으로 commit되고 세션이 닫힘
        
        Raises:
            Exception: 트랜잭션 중 발생한 모든 예외를 다시 발생시킴
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
            logger.debug("Transaction committed successfully")
        except Exception as e:
            session.rollback()
            logger.error(f"Transaction error, rolled back: {e}", exc_info=True)
            raise
        finally:
            session.close()
            logger.debug("Session closed")

    @contextmanager
    def transaction_with_retry(self, retry_count: int = 3):
        """
        Transaction with retry logic for transient errors.
        
        Usage:
            with get_postgres().transaction_with_retry(retry_count=3) as session:
                session.add(user)
        
        Args:
            retry_count: 재시도 횟수 (기본값: 3)
        
        Raises:
            OperationalError: 재시도 후에도 실패 시
            Exception: 다른 모든 예외는 즉시 발생
        """
        last_exception = None
        
        for attempt in range(retry_count):
            session = self.SessionLocal()
            try:
                yield session
                session.commit()
                logger.debug(f"Transaction committed successfully (attempt {attempt + 1})")
                return
            except OperationalError as e:
                session.rollback()
                last_exception = e
                if attempt == retry_count - 1:
                    logger.error(f"Transaction failed after {retry_count} retries", exc_info=True)
                    raise
                logger.warning(f"Transient DB error, retrying ({attempt + 1}/{retry_count})... {e}")
            except Exception as e:
                session.rollback()
                logger.error(f"Transaction error (non-retryable): {e}", exc_info=True)
                raise
            finally:
                session.close()

    # =========================================================
    # Basic Helpers
    # =========================================================
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Run raw SQL safely and return results as dicts.
        
        Args:
            query: SQL 쿼리 문자열
            params: 쿼리 파라미터 딕셔너리
        
        Returns:
            List[Dict[str, Any]]: 쿼리 결과 리스트
        """
        with self.transaction() as session:
            result = session.execute(text(query), params or {})
            if result.returns_rows:
                columns = result.keys()
                return [dict(zip(columns, row)) for row in result.fetchall()]
            return []

    def bulk_insert(self, model: Type[T], records: List[Dict[str, Any]]) -> int:
        """
        Efficient bulk insert.
        
        Args:
            model: SQLAlchemy 모델 클래스
            records: 삽입할 레코드 리스트
        
        Returns:
            int: 삽입된 레코드 수
        """
        if not records:
            logger.warning("No records to insert")
            return 0
        
        with self.transaction() as session:
            session.bulk_insert_mappings(model, records)
            logger.info(f"✅ Inserted {len(records)} rows into {model.__tablename__}")
            return len(records)

    def upsert(self, model: Type[T], records: List[Dict[str, Any]], conflict_columns: List[str]) -> int:
        """
        Upsert (INSERT ... ON CONFLICT DO UPDATE).
        
        Args:
            model: SQLAlchemy 모델 클래스
            records: Upsert할 레코드 리스트
            conflict_columns: Conflict 체크할 컬럼 리스트
        
        Returns:
            int: 영향받은 레코드 수
        """
        if not records:
            logger.warning("No records to upsert")
            return 0
        
        table = model.__table__
        stmt = insert(table).values(records)
        update_cols = [
            c.name for c in table.columns 
            if c.name not in conflict_columns and c.name not in ["id", "created_at"]
        ]
        stmt = stmt.on_conflict_do_update(
            index_elements=conflict_columns,
            set_={c: getattr(stmt.excluded, c) for c in update_cols},
        )
        
        with self.transaction() as session:
            result = session.execute(stmt)
            affected_rows = result.rowcount or len(records)
            logger.info(f"✅ Upserted {affected_rows} rows into {table.name}")
            return affected_rows

    def health_check(self) -> Dict[str, Any]:
        """
        Simple DB connectivity check.
        
        Returns:
            Dict[str, Any]: 상태 정보 {"status": "ok"/"error", ...}
        """
        try:
            result = self.execute_query("SELECT 1 AS ok, current_database() AS db, version() AS version")
            return {
                "status": "ok",
                "database": result[0].get("db") if result else None,
                "connected": True
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "connected": False
            }

    def get_session(self) -> Session:
        """
        수동으로 세션을 가져옵니다. (context manager 사용 권장)
        
        Warning:
            반드시 사용 후 session.close()를 호출해야 합니다.
        
        Returns:
            Session: SQLAlchemy 세션
        """
        return self.SessionLocal()


# =========================================================
# Public Getter
# =========================================================
def get_postgres() -> PostgresSingleton:
    """
    Return global Postgres singleton instance.
    
    Returns:
        PostgresSingleton: PostgreSQL 싱글톤 인스턴스
    """
    return PostgresSingleton()


if __name__ == "__main__":
    from config.logging_config import setup_logging

    setup_logging()
    pg = get_postgres()
    
    print("\n" + "="*60)
    print("🔍 PostgreSQL Health Check")
    print("="*60)
    health = pg.health_check()
    print(f"Status: {health['status']}")
    print(f"Connected: {health.get('connected', False)}")
    if health.get('database'):
        print(f"Database: {health['database']}")
    
    print("\n" + "="*60)
    print("📊 Database Info")
    print("="*60)
    rows = pg.execute_query("SELECT current_database(), version()")
    if rows:
        print(f"Database: {rows[0].get('current_database')}")
        version = rows[0].get('version', '').split(',')[0]
        print(f"Version: {version}")
    
    print("\n" + "="*60)
    print("📋 Available Tables")
    print("="*60)
    tables = pg.execute_query("""
        SELECT tablename 
        FROM pg_tables 
        WHERE schemaname = 'public'
        ORDER BY tablename
    """)
    for i, table in enumerate(tables, 1):
        print(f"{i}. {table['tablename']}")
    
    print("\n✅ PostgreSQL connection test completed!")