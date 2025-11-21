from typing import Generic, TypeVar, Type, Optional, List, Dict, Any
from abc import ABC, abstractmethod
import logging

from sqlalchemy import select, update, delete, func
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

from ..base import Base

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=Base)


class BaseRepository(ABC, Generic[T]):
    """
    Abstract base repository with common CRUD operations.
    
    Type parameter T must be a SQLAlchemy model inheriting from Base.
    """
    
    def __init__(self, model: Type[T], session: Session):
        """
        Initialize repository.
        
        Args:
            model: SQLAlchemy model class
            session: Database session
        """
        self.model = model
        self.session = session
        self.logger = logging.getLogger(f"{__name__}.{model.__name__}Repository")
    
    def create(self, **kwargs) -> Optional[T]:
        """
        Create a new record.
        
        Args:
            **kwargs: Model attributes
        
        Returns:
            Created model instance or None on failure
        
        Example:
            >>> repo = UserRepository(session)
            >>> user = repo.create(name="Alice", email="alice@example.com")
        """
        try:
            instance = self.model(**kwargs)
            self.session.add(instance)
            self.session.flush()  # Get ID without committing
            self.logger.debug(f"Created {self.model.__name__} with id={instance.id}")
            return instance
        except IntegrityError as e:
            self.logger.error(f"Integrity error creating {self.model.__name__}: {e}")
            self.session.rollback()
            raise
        except SQLAlchemyError as e:
            self.logger.error(f"Database error creating {self.model.__name__}: {e}")
            self.session.rollback()
            return None
    
    def get_by_id(self, record_id: int) -> Optional[T]:
        """
        Get record by ID.
        
        Args:
            record_id: Record ID
        
        Returns:
            Model instance or None if not found
        
        Example:
            >>> repo = UserRepository(session)
            >>> user = repo.get_by_id(123)
        """
        try:
            return self.session.get(self.model, record_id)
        except SQLAlchemyError as e:
            self.logger.error(f"Error fetching {self.model.__name__} by id={record_id}: {e}")
            return None
    
    def get_by_field(self, field_name: str, field_value: Any) -> Optional[T]:
        """
        Get single record by field value.
        
        Args:
            field_name: Model attribute name
            field_value: Value to match
        
        Returns:
            Model instance or None
        
        Example:
            >>> repo = UserRepository(session)
            >>> user = repo.get_by_field("email", "alice@example.com")
        """
        try:
            stmt = select(self.model).where(
                getattr(self.model, field_name) == field_value
            )
            result = self.session.execute(stmt)
            return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            self.logger.error(f"Error fetching {self.model.__name__} by {field_name}: {e}")
            return None
    
    def get_all(self, limit: Optional[int] = None, offset: int = 0) -> List[T]:
        """
        Get all records with pagination.
        
        Args:
            limit: Maximum number of records
            offset: Number of records to skip
        
        Returns:
            List of model instances
        
        Example:
            >>> repo = UserRepository(session)
            >>> users = repo.get_all(limit=10, offset=20)
        """
        try:
            stmt = select(self.model).offset(offset)
            if limit:
                stmt = stmt.limit(limit)
            result = self.session.execute(stmt)
            return list(result.scalars().all())
        except SQLAlchemyError as e:
            self.logger.error(f"Error fetching all {self.model.__name__}: {e}")
            return []
    
    def filter_by(self, **kwargs) -> List[T]:
        """
        Filter records by multiple fields.
        
        Args:
            **kwargs: Field name-value pairs
        
        Returns:
            List of matching model instances
        
        Example:
            >>> repo = UserRepository(session)
            >>> active_users = repo.filter_by(is_active=True, role=1)
        """
        try:
            stmt = select(self.model).filter_by(**kwargs)
            result = self.session.execute(stmt)
            return list(result.scalars().all())
        except SQLAlchemyError as e:
            self.logger.error(f"Error filtering {self.model.__name__}: {e}")
            return []
    
    def update(self, record_id: int, **kwargs) -> Optional[T]:
        """
        Update record by ID.
        
        Args:
            record_id: Record ID
            **kwargs: Fields to update
        
        Returns:
            Updated model instance or None
        
        Example:
            >>> repo = UserRepository(session)
            >>> user = repo.update(123, name="Alice Updated", age=31)
        """
        try:
            instance = self.get_by_id(record_id)
            if not instance:
                self.logger.warning(f"{self.model.__name__} with id={record_id} not found")
                return None
            
            for key, value in kwargs.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)
            
            self.session.flush()
            self.logger.debug(f"Updated {self.model.__name__} with id={record_id}")
            return instance
        except SQLAlchemyError as e:
            self.logger.error(f"Error updating {self.model.__name__} id={record_id}: {e}")
            self.session.rollback()
            return None
    
    def delete(self, record_id: int) -> bool:
        """
        Delete record by ID.
        
        Args:
            record_id: Record ID
        
        Returns:
            True if deleted, False otherwise
        
        Example:
            >>> repo = UserRepository(session)
            >>> success = repo.delete(123)
        """
        try:
            instance = self.get_by_id(record_id)
            if not instance:
                self.logger.warning(f"{self.model.__name__} with id={record_id} not found")
                return False
            
            self.session.delete(instance)
            self.session.flush()
            self.logger.debug(f"Deleted {self.model.__name__} with id={record_id}")
            return True
        except SQLAlchemyError as e:
            self.logger.error(f"Error deleting {self.model.__name__} id={record_id}: {e}")
            self.session.rollback()
            return False
    
    def count(self, **filters) -> int:
        """
        Count records matching filters.
        
        Args:
            **filters: Optional filter conditions
        
        Returns:
            Number of matching records
        
        Example:
            >>> repo = UserRepository(session)
            >>> active_count = repo.count(is_active=True)
        """
        try:
            stmt = select(func.count()).select_from(self.model)
            if filters:
                stmt = stmt.filter_by(**filters)
            result = self.session.execute(stmt)
            return result.scalar() or 0
        except SQLAlchemyError as e:
            self.logger.error(f"Error counting {self.model.__name__}: {e}")
            return 0
    
    def exists(self, record_id: int) -> bool:
        """
        Check if record exists by ID.
        
        Args:
            record_id: Record ID
        
        Returns:
            True if exists, False otherwise
        
        Example:
            >>> repo = UserRepository(session)
            >>> if repo.exists(123):
            ...     print("User exists")
        """
        try:
            stmt = select(func.count()).select_from(self.model).where(
                self.model.id == record_id
            )
            result = self.session.execute(stmt)
            return (result.scalar() or 0) > 0
        except SQLAlchemyError as e:
            self.logger.error(f"Error checking existence of {self.model.__name__}: {e}")
            return False
    
    def bulk_create(self, records: List[Dict[str, Any]]) -> int:
        """
        Create multiple records efficiently.
        
        Args:
            records: List of record dictionaries
        
        Returns:
            Number of created records
        
        Example:
            >>> repo = UserRepository(session)
            >>> users = [
            ...     {"name": "Alice", "email": "alice@example.com"},
            ...     {"name": "Bob", "email": "bob@example.com"}
            ... ]
            >>> count = repo.bulk_create(users)
        """
        if not records:
            return 0
        
        try:
            instances = [self.model(**record) for record in records]
            self.session.add_all(instances)
            self.session.flush()
            self.logger.info(f"Bulk created {len(instances)} {self.model.__name__} records")
            return len(instances)
        except IntegrityError as e:
            self.logger.error(f"Integrity error in bulk create: {e}")
            self.session.rollback()
            raise
        except SQLAlchemyError as e:
            self.logger.error(f"Error in bulk create: {e}")
            self.session.rollback()
            return 0
    
    def bulk_update_by_id(self, updates: List[Dict[str, Any]]) -> int:
        """
        Update multiple records by ID.
        
        Args:
            updates: List of dictionaries with 'id' and fields to update
        
        Returns:
            Number of updated records
        
        Example:
            >>> repo = UserRepository(session)
            >>> updates = [
            ...     {"id": 1, "status": "active"},
            ...     {"id": 2, "status": "inactive"}
            ... ]
            >>> count = repo.bulk_update_by_id(updates)
        """
        if not updates:
            return 0
        
        try:
            updated_count = 0
            for update_data in updates:
                if 'id' not in update_data:
                    continue
                
                record_id = update_data.pop('id')
                stmt = update(self.model).where(
                    self.model.id == record_id
                ).values(**update_data)
                result = self.session.execute(stmt)
                updated_count += result.rowcount
            
            self.session.flush()
            self.logger.info(f"Bulk updated {updated_count} {self.model.__name__} records")
            return updated_count
        except SQLAlchemyError as e:
            self.logger.error(f"Error in bulk update: {e}")
            self.session.rollback()
            return 0
    
    @abstractmethod
    def get_repository_name(self) -> str:
        """
        Get repository name for logging.
        
        Must be implemented by concrete repositories.
        """
        pass


class RepositoryError(Exception):
    """Base exception for repository operations."""
    pass


class RecordNotFoundError(RepositoryError):
    """Raised when a record is not found."""
    pass


class DuplicateRecordError(RepositoryError):
    """Raised when attempting to create a duplicate record."""
    pass


if __name__ == "__main__":
    # Example usage
    print("BaseRepository is an abstract class.")
    print("Create concrete repositories by inheriting from BaseRepository.")
    print("\nExample:")
    print("""
    class UserRepository(BaseRepository[User]):
        def get_repository_name(self) -> str:
            return "UserRepository"
        
        def get_by_email(self, email: str) -> Optional[User]:
            return self.get_by_field("email", email)
    """)