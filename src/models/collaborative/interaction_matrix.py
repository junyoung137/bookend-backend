from typing import Dict, Any, Optional, Tuple, List
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, dok_matrix
from sqlalchemy.orm import Session
from sqlalchemy import func

from src.database.models import User, Item, Interaction

logger = logging.getLogger(__name__)


class InteractionMatrixBuilder:
    """
    Build and manage user-item interaction matrices.
    
    Supports multiple weighting schemes:
    - binary: 1 if interaction exists, 0 otherwise
    - count: number of interactions
    - weighted: custom weights (e.g., recency-weighted)
    """
    
    def __init__(self, session: Session, config: Optional[Dict[str, Any]] = None):
        """
        Initialize matrix builder.
        
        Args:
            session: Database session
            config: Configuration dictionary
        """
        self.session = session
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Matrix components
        self.matrix: Optional[csr_matrix] = None
        self.user_id_to_idx: Dict[int, int] = {}
        self.idx_to_user_id: Dict[int, int] = {}
        self.item_id_to_idx: Dict[int, int] = {}
        self.idx_to_item_id: Dict[int, int] = {}
        
        # Metadata
        self.n_users: int = 0
        self.n_items: int = 0
        self.n_interactions: int = 0
        self.sparsity: float = 0.0
        self.last_build_time: Optional[datetime] = None
    
    def build_matrix(
        self,
        weighting: str = "binary",
        min_interactions: int = 1,
        lookback_days: Optional[int] = None,
        normalize: bool = False
    ) -> csr_matrix:
        """
        Build user-item interaction matrix from database.
        
        Args:
            weighting: Weighting scheme ('binary', 'count', 'weighted')
            min_interactions: Minimum interactions for user/item to include
            lookback_days: Only include interactions from last N days
            normalize: Whether to normalize the matrix
        
        Returns:
            Sparse CSR matrix (n_users x n_items)
        
        Example:
            >>> builder = InteractionMatrixBuilder(session)
            >>> matrix = builder.build_matrix(weighting='count', min_interactions=2)
        """
        try:
            self.logger.info(
                f"Building interaction matrix: weighting={weighting}, "
                f"min_interactions={min_interactions}, lookback_days={lookback_days}"
            )
            
            # Step 1: Get interactions from database
            interactions_df = self._fetch_interactions(lookback_days)
            
            if interactions_df.empty:
                self.logger.warning("No interactions found")
                return self._create_empty_matrix()
            
            self.logger.debug(f"Fetched {len(interactions_df)} interactions")
            
            # Step 2: Filter by minimum interactions
            interactions_df = self._filter_by_min_interactions(
                interactions_df,
                min_interactions
            )
            
            if interactions_df.empty:
                self.logger.warning(
                    f"No interactions after filtering (min={min_interactions})"
                )
                return self._create_empty_matrix()
            
            # Step 3: Build index mappings
            self._build_index_mappings(interactions_df)
            
            # Step 4: Create matrix with specified weighting
            matrix = self._create_matrix(interactions_df, weighting)
            
            # Step 5: Normalize if requested
            if normalize:
                matrix = self._normalize_matrix(matrix)
            
            # Step 6: Update metadata
            self._update_metadata(matrix)
            
            # Store matrix
            self.matrix = matrix
            self.last_build_time = datetime.now()
            
            self.logger.info(
                f"Matrix built: {self.n_users} users x {self.n_items} items, "
                f"sparsity={self.sparsity:.4f}"
            )
            
            return matrix
        
        except Exception as e:
            self.logger.error(f"Failed to build interaction matrix: {e}", exc_info=True)
            return self._create_empty_matrix()
    
    def get_user_vector(self, user_id: int) -> Optional[np.ndarray]:
        """
        Get interaction vector for a specific user.
        
        Args:
            user_id: User database ID
        
        Returns:
            User's interaction vector, or None if user not found
        
        Example:
            >>> builder = InteractionMatrixBuilder(session)
            >>> builder.build_matrix()
            >>> user_vec = builder.get_user_vector(user_id=123)
        """
        try:
            if self.matrix is None:
                self.logger.error("Matrix not built yet")
                return None
            
            if user_id not in self.user_id_to_idx:
                self.logger.debug(f"User {user_id} not in matrix")
                return None
            
            user_idx = self.user_id_to_idx[user_id]
            return self.matrix[user_idx, :].toarray().flatten()
        
        except Exception as e:
            self.logger.error(f"Failed to get user vector: {e}")
            return None
    
    def get_item_vector(self, item_id: int) -> Optional[np.ndarray]:
        """
        Get interaction vector for a specific item.
        
        Args:
            item_id: Item database ID
        
        Returns:
            Item's interaction vector, or None if item not found
        """
        try:
            if self.matrix is None:
                self.logger.error("Matrix not built yet")
                return None
            
            if item_id not in self.item_id_to_idx:
                self.logger.debug(f"Item {item_id} not in matrix")
                return None
            
            item_idx = self.item_id_to_idx[item_id]
            return self.matrix[:, item_idx].toarray().flatten()
        
        except Exception as e:
            self.logger.error(f"Failed to get item vector: {e}")
            return None
    
    def get_matrix_info(self) -> Dict[str, Any]:
        """
        Get matrix metadata and statistics.
        
        Returns:
            Dictionary containing matrix information
        """
        return {
            "n_users": self.n_users,
            "n_items": self.n_items,
            "n_interactions": self.n_interactions,
            "sparsity": self.sparsity,
            "density": 1.0 - self.sparsity,
            "matrix_shape": self.matrix.shape if self.matrix is not None else None,
            "last_build_time": self.last_build_time.isoformat() if self.last_build_time else None,
        }
    
    # =========================================================
    # Private Helper Methods
    # =========================================================
    
    def _fetch_interactions(
        self,
        lookback_days: Optional[int]
    ) -> pd.DataFrame:
        """Fetch interactions from database."""
        try:
            query = self.session.query(
                Interaction.user_id,
                Interaction.item_id,
                Interaction.event_name,
                Interaction.event_time
            )
            
            # Apply time filter if specified
            if lookback_days:
                cutoff_date = datetime.now() - timedelta(days=lookback_days)
                query = query.filter(Interaction.event_time >= cutoff_date)
            
            # Execute query
            results = query.all()
            
            if not results:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(results, columns=['user_id', 'item_id', 'event_name', 'event_time'])
            
            return df
        
        except Exception as e:
            self.logger.error(f"Failed to fetch interactions: {e}")
            return pd.DataFrame()
    
    def _filter_by_min_interactions(
        self,
        df: pd.DataFrame,
        min_interactions: int
    ) -> pd.DataFrame:
        """Filter users and items by minimum interaction count."""
        try:
            if min_interactions <= 1:
                return df
            
            # Count interactions per user
            user_counts = df['user_id'].value_counts()
            valid_users = user_counts[user_counts >= min_interactions].index
            
            # Count interactions per item
            item_counts = df['item_id'].value_counts()
            valid_items = item_counts[item_counts >= min_interactions].index
            
            # Filter DataFrame
            df_filtered = df[
                df['user_id'].isin(valid_users) &
                df['item_id'].isin(valid_items)
            ].copy()
            
            self.logger.debug(
                f"Filtered: {len(df)} -> {len(df_filtered)} interactions, "
                f"{len(valid_users)} users, {len(valid_items)} items"
            )
            
            return df_filtered
        
        except Exception as e:
            self.logger.error(f"Failed to filter interactions: {e}")
            return df
    
    def _build_index_mappings(self, df: pd.DataFrame) -> None:
        """Build mappings between database IDs and matrix indices."""
        try:
            # Get unique user and item IDs
            unique_user_ids = df['user_id'].unique()
            unique_item_ids = df['item_id'].unique()
            
            # Create mappings
            self.user_id_to_idx = {uid: idx for idx, uid in enumerate(unique_user_ids)}
            self.idx_to_user_id = {idx: uid for uid, idx in self.user_id_to_idx.items()}
            
            self.item_id_to_idx = {iid: idx for idx, iid in enumerate(unique_item_ids)}
            self.idx_to_item_id = {idx: iid for iid, idx in self.item_id_to_idx.items()}
            
            self.n_users = len(unique_user_ids)
            self.n_items = len(unique_item_ids)
            
            self.logger.debug(
                f"Built index mappings: {self.n_users} users, {self.n_items} items"
            )
        
        except Exception as e:
            self.logger.error(f"Failed to build index mappings: {e}")
            raise
    
    def _create_matrix(
        self,
        df: pd.DataFrame,
        weighting: str
    ) -> csr_matrix:
        """Create sparse interaction matrix with specified weighting."""
        try:
            # Initialize DOK matrix (efficient for construction)
            matrix = dok_matrix((self.n_users, self.n_items), dtype=np.float32)
            
            # Compute weights based on scheme
            if weighting == "binary":
                weights = self._compute_binary_weights(df)
            elif weighting == "count":
                weights = self._compute_count_weights(df)
            elif weighting == "weighted":
                weights = self._compute_custom_weights(df)
            else:
                self.logger.warning(f"Unknown weighting '{weighting}', using binary")
                weights = self._compute_binary_weights(df)
            
            # Fill matrix
            for (user_id, item_id), weight in weights.items():
                user_idx = self.user_id_to_idx[user_id]
                item_idx = self.item_id_to_idx[item_id]
                matrix[user_idx, item_idx] = weight
            
            # Convert to CSR format (efficient for operations)
            matrix_csr = matrix.tocsr()
            
            self.logger.debug(f"Created matrix with {matrix_csr.nnz} non-zero entries")
            
            return matrix_csr
        
        except Exception as e:
            self.logger.error(f"Failed to create matrix: {e}")
            raise
    
    def _compute_binary_weights(
        self,
        df: pd.DataFrame
    ) -> Dict[Tuple[int, int], float]:
        """Compute binary weights (1 if interaction exists)."""
        weights = {}
        for user_id, item_id in df[['user_id', 'item_id']].drop_duplicates().values:
            weights[(user_id, item_id)] = 1.0
        return weights
    
    def _compute_count_weights(
        self,
        df: pd.DataFrame
    ) -> Dict[Tuple[int, int], float]:
        """Compute count-based weights."""
        counts = df.groupby(['user_id', 'item_id']).size()
        weights = {(user_id, item_id): float(count) 
                   for (user_id, item_id), count in counts.items()}
        return weights
    
    def _compute_custom_weights(
        self,
        df: pd.DataFrame
    ) -> Dict[Tuple[int, int], float]:
        """Compute custom weights (e.g., recency-weighted)."""
        # Apply recency decay
        now = datetime.now()
        df['days_ago'] = (now - pd.to_datetime(df['event_time'])).dt.days
        df['recency_weight'] = np.exp(-df['days_ago'] / 30.0)  # 30-day half-life
        
        # Weight by event type
        event_weights = {
            'run_paraphrasing': 1.0,
            'selected_paraphrasing': 1.5,
            'copy_sentence': 2.0,
        }
        df['event_weight'] = df['event_name'].map(event_weights).fillna(1.0)
        
        # Combined weight
        df['weight'] = df['recency_weight'] * df['event_weight']
        
        # Aggregate by user-item pair
        weights_df = df.groupby(['user_id', 'item_id'])['weight'].sum()
        weights = {(user_id, item_id): float(weight) 
                   for (user_id, item_id), weight in weights_df.items()}
        
        return weights
    
    def _normalize_matrix(self, matrix: csr_matrix) -> csr_matrix:
        """Normalize matrix rows (L2 normalization)."""
        try:
            from sklearn.preprocessing import normalize
            
            matrix_normalized = normalize(matrix, norm='l2', axis=1)
            
            self.logger.debug("Matrix normalized (L2)")
            
            return matrix_normalized
        
        except Exception as e:
            self.logger.warning(f"Normalization failed: {e}, returning original matrix")
            return matrix
    
    def _update_metadata(self, matrix: csr_matrix) -> None:
        """Update matrix metadata."""
        try:
            self.n_interactions = matrix.nnz
            total_cells = self.n_users * self.n_items
            self.sparsity = 1.0 - (self.n_interactions / total_cells) if total_cells > 0 else 1.0
        
        except Exception as e:
            self.logger.error(f"Failed to update metadata: {e}")
    
    def _create_empty_matrix(self) -> csr_matrix:
        """Create an empty sparse matrix."""
        self.n_users = 0
        self.n_items = 0
        self.n_interactions = 0
        self.sparsity = 1.0
        return csr_matrix((0, 0), dtype=np.float32)


if __name__ == "__main__":
    from config.logging_config import setup_logging
    from src.database.postgres_singleton import get_postgres
    
    setup_logging()
    
    pg = get_postgres()
    
    with pg.transaction() as session:
        builder = InteractionMatrixBuilder(session)
        
        # Build matrix
        matrix = builder.build_matrix(
            weighting='count',
            min_interactions=2,
            lookback_days=90
        )
        
        # Print info
        info = builder.get_matrix_info()
        print("\n" + "="*70)
        print("📊 INTERACTION MATRIX INFO")
        print("="*70)
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Test user vector retrieval
        if builder.n_users > 0:
            first_user_id = builder.idx_to_user_id[0]
            user_vec = builder.get_user_vector(first_user_id)
            if user_vec is not None:
                print(f"\n✅ User {first_user_id} vector shape: {user_vec.shape}")
                print(f"   Non-zero items: {np.count_nonzero(user_vec)}")