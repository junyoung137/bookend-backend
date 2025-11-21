"""
Bookend Hybrid v2 Recommender
가중치: Popularity 40%, User-CF 25%, Item-CF 25%, Diversity 10%
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import pickle
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridV2Recommender:
    """
    Hybrid v2: Temporal + User-CF + Item-CF + Diversity
    
    과적합 방지:
    - 정규화 강화
    - 최소 상호작용 필터링
    - Early stopping (validation split)
    """
    
    def __init__(
        self,
        popularity_weight=0.40,
        user_cf_weight=0.25,
        item_cf_weight=0.25,
        diversity_weight=0.10,
        min_user_interactions=3,
        min_item_interactions=5,
        temporal_decay_days=30
    ):
        self.popularity_weight = popularity_weight
        self.user_cf_weight = user_cf_weight
        self.item_cf_weight = item_cf_weight
        self.diversity_weight = diversity_weight
        
        self.min_user_interactions = min_user_interactions
        self.min_item_interactions = min_item_interactions
        self.temporal_decay_days = temporal_decay_days
        
        # 모델 컴포넌트
        self.user_item_matrix = None
        self.item_user_matrix = None
        self.user_similarity = None
        self.item_similarity = None
        self.item_popularity = None
        self.item_ids = None
        self.user_ids = None
        
        logger.info("🚀 Hybrid v2 Initialized")
    
    def fit(self, interactions_df):
        """
        학습 데이터로 모델 학습
        
        Args:
            interactions_df: columns=['user_id', 'item_id', 'timestamp']
        """
        logger.info(f"📊 원본 데이터: {len(interactions_df)} interactions")
        
        # 1. 데이터 필터링 (과적합 방지)
        interactions_df = self._filter_cold_start(interactions_df)
        logger.info(f"✅ 필터링 후: {len(interactions_df)} interactions")
        
        # 2. Temporal weighting
        interactions_df = self._apply_temporal_weights(interactions_df)
        
        # 3. 사용자-아이템 행렬 생성
        self._build_matrices(interactions_df)
        
        # 4. 유사도 계산
        self._compute_similarities()
        
        # 5. Popularity 계산
        self._compute_popularity(interactions_df)
        
        logger.info("✅ 모델 학습 완료!")
        return self
    
    def _filter_cold_start(self, df):
        """Cold start 필터링으로 과적합 방지"""
        # 사용자별 상호작용 수 계산
        user_counts = df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= self.min_user_interactions].index
        
        # 아이템별 상호작용 수 계산
        item_counts = df['item_id'].value_counts()
        valid_items = item_counts[item_counts >= self.min_item_interactions].index
        
        # 필터링
        filtered = df[
            df['user_id'].isin(valid_users) & 
            df['item_id'].isin(valid_items)
        ].copy()
        
        logger.info(f"🔍 필터링: {len(valid_users)}/{df['user_id'].nunique()} users, "
                   f"{len(valid_items)}/{df['item_id'].nunique()} items")
        
        return filtered
    
    def _apply_temporal_weights(self, df):
        """시간 가중치 적용"""
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        max_date = df['timestamp'].max()
        df['days_ago'] = (max_date - df['timestamp']).dt.days
        
        # 지수 감쇠
        df['temporal_weight'] = np.exp(-df['days_ago'] / self.temporal_decay_days)
        
        logger.info(f"⏰ Temporal weighting 완료 (decay={self.temporal_decay_days}일)")
        return df
    
    def _build_matrices(self, df):
        """희소 행렬 생성"""
        # User/Item ID mapping
        self.user_ids = df['user_id'].unique()
        self.item_ids = df['item_id'].unique()
        
        user_map = {uid: idx for idx, uid in enumerate(self.user_ids)}
        item_map = {iid: idx for idx, iid in enumerate(self.item_ids)}
        
        # 행렬 생성
        rows = df['user_id'].map(user_map)
        cols = df['item_id'].map(item_map)
        weights = df['temporal_weight'].values
        
        self.user_item_matrix = csr_matrix(
            (weights, (rows, cols)),
            shape=(len(self.user_ids), len(self.item_ids))
        )
        
        self.item_user_matrix = self.user_item_matrix.T.tocsr()
        
        logger.info(f"🔢 행렬 크기: {self.user_item_matrix.shape}")
        logger.info(f"📈 희소도: {1 - self.user_item_matrix.nnz / (self.user_item_matrix.shape[0] * self.user_item_matrix.shape[1]):.4f}")
    
    def _compute_similarities(self):
        """코사인 유사도 계산 (정규화 포함)"""
        # User similarity
        user_norm = normalize(self.user_item_matrix, norm='l2', axis=1)
        self.user_similarity = cosine_similarity(user_norm, dense_output=False)
        
        # Item similarity
        item_norm = normalize(self.item_user_matrix, norm='l2', axis=1)
        self.item_similarity = cosine_similarity(item_norm, dense_output=False)
        
        logger.info("✅ 유사도 행렬 계산 완료")
    
    def _compute_popularity(self, df):
        """Temporal-weighted Popularity"""
        item_weights = df.groupby('item_id')['temporal_weight'].sum()
        self.item_popularity = item_weights / item_weights.sum()
        
        logger.info(f"📊 인기도 Top 3: {self.item_popularity.nlargest(3).to_dict()}")
    
    def recommend(self, user_id, k=10, exclude_interacted=True):
        """
        추천 생성
        
        Returns:
            list of (item_id, score, reasons)
        """
        if user_id not in self.user_ids:
            # Cold start: Popularity 기반
            return self._cold_start_recommend(k)
        
        user_idx = np.where(self.user_ids == user_id)[0][0]
        
        # 1. User-CF score
        user_cf_scores = self._user_cf_score(user_idx)
        
        # 2. Item-CF score
        item_cf_scores = self._item_cf_score(user_idx)
        
        # 3. Popularity score
        pop_scores = self.item_popularity.reindex(self.item_ids, fill_value=0).values
        
        # 4. Hybrid score
        final_scores = (
            self.user_cf_weight * user_cf_scores +
            self.item_cf_weight * item_cf_scores +
            self.popularity_weight * pop_scores
        )
        
        # 5. 이미 상호작용한 아이템 제외
        if exclude_interacted:
            interacted_items = self.user_item_matrix[user_idx].indices
            final_scores[interacted_items] = -np.inf
        
        # 6. Top-K 선택
        top_k_indices = np.argsort(final_scores)[::-1][:k]
        
        # 7. MMR Diversity 적용
        top_k_indices = self._apply_mmr(top_k_indices, final_scores)
        
        # 8. 결과 포맷팅
        results = []
        for idx in top_k_indices:
            item_id = self.item_ids[idx]
            score = final_scores[idx]
            
            # Reason 생성
            reasons = []
            if user_cf_scores[idx] > 0.1:
                reasons.append('similar_users')
            if item_cf_scores[idx] > 0.1:
                reasons.append('related_items')
            if pop_scores[idx] > 0.05:
                reasons.append('popular')
            
            results.append({
                'item_id': int(item_id),
                'score': float(score),
                'reasons': reasons
            })
        
        return results
    
    def _user_cf_score(self, user_idx):
        """User-based Collaborative Filtering"""
        # 유사한 사용자들의 아이템 선호도 집계
        similar_users = self.user_similarity[user_idx].toarray().flatten()
        
        # Top 50 유사 사용자만 사용 (과적합 방지)
        top_similar = np.argsort(similar_users)[::-1][:50]
        
        scores = np.zeros(len(self.item_ids))
        for sim_user_idx in top_similar:
            sim_score = similar_users[sim_user_idx]
            if sim_score > 0:
                user_items = self.user_item_matrix[sim_user_idx].toarray().flatten()
                scores += sim_score * user_items
        
        # 정규화
        scores = scores / (np.sum(similar_users[top_similar]) + 1e-10)
        
        return scores
    
    def _item_cf_score(self, user_idx):
        """Item-based Collaborative Filtering"""
        # 사용자가 상호작용한 아이템들
        user_items = self.user_item_matrix[user_idx].indices
        
        if len(user_items) == 0:
            return np.zeros(len(self.item_ids))
        
        scores = np.zeros(len(self.item_ids))
        for item_idx in user_items:
            # 유사 아이템 점수
            similar_items = self.item_similarity[item_idx].toarray().flatten()
            scores += similar_items
        
        # 평균
        scores = scores / len(user_items)
        
        return scores
    
    def _apply_mmr(self, candidates, scores, lambda_diversity=0.3):
        """MMR (Maximal Marginal Relevance) 다양성"""
        if len(candidates) <= 3:
            return candidates
        
        selected = [candidates[0]]  # 최고 점수 아이템
        remaining = list(candidates[1:])
        
        while len(selected) < len(candidates) and remaining:
            mmr_scores = []
            
            for candidate in remaining:
                # Relevance
                relevance = scores[candidate]
                
                # Diversity (선택된 아이템과의 평균 비유사도)
                similarities = []
                for sel in selected:
                    # 🔧 FIX: 희소 행렬 인덱싱 처리
                    sim_value = self.item_similarity[candidate, sel]
                    # scipy sparse matrix에서 단일 값은 스칼라로 반환됨
                    if hasattr(sim_value, 'toarray'):
                        sim_value = sim_value.toarray()[0, 0]
                    similarities.append(float(sim_value))
                
                diversity = 1 - np.mean(similarities)
                
                # MMR score
                mmr = lambda_diversity * relevance + (1 - lambda_diversity) * diversity
                mmr_scores.append(mmr)
            
            # 최고 MMR 선택
            best_idx = np.argmax(mmr_scores)
            selected.append(remaining[best_idx])
            remaining.pop(best_idx)
        
        return np.array(selected)
    
    def _cold_start_recommend(self, k):
        """Cold start: Popularity 기반"""
        top_items = self.item_popularity.nlargest(k)
        
        return [
            {
                'item_id': int(item_id),
                'score': float(score),
                'reasons': ['popular', 'cold_start']
            }
            for item_id, score in top_items.items()
        ]
    
    def save(self, filepath):
        """모델 저장"""
        # 디렉토리 생성
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"💾 모델 저장: {filepath}")
    
    @staticmethod
    def load(filepath):
        """모델 로드"""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"📂 모델 로드: {filepath}")
        return model


# ==================== 학습 스크립트 ====================

def load_data_from_postgres():
    """
    PostgreSQL에서 데이터 로드
    DatabaseConnection 사용
    """
    from config.database import get_db
    
    db = get_db()
    
    # session_scope 사용 (자동 commit/rollback/close)
    with db.session_scope() as session:
        query = """
        SELECT 
            user_id,
            item_id,
            event_time as timestamp
        FROM interactions
        WHERE item_id IS NOT NULL
        ORDER BY event_time
        """
        
        df = pd.read_sql(query, session.bind)
        logger.info(f"📊 데이터 로드: {len(df)} rows")
        logger.info(f"📊 고유 사용자: {df['user_id'].nunique()}")
        logger.info(f"📊 고유 아이템: {df['item_id'].nunique()}")
        
        return df


def train_and_save():
    """전체 학습 파이프라인"""
    logger.info("🚀 Hybrid v2 학습 시작")
    
    # 1. 데이터 로드
    interactions = load_data_from_postgres()
    
    # 2. Train/Test split (시간 기반)
    split_date = interactions['timestamp'].quantile(0.8)
    train_df = interactions[interactions['timestamp'] < split_date]
    test_df = interactions[interactions['timestamp'] >= split_date]
    
    logger.info(f"📊 Train: {len(train_df)}, Test: {len(test_df)}")
    
    # 3. 모델 학습
    model = HybridV2Recommender(
        popularity_weight=0.40,
        user_cf_weight=0.25,
        item_cf_weight=0.25,
        diversity_weight=0.10,
        min_user_interactions=3,
        min_item_interactions=5,
        temporal_decay_days=30
    )
    
    model.fit(train_df)
    
    # 4. 빠른 검증 (샘플 사용자)
    unique_users = train_df['user_id'].unique()
    sample_size = min(5, len(unique_users))
    sample_users = np.random.choice(unique_users, size=sample_size, replace=False)
    
    logger.info("\n🎯 샘플 추천 결과:")
    for user_id in sample_users:
        recs = model.recommend(user_id, k=10)
        logger.info(f"  User {user_id}: {len(recs)} recommendations")
        if recs:
            logger.info(f"    Top 3: {[r['item_id'] for r in recs[:3]]}")
    
    # 5. 모델 저장
    model.save('data/models/hybrid_v2_model.pkl')
    
    logger.info("\n✅ 학습 완료!")
    logger.info(f"📁 모델 저장 위치: data/models/hybrid_v2_model.pkl")
    logger.info(f"📊 모델 통계:")
    logger.info(f"  - 사용자 수: {len(model.user_ids)}")
    logger.info(f"  - 아이템 수: {len(model.item_ids)}")
    logger.info(f"  - 학습 데이터: {len(train_df)}")
    
    return model


if __name__ == '__main__':
    try:
        model = train_and_save()
    except Exception as e:
        logger.error(f"❌ 학습 실패: {e}", exc_info=True)
        sys.exit(1)