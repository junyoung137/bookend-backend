"""
재조정된 가중치로 하이브리드 모델 재학습
Popularity 가중치를 낮추고 개인화를 강화

사용법:
    python src/models/hybrid/Retrain_rebalanced.py
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import logging

# Hybrid_v1의 HybridV2Recommender 임포트
from src.models.hybrid.Hybrid_v1 import HybridV2Recommender, load_data_from_postgres

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_rebalanced():
    """재조정된 가중치로 학습"""
    
    logger.info("🚀 Hybrid v2 재학습 시작 (재조정 버전)")
    
    # 1. 데이터 로드
    interactions = load_data_from_postgres()
    
    # 2. Train/Test split
    split_date = interactions['timestamp'].quantile(0.8)
    train_df = interactions[interactions['timestamp'] < split_date]
    test_df = interactions[interactions['timestamp'] >= split_date]
    
    logger.info(f"📊 Train: {len(train_df)}, Test: {len(test_df)}")
    
    # 3. 가중치 재조정
    logger.info("\n⚖️  가중치 재조정:")
    logger.info("  Popularity:  40% → 20% (낮춤)")
    logger.info("  User-CF:     25% → 35% (강화)")
    logger.info("  Item-CF:     25% → 35% (강화)")
    logger.info("  Diversity:   10% → 10% (유지)\n")
    
    model = HybridV2Recommender(
        popularity_weight=0.20,  # 40% → 20%
        user_cf_weight=0.35,     # 25% → 35%
        item_cf_weight=0.35,     # 25% → 35%
        diversity_weight=0.10,
        min_user_interactions=3,
        min_item_interactions=5,
        temporal_decay_days=30
    )
    
    model.fit(train_df)
    
    # 4. 다양성 검증
    logger.info("\n🎯 추천 다양성 검증:")
    
    unique_users = train_df['user_id'].unique()
    sample_size = min(20, len(unique_users))
    sample_users = np.random.choice(unique_users, size=sample_size, replace=False)
    
    all_recs = []
    for user_id in sample_users:
        recs = model.recommend(user_id, k=5)
        top_items = tuple([r['item_id'] for r in recs])
        all_recs.append(top_items)
        
        if len(all_recs) <= 10:  # 처음 10개만 로그
            logger.info(f"  User {user_id}: {list(top_items)}")
    
    # 중복 체크
    unique_recs = len(set(all_recs))
    total_recs = len(all_recs)
    diversity_ratio = unique_recs / total_recs
    
    logger.info(f"\n📊 다양성 분석:")
    logger.info(f"  총 사용자: {total_recs}")
    logger.info(f"  고유 추천: {unique_recs}")
    logger.info(f"  다양성 비율: {diversity_ratio:.2%}")
    
    if diversity_ratio > 0.5:
        logger.info("  ✅ 개인화 성공!")
    elif diversity_ratio > 0.3:
        logger.info("  ⚠️  개인화 부분적 성공")
    else:
        logger.info("  ❌ 개인화 실패 (추가 조정 필요)")
    
    # 5. 인기도 영향 분석
    logger.info("\n📊 인기도 영향 분석:")
    logger.info(f"  Top 1 아이템 비중: {model.item_popularity.iloc[0]:.2%}")
    logger.info(f"  Top 3 아이템 누적: {model.item_popularity.nlargest(3).sum():.2%}")
    
    # 6. 모델 저장 경로 확인 및 생성
    save_path = Path('data/models/hybrid_v2_rebalanced.pkl')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 7. 모델 저장
    model.save(str(save_path))
    
    logger.info("\n✅ 재학습 완료!")
    logger.info(f"📁 모델 저장: {save_path}")
    logger.info(f"📊 모델 통계:")
    logger.info(f"  - 사용자 수: {len(model.user_ids)}")
    logger.info(f"  - 아이템 수: {len(model.item_ids)}")
    logger.info(f"  - 학습 데이터: {len(train_df)}")
    
    return model, train_df, test_df


if __name__ == '__main__':
    try:
        model, train_df, test_df = train_rebalanced()
    except Exception as e:
        logger.error(f"❌ 학습 실패: {e}", exc_info=True)
        sys.exit(1)