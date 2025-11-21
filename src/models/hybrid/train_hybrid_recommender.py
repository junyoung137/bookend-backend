"""
Hybrid Recommender 학습 스크립트
기존 hybrid_recommender.py 시스템을 위한 학습 및 평가
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import logging
import pickle
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from config.database import get_db
from config.logging_config import setup_logging
from src.models.hybrid.hybrid_recommender import HybridRecommender
from src.database.models import Interaction, User, Item

# 로깅 설정
setup_logging()
logger = logging.getLogger(__name__)


def load_interaction_data(session, lookback_days=180):
    """
    학습용 상호작용 데이터 로드
    
    Args:
        session: Database session
        lookback_days: 최근 몇 일간의 데이터를 사용할지
    
    Returns:
        DataFrame with columns: user_id, item_id, timestamp
    """
    logger.info(f"📊 최근 {lookback_days}일 데이터 로드 중...")
    
    cutoff_date = datetime.now() - timedelta(days=lookback_days)
    
    query = session.query(
        Interaction.user_id,
        Interaction.item_id,
        Interaction.event_time
    ).filter(
        Interaction.item_id.isnot(None),
        Interaction.event_time >= cutoff_date
    ).order_by(Interaction.event_time)
    
    df = pd.read_sql(query.statement, session.bind)
    df.rename(columns={'event_time': 'timestamp'}, inplace=True)
    
    logger.info(f"✅ 데이터 로드 완료:")
    logger.info(f"   - 총 상호작용: {len(df):,}")
    logger.info(f"   - 고유 사용자: {df['user_id'].nunique():,}")
    logger.info(f"   - 고유 아이템: {df['item_id'].nunique():,}")
    logger.info(f"   - 기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
    
    return df


def analyze_data_distribution(df):
    """데이터 분포 분석"""
    logger.info("\n📈 데이터 분포 분석:")
    
    # 사용자별 상호작용 수
    user_counts = df['user_id'].value_counts()
    logger.info(f"   사용자별 상호작용 수:")
    logger.info(f"     - 평균: {user_counts.mean():.2f}")
    logger.info(f"     - 중앙값: {user_counts.median():.2f}")
    logger.info(f"     - 최소: {user_counts.min()}")
    logger.info(f"     - 최대: {user_counts.max()}")
    
    # 아이템별 상호작용 수
    item_counts = df['item_id'].value_counts()
    logger.info(f"   아이템별 상호작용 수:")
    logger.info(f"     - 평균: {item_counts.mean():.2f}")
    logger.info(f"     - 중앙값: {item_counts.median():.2f}")
    logger.info(f"     - 최소: {item_counts.min()}")
    logger.info(f"     - 최대: {item_counts.max()}")
    
    # 상위 10개 아이템
    logger.info(f"\n   🔥 Top 10 인기 아이템:")
    top_items = item_counts.head(10)
    for idx, (item_id, count) in enumerate(top_items.items(), 1):
        logger.info(f"      {idx}. Item {item_id}: {count}회 ({count/len(df)*100:.2f}%)")


def train_hybrid_model(session, config=None):
    """
    Hybrid Recommender 학습
    
    Args:
        session: Database session
        config: 모델 설정 (None이면 기본값 사용)
    
    Returns:
        Trained HybridRecommender instance
    """
    logger.info("\n" + "="*60)
    logger.info("🚀 Hybrid Recommender 학습 시작")
    logger.info("="*60)
    
    # 기본 설정
    if config is None:
        config = {
            # 가중치 (합이 1.0이 되도록)
            "user_cf_weight": 0.35,      # User-based CF
            "item_cf_weight": 0.35,      # Item-based CF
            "context_weight": 0.20,      # 컨텍스트 특성
            "recency_weight": 0.10,      # 최신 인기도
            
            # MMR 다양성 설정
            "enable_mmr": True,
            "mmr_lambda": 0.5,           # 0.0=다양성 최대, 1.0=관련성 최대
            
            # 캐시 설정
            "recency_cache_ttl": 3600,   # 1시간
        }
    
    logger.info("\n⚙️  모델 설정:")
    for key, value in config.items():
        logger.info(f"   {key}: {value}")
    
    # 모델 초기화
    logger.info("\n🔧 모델 초기화 중...")
    recommender = HybridRecommender(session, config)
    
    # 학습
    logger.info("\n📚 모델 학습 중...")
    try:
        recommender.fit(
            weighting="count",           # 가중치 방식: count, tfidf, bm25
            min_interactions=3,          # 최소 상호작용 수
            lookback_days=90             # 최근 90일 데이터 사용
        )
        logger.info("✅ 모델 학습 완료!")
        
    except Exception as e:
        logger.error(f"❌ 모델 학습 실패: {e}", exc_info=True)
        raise
    
    return recommender


def evaluate_recommendations(recommender, session, num_users=10):
    """
    샘플 사용자들에 대한 추천 결과 평가
    
    Args:
        recommender: 학습된 HybridRecommender
        session: Database session
        num_users: 평가할 사용자 수
    """
    logger.info("\n" + "="*60)
    logger.info("🎯 추천 결과 샘플링")
    logger.info("="*60)
    
    try:
        # 활성 사용자 샘플링 (is_active 없으므로 모든 사용자에서)
        all_users = session.query(User.id).limit(num_users * 5).all()
        
        if not all_users:
            logger.warning("⚠️  사용자가 없습니다.")
            return
        
        user_ids = [u.id for u in all_users]
        sample_user_ids = np.random.choice(user_ids, size=min(num_users, len(user_ids)), replace=False)
        
        # 컨텍스트 예시
        contexts = [
            {
                'timestamp': datetime.now(),
                'browser': 'Chrome',
                'os': 'Windows',
                'time_of_day': 'morning'
            },
            {
                'timestamp': datetime.now(),
                'browser': 'Safari',
                'os': 'iOS',
                'device_id': 'mobile',
                'time_of_day': 'evening'
            },
            {
                'timestamp': datetime.now(),
                'browser': 'Firefox',
                'os': 'Linux',
                'time_of_day': 'afternoon'
            }
        ]
        
        success_count = 0
        total_recs = 0
        
        for idx, user_id in enumerate(sample_user_ids, 1):
            try:
                context = contexts[idx % len(contexts)]
                
                logger.info(f"\n👤 User {user_id} (샘플 {idx}/{len(sample_user_ids)}):")
                logger.info(f"   컨텍스트: {context.get('time_of_day')}, {context.get('os')}")
                
                # 추천 생성
                recommendations = recommender.recommend(
                    user_id=user_id,
                    context=context,
                    limit=5,
                    min_score=0.0
                )
                
                if recommendations:
                    success_count += 1
                    total_recs += len(recommendations)
                    
                    logger.info(f"   ✅ {len(recommendations)}개 추천 생성:")
                    for rec in recommendations:
                        logger.info(f"      {rec.rank}. {rec.item_name or f'Item {rec.item_id}'}")
                        logger.info(f"         Score: {rec.score:.4f} | {rec.reason}")
                else:
                    logger.warning(f"   ⚠️  추천 생성 실패")
                    
            except Exception as e:
                logger.error(f"   ❌ 추천 실패: {e}")
        
        # 통계 요약
        logger.info(f"\n📊 추천 통계:")
        logger.info(f"   성공률: {success_count}/{len(sample_user_ids)} ({success_count/len(sample_user_ids)*100:.1f}%)")
        if success_count > 0:
            logger.info(f"   평균 추천 수: {total_recs/success_count:.1f}")
        
    except Exception as e:
        logger.error(f"❌ 평가 실패: {e}", exc_info=True)


def save_model_components(recommender, filepath="data/models/hybrid_recommender.pkl"):
    """
    모델의 필수 컴포넌트만 저장 (Session 제외)
    
    Args:
        recommender: HybridRecommender 인스턴스
        filepath: 저장 경로
    """
    logger.info(f"\n💾 모델 컴포넌트 저장 중: {filepath}")
    
    try:
        # 디렉토리 생성
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 저장할 컴포넌트 추출
        model_data = {
            'config': recommender.config,
            'weights': {
                'user_cf_weight': recommender.user_cf_weight,
                'item_cf_weight': recommender.item_cf_weight,
                'context_weight': recommender.context_weight,
                'recency_weight': recommender.recency_weight,
            },
            'user_cf': {
                'matrix_builder': recommender.user_cf.matrix_builder,
                'user_similarity': recommender.user_cf.user_similarity,
            },
            'item_cf': {
                'matrix_builder': recommender.item_cf.matrix_builder,
                'item_similarity': recommender.item_cf.item_similarity,
            },
            'recency_cache': {
                'scores': recommender._recency_scores_cache,
                'cache_time': recommender._recency_cache_time,
            },
            'mmr_settings': {
                'enable_mmr': recommender.enable_mmr,
                'mmr_lambda': recommender.mmr_lambda,
            },
            'is_fitted': recommender.is_fitted,
        }
        
        # 저장
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # 파일 크기 확인
        size_mb = save_path.stat().st_size / (1024 * 1024)
        logger.info(f"✅ 모델 컴포넌트 저장 완료: {size_mb:.2f} MB")
        
        logger.info("\n📦 저장된 컴포넌트:")
        logger.info("   • Config 설정")
        logger.info("   • User-CF 행렬 및 유사도")
        logger.info("   • Item-CF 행렬 및 유사도")
        logger.info("   • Recency 캐시")
        logger.info("   • MMR 설정")
        
    except Exception as e:
        logger.error(f"❌ 모델 저장 실패: {e}", exc_info=True)
        raise


def main():
    """메인 실행 함수"""
    
    logger.info("\n" + "="*80)
    logger.info("🚀 HYBRID RECOMMENDER 학습 파이프라인")
    logger.info("="*80)
    
    db = get_db()
    
    try:
        with db.session_scope() as session:
            
            # 1. 데이터 로드 및 분석
            logger.info("\n📊 Step 1: 데이터 로드 및 분석")
            df = load_interaction_data(session, lookback_days=180)
            
            if len(df) == 0:
                logger.error("❌ 상호작용 데이터가 없습니다.")
                return
            
            analyze_data_distribution(df)
            
            # 2. 모델 설정
            logger.info("\n⚙️  Step 2: 모델 설정")
            config = {
                # 가중치 재조정 (개인화 강화)
                "user_cf_weight": 0.35,      # User-CF 강화
                "item_cf_weight": 0.35,      # Item-CF 강화
                "context_weight": 0.20,      # Context 유지
                "recency_weight": 0.10,      # Recency 낮춤
                
                # MMR 다양성
                "enable_mmr": True,
                "mmr_lambda": 0.5,           # 관련성과 다양성 균형
                
                # 캐시
                "recency_cache_ttl": 3600,
            }
            
            logger.info("\n   가중치 분배:")
            logger.info(f"   • User-CF:  {config['user_cf_weight']:.0%}")
            logger.info(f"   • Item-CF:  {config['item_cf_weight']:.0%}")
            logger.info(f"   • Context:  {config['context_weight']:.0%}")
            logger.info(f"   • Recency:  {config['recency_weight']:.0%}")
            
            # 3. 모델 학습
            logger.info("\n📚 Step 3: 모델 학습")
            recommender = train_hybrid_model(session, config)
            
            # 4. 추천 결과 샘플링 및 평가
            logger.info("\n🎯 Step 4: 추천 결과 평가")
            evaluate_recommendations(recommender, session, num_users=10)
            
            # 5. 모델 저장
            logger.info("\n💾 Step 5: 모델 컴포넌트 저장")
            save_model_components(recommender, "data/models/hybrid_recommender.pkl")
            
            # 최종 요약
            logger.info("\n" + "="*80)
            logger.info("✅ 학습 파이프라인 완료!")
            logger.info("="*80)
            
            logger.info("\n📁 생성된 파일:")
            logger.info("   • data/models/hybrid_recommender.pkl")
            
            logger.info("\n💡 다음 단계:")
            logger.info("   1. API 서버에서 모델 로드 테스트")
            logger.info("   2. 모델 로더 수정 필요:")
            logger.info("      - pickle.load()로 model_data 로드")
            logger.info("      - HybridRecommender 재생성")
            logger.info("      - 컴포넌트 복원")
            
    except Exception as e:
        logger.error(f"\n❌ 학습 파이프라인 실패: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n⚠️  사용자에 의해 중단됨")
    except Exception as e:
        logger.error(f"\n❌ 실행 실패: {e}")
        sys.exit(1)