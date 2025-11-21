import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle
import logging
from typing import Optional, Dict, List, Any
import pytz

# ✅ DB 연결 함수 임포트
from src.models.hybrid.Hybrid_v1 import load_data_from_postgres

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ 데이터 수집 기준 날짜 (2025년 10월 22일)
REFERENCE_DATE = datetime(2025, 10, 22, tzinfo=pytz.UTC)


class UserSegment:
    """사용자 세그먼트 정의"""
    
    NEW_USER = "신규사용자"              # 신규 (0-2 interactions)
    BEGINNER = "초보사용자"              # 초보 (3-9 interactions)
    REGULAR = "일반사용자"               # 활성 (10-29 interactions)
    HEAVY_USER = "헤비사용자"            # 파워 (30+ interactions)
    INACTIVE = "휴면사용자"              # 휴면 (30일 이상 미사용)


class AdaptiveHybridRecommender:
    """
    적응형 하이브리드 추천 시스템
    
    사용자 행동 패턴 분석:
    1. 상호작용 횟수 (Interaction Count)
    2. 최근 활동성 (Recency)
    3. 다양성 (Diversity)
    4. 활동 기간 (Tenure)
    
    → 세그먼트별로 다른 추천 전략 적용
    """
    
    def __init__(self, base_model, rebalanced_model=None):
        """
        Args:
            base_model: 기본 모델 객체 (Popularity 40%)
            rebalanced_model: 재조정 모델 객체 (Popularity 20%)
        """
        self.base_model = base_model
        self.rebalanced_model = rebalanced_model
        
        # 사용자 프로파일 캐시
        self.user_profiles: Dict[int, Dict] = {}
        
        # ✅ 전체 데이터 한 번만 로드 (성능 최적화)
        self._all_interactions = None
        
        logger.info("🚀 적응형 하이브리드 추천 시스템 초기화")
    
    def _load_all_interactions(self) -> pd.DataFrame:
        """전체 상호작용 데이터 로드 (캐싱)"""
        if self._all_interactions is None:
            logger.info("📂 전체 상호작용 데이터 로딩 중...")
            self._all_interactions = load_data_from_postgres()
            logger.info(f"   ✅ {len(self._all_interactions)}개 상호작용 로드 완료")
        return self._all_interactions
    
    def analyze_user_behavior(
        self, 
        user_id: int, 
        interactions_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        사용자 행동 패턴 분석
        
        Args:
            user_id: 사용자 ID
            interactions_df: 전체 상호작용 데이터 (선택)
        
        Returns:
            dict: {
                'segment': UserSegment,
                'interaction_count': int,
                'last_interaction_days': int,
                'diversity_score': float,
                'tenure_days': int
            }
        """
        # 캐시 확인
        if user_id in self.user_profiles:
            return self.user_profiles[user_id]
        
        # ✅ 데이터가 제공되지 않으면 캐시된 전체 데이터 사용
        if interactions_df is None:
            interactions_df = self._load_all_interactions()
        
        # 사용자 데이터 필터링
        user_data = interactions_df[interactions_df['user_id'] == user_id]
        
        if len(user_data) == 0:
            profile = {
                'segment': UserSegment.NEW_USER,
                'interaction_count': 0,
                'last_interaction_days': 999,
                'diversity_score': 0.0,
                'tenure_days': 0
            }
        else:
            # 1. 상호작용 횟수
            interaction_count = len(user_data)
            
            # 2. 최근 활동성 (마지막 상호작용 이후 일수)
            last_interaction = pd.to_datetime(user_data['timestamp'].max())
            
            # ✅ 기준 날짜를 데이터 종료일(2025-10-22)로 설정
            if last_interaction.tzinfo is None:
                last_interaction = last_interaction.tz_localize('UTC')
            
            # 데이터 수집 종료일 기준으로 계산
            last_interaction_days = (REFERENCE_DATE - last_interaction).days
            
            # 3. 다양성 (고유 아이템 수 / 전체 상호작용)
            unique_items = user_data['item_id'].nunique()
            diversity_score = unique_items / interaction_count if interaction_count > 0 else 0
            
            # 4. 활동 기간 (첫 상호작용부터 기준 날짜까지)
            first_interaction = pd.to_datetime(user_data['timestamp'].min())
            if first_interaction.tzinfo is None:
                first_interaction = first_interaction.tz_localize('UTC')
            tenure_days = (REFERENCE_DATE - first_interaction).days
            
            # 세그먼트 분류
            segment = self._classify_segment(
                interaction_count,
                last_interaction_days,
                diversity_score
            )
            
            profile = {
                'segment': segment,
                'interaction_count': interaction_count,
                'last_interaction_days': last_interaction_days,
                'diversity_score': diversity_score,
                'tenure_days': tenure_days
            }
        
        # 캐시 저장
        self.user_profiles[user_id] = profile
        return profile
    
    def _classify_segment(
        self, 
        interaction_count: int, 
        last_interaction_days: int, 
        diversity_score: float
    ) -> str:
        """사용자 세그먼트 분류"""
        
        # 휴면 사용자 (30일 이상 미사용)
        if last_interaction_days > 30:
            return UserSegment.INACTIVE
        
        # 상호작용 횟수 기반
        if interaction_count <= 2:
            return UserSegment.NEW_USER
        elif interaction_count <= 9:
            return UserSegment.BEGINNER
        elif interaction_count <= 29:
            return UserSegment.REGULAR
        else:
            return UserSegment.HEAVY_USER
    
    def get_recommendation_strategy(self, user_profile: Dict) -> Dict[str, Any]:
        """
        세그먼트별 추천 전략 반환
        
        Args:
            user_profile: 사용자 프로파일 딕셔너리
        
        Returns:
            dict: {
                'model': 'base' or 'rebalanced',
                'weights': dict,
                'k': int,
                'explanation': str
            }
        """
        segment = user_profile['segment']
        
        strategies = {
            UserSegment.NEW_USER: {
                'model': 'base',
                'weights': {
                    'popularity': 0.70,
                    'user_cf': 0.10,
                    'item_cf': 0.10,
                    'diversity': 0.10
                },
                'k': 5,
                'explanation': '신규 사용자 - 인기 기능 위주 추천'
            },
            UserSegment.BEGINNER: {
                'model': 'base',
                'weights': {
                    'popularity': 0.50,
                    'user_cf': 0.20,
                    'item_cf': 0.20,
                    'diversity': 0.10
                },
                'k': 8,
                'explanation': '초보 사용자 - 인기 + 개인화 혼합'
            },
            UserSegment.REGULAR: {
                'model': 'rebalanced',
                'weights': {
                    'popularity': 0.20,
                    'user_cf': 0.35,
                    'item_cf': 0.35,
                    'diversity': 0.10
                },
                'k': 10,
                'explanation': '일반 사용자 - 개인화 중심 추천'
            },
            UserSegment.HEAVY_USER: {
                'model': 'rebalanced',
                'weights': {
                    'popularity': 0.10,
                    'user_cf': 0.40,
                    'item_cf': 0.35,
                    'diversity': 0.15
                },
                'k': 12,
                'explanation': '헤비 사용자 - 고급/다양한 기능 추천'
            },
            UserSegment.INACTIVE: {
                'model': 'base',
                'weights': {
                    'popularity': 0.60,
                    'user_cf': 0.15,
                    'item_cf': 0.15,
                    'diversity': 0.10
                },
                'k': 5,
                'explanation': '휴면 사용자 - 재활성화 전략'
            }
        }
        
        return strategies.get(segment, strategies[UserSegment.NEW_USER])
    
    def recommend(
        self, 
        user_id: int, 
        k: Optional[int] = None, 
        interactions_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        적응형 추천 생성
        
        Args:
            user_id: 사용자 ID
            k: 추천 개수 (None이면 전략에 따라 자동 결정)
            interactions_df: 전체 상호작용 데이터 (선택)
        
        Returns:
            dict: {
                'recommendations': list,
                'user_profile': dict,
                'strategy': dict
            }
        """
        # 1. 사용자 행동 분석
        user_profile = self.analyze_user_behavior(user_id, interactions_df)
        
        # 2. 추천 전략 선택
        strategy = self.get_recommendation_strategy(user_profile)
        
        # 3. 적절한 모델 선택
        if strategy['model'] == 'rebalanced' and self.rebalanced_model:
            model = self.rebalanced_model
        else:
            model = self.base_model
        
        # 4. 추천 개수 결정
        recommend_k = k if k is not None else strategy['k']
        
        # 5. 추천 생성
        recommendations = model.recommend(user_id, k=recommend_k)
        
        # 6. 메타데이터 추가
        for rec in recommendations:
            rec['strategy'] = strategy['explanation']
            rec['segment'] = user_profile['segment']
        
        return {
            'recommendations': recommendations,
            'user_profile': user_profile,
            'strategy': strategy
        }
    
    def get_segment_distribution(self, user_ids: Optional[List[int]] = None) -> Dict[str, int]:
        """
        사용자 세그먼트 분포 분석
        
        Args:
            user_ids: 분석할 사용자 ID 리스트 (None이면 샘플링)
        
        Returns:
            dict: {segment: count}
        """
        # ✅ 전체 데이터 로드
        interactions_df = self._load_all_interactions()
        
        if user_ids is None:
            # 샘플링: 상위 100명
            user_ids = interactions_df['user_id'].unique()[:100].tolist()
        
        if not user_ids:
            logger.warning("⚠️  분석할 사용자가 없습니다.")
            return {}
        
        logger.info(f"   📌 {len(user_ids)}명 분석 중...")
        
        # 세그먼트 분류
        segments = {}
        for user_id in user_ids:
            profile = self.analyze_user_behavior(user_id, interactions_df)
            segment = profile['segment']
            segments[segment] = segments.get(segment, 0) + 1
        
        return segments
    
    def clear_cache(self) -> None:
        """사용자 프로파일 캐시 초기화"""
        self.user_profiles.clear()
        self._all_interactions = None
        logger.info("🧹 사용자 프로파일 캐시 초기화 완료")


# ==================== 평가 스크립트 ====================

def load_model_with_fix(model_path: Path) -> Any:
    """Pickle 호환성 문제 해결하여 모델 로드"""
    
    import importlib.util
    
    # Hybrid_v1.py 모듈 동적 로드
    hybrid_path = Path(__file__).parent / "Hybrid_v1.py"
    
    if not hybrid_path.exists():
        logger.warning(f"⚠️  Hybrid_v1.py를 찾을 수 없습니다: {hybrid_path}")
        # 그냥 pickle 로드 시도
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    
    spec = importlib.util.spec_from_file_location("Hybrid_v1", hybrid_path)
    hybrid_module = importlib.util.module_from_spec(spec)
    
    # 🔑 핵심: sys.modules에 등록해서 pickle이 찾을 수 있게 함
    sys.modules['Hybrid_v1'] = hybrid_module
    sys.modules['__main__'] = hybrid_module
    
    spec.loader.exec_module(hybrid_module)
    
    # 이제 pickle 로드
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model


def evaluate_adaptive_system():
    """적응형 시스템 평가"""
    logger.info("="*60)
    logger.info("🔬 적응형 하이브리드 추천 시스템 평가")
    logger.info(f"📅 기준 날짜: {REFERENCE_DATE.strftime('%Y-%m-%d')}")
    logger.info(f"📊 데이터 기간: 2025-06-01 ~ 2025-10-22 (143일)")
    logger.info("="*60 + "\n")
    
    # 1. 모델 로드 (pickle 호환성 문제 해결)
    logger.info("📂 모델 로딩 중...")
    
    base_model_path = Path('data/models/hybrid_v2_model.pkl')
    rebalanced_model_path = Path('data/models/hybrid_v2_rebalanced.pkl')
    
    if not base_model_path.exists():
        logger.error(f"❌ 기본 모델을 찾을 수 없습니다: {base_model_path}")
        logger.info("\n💡 먼저 모델을 학습하세요:")
        logger.info("   python src/models/hybrid/Hybrid_v1.py")
        logger.info("   python src/models/hybrid/Retrain_rebalanced.py")
        return
    
    try:
        base_model = load_model_with_fix(base_model_path)
        logger.info("   ✅ 기본 모델 로드 완료")
    except Exception as e:
        logger.error(f"   ❌ 기본 모델 로드 실패: {e}")
        return
    
    # 재조정 모델 로드 (있으면)
    rebalanced_model = None
    if rebalanced_model_path.exists():
        try:
            rebalanced_model = load_model_with_fix(rebalanced_model_path)
            logger.info("   ✅ 재조정 모델 로드 완료")
        except Exception as e:
            logger.warning(f"   ⚠️  재조정 모델 로드 실패: {e}")
    else:
        logger.warning("   ⚠️  재조정 모델 없음 - 기본 모델만 사용")
    
    # 2. 적응형 시스템 초기화
    recommender = AdaptiveHybridRecommender(
        base_model=base_model,
        rebalanced_model=rebalanced_model
    )
    
    # 3. 세그먼트 분포 분석
    logger.info("\n📊 사용자 세그먼트 분포 분석 중...")
    
    segment_dist = recommender.get_segment_distribution()
    
    total_users = sum(segment_dist.values())
    logger.info(f"\n📊 세그먼트 분포 (샘플 {total_users}명):")
    for segment, count in sorted(segment_dist.items(), key=lambda x: -x[1]):
        percentage = count / total_users * 100
        bar = "█" * int(percentage / 2)
        logger.info(f"   {segment:15s}: {count:3d}명 ({percentage:5.1f}%) {bar}")
    
    # 4. 세그먼트별 추천 예시
    logger.info("\n🎯 세그먼트별 추천 예시:")
    logger.info("-" * 60)
    
    # 각 세그먼트에서 1명씩 샘플링
    segment_samples = {}
    for user_id, profile in recommender.user_profiles.items():
        segment = profile['segment']
        if segment not in segment_samples:
            segment_samples[segment] = user_id
    
    for segment, user_id in segment_samples.items():
        try:
            result = recommender.recommend(user_id, k=5)
            
            profile = result['user_profile']
            strategy = result['strategy']
            recs = result['recommendations']
            
            logger.info(f"\n📌 {segment.upper()}")
            logger.info(f"   사용자 ID: {user_id}")
            logger.info(f"   상호작용: {profile['interaction_count']}회")
            logger.info(f"   최근 활동: {profile['last_interaction_days']}일 전")
            logger.info(f"   다양성: {profile['diversity_score']:.2f}")
            logger.info(f"   전략: {strategy['explanation']}")
            logger.info(f"   모델: {strategy['model']}")
            logger.info(f"   추천: {[r['item_id'] for r in recs[:5]]}")
        except Exception as e:
            logger.error(f"   ❌ {segment} 추천 실패: {e}")
    
    logger.info("\n" + "="*60)
    logger.info("✅ 평가 완료!")
    logger.info("="*60)


if __name__ == '__main__':
    try:
        evaluate_adaptive_system()
    except Exception as e:
        logger.error(f"❌ 평가 실패: {e}", exc_info=True)
        sys.exit(1)