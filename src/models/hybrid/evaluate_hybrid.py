"""
Hybrid v2 모델 평가 및 진단
- 개인화 정도 측정
- 다양성 분석
- 인기도 영향 분석
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pickle
import numpy as np
import pandas as pd
import logging

# 🔧 FIX: HybridV2Recommender 클래스 import
# pickle.load()가 클래스를 찾을 수 있도록
try:
    from Hybrid_v1 import HybridV2Recommender
except ImportError:
    try:
        # 현재 디렉토리에서 import 시도
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "hybrid_module", 
            Path(__file__).parent / "Hybrid_v1.py"
        )
        hybrid_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hybrid_module)
        HybridV2Recommender = hybrid_module.HybridV2Recommender
    except Exception as e:
        logging.warning(f"⚠️  HybridV2Recommender import 실패: {e}")
        logging.warning("   pickle 로드가 실패할 수 있습니다.")
        HybridV2Recommender = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_model(model_path='data/models/hybrid_v2_model.pkl'):
    """
    모델 평가 및 진단
    """
    logger.info(f"🔍 모델 평가: {model_path}")
    logger.info("="*60)
    
    # 1. 모델 로드
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        logger.error(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return None
    except Exception as e:
        logger.error(f"❌ 모델 로드 실패: {e}")
        return None
    
    # 2. 기본 통계
    logger.info("\n📊 모델 기본 통계:")
    logger.info(f"  사용자 수: {len(model.user_ids)}")
    logger.info(f"  아이템 수: {len(model.item_ids)}")
    logger.info(f"  행렬 밀도: {(1 - model.user_item_matrix.nnz / (model.user_item_matrix.shape[0] * model.user_item_matrix.shape[1]) ):.4f}")
    logger.info(f"  희소도: {model.user_item_matrix.nnz / (model.user_item_matrix.shape[0] * model.user_item_matrix.shape[1]) :.2%}")
    
    # 3. 인기도 분포
    logger.info("\n📊 아이템 인기도 Top 5:")
    for item_id, score in model.item_popularity.nlargest(5).items():
        logger.info(f"  Item {item_id}: {score:.4f} ({score*100:.1f}%)")
    
    top3_ratio = model.item_popularity.nlargest(3).sum()
    logger.info(f"  → Top 3 누적: {top3_ratio:.2%}")
    
    # 4. 유사도 행렬 체크
    logger.info("\n🔗 유사도 행렬:")
    
    user_sim_dense = model.user_similarity.toarray()
    # 대각선 제외 (자기 자신과의 유사도)
    np.fill_diagonal(user_sim_dense, 0)
    user_sim_mean = user_sim_dense[user_sim_dense > 0].mean() if (user_sim_dense > 0).any() else 0
    
    logger.info(f"  User Similarity 평균: {user_sim_mean:.4f}")
    logger.info(f"  User Similarity 최대: {user_sim_dense.max():.4f}")
    
    item_sim_dense = model.item_similarity.toarray()
    np.fill_diagonal(item_sim_dense, 0)
    item_sim_mean = item_sim_dense[item_sim_dense > 0].mean() if (item_sim_dense > 0).any() else 0
    
    logger.info(f"  Item Similarity 평균: {item_sim_mean:.4f}")
    logger.info(f"  Item Similarity 최대: {item_sim_dense.max():.4f}")
    
    # 5. 추천 다양성 테스트
    logger.info("\n🎯 추천 다양성 테스트:")
    
    sample_size = min(20, len(model.user_ids))
    sample_users = np.random.choice(model.user_ids, size=sample_size, replace=False)
    
    all_recs = []
    for i, user_id in enumerate(sample_users):
        recs = model.recommend(user_id, k=5)
        top_items = tuple([r['item_id'] for r in recs])
        all_recs.append(top_items)
        
        # 처음 10개만 출력
        if i < 10:
            reasons_str = ", ".join(recs[0]['reasons']) if recs else "없음"
            logger.info(f"  User {user_id}: {list(top_items)} (이유: {reasons_str})")
    
    # 중복 체크
    unique_recs = len(set(all_recs))
    total_recs = len(all_recs)
    diversity_ratio = unique_recs / total_recs
    
    logger.info(f"\n📊 다양성 분석:")
    logger.info(f"  총 샘플 사용자: {total_recs}")
    logger.info(f"  고유 추천 패턴: {unique_recs}")
    logger.info(f"  다양성 비율: {diversity_ratio:.2%}")
    
    # 6. 평가 결과
    logger.info("\n"+"="*60)
    logger.info("📝 평가 결과:")
    logger.info("="*60)
    
    if diversity_ratio > 0.5:
        logger.info("✅ 개인화 성공!")
        logger.info("   → 사용자별로 다른 추천 제공")
        logger.info("   → 대시보드 제작 가능")
    elif diversity_ratio > 0.3:
        logger.info("⚠️  개인화 부분적 성공")
        logger.info("   → 일부 개인화 작동")
        logger.info("   → 가중치 재조정 권장")
    else:
        logger.info("❌ 개인화 실패")
        logger.info("   → 대부분 동일한 추천")
        logger.info("   → 재학습 필수 (Hybrid_v2_rebalanced.py)")
    
    # 7. 권장사항
    logger.info("\n💡 권장사항:")
    
    if top3_ratio > 0.7:
        logger.info("  - Top 3 아이템이 70% 이상 차지")
        logger.info("  - Popularity 가중치 낮추기 권장")
        logger.info("  - python src/models/hybrid/Hybrid_v2_rebalanced.py")
    
    if user_sim_mean < 0.01:
        logger.info("  - User 유사도가 매우 낮음")
        logger.info("  - User-CF 효과 제한적")
    
    if item_sim_mean < 0.01:
        logger.info("  - Item 유사도가 매우 낮음")
        logger.info("  - Item-CF 효과 제한적")
    
    logger.info("\n"+"="*60)
    
    return {
        'diversity_ratio': diversity_ratio,
        'top3_ratio': float(top3_ratio),
        'user_sim_mean': float(user_sim_mean),
        'item_sim_mean': float(item_sim_mean),
        'recommendation': 'rebalanced' if diversity_ratio < 0.3 else 'ok'
    }


def compare_models():
    """두 모델 비교 (기본 vs 재조정)"""
    
    logger.info("\n"+"="*60)
    logger.info("🔄 모델 비교: 기본 vs 재조정")
    logger.info("="*60 + "\n")
    
    # 기본 모델
    logger.info("1️⃣ 기본 모델 (Hybrid_v1)")
    logger.info("-"*60)
    result1 = evaluate_model('data/models/hybrid_v2_model.pkl')
    
    if result1 is None:
        logger.error("❌ 기본 모델을 찾을 수 없습니다")
        logger.info("   → python src/models/hybrid/Hybrid_v1.py 실행 필요")
    
    # 재조정 모델
    logger.info("\n2️⃣ 재조정 모델 (Hybrid_v2_rebalanced)")
    logger.info("-"*60)
    result2 = evaluate_model('data/models/hybrid_v2_rebalanced.pkl')
    
    if result2 is None:
        logger.warning("⚠️  재조정 모델을 찾을 수 없습니다")
        logger.info("   → python src/models/hybrid/Hybrid_v2_rebalanced.py 실행 필요")
    
    # 비교
    if result1 and result2:
        logger.info("\n"+"="*60)
        logger.info("📊 비교 결과")
        logger.info("="*60)
        
        logger.info(f"\n다양성 비율:")
        logger.info(f"  기본:    {result1['diversity_ratio']:.2%}")
        logger.info(f"  재조정:  {result2['diversity_ratio']:.2%}")
        logger.info(f"  개선:    {(result2['diversity_ratio'] - result1['diversity_ratio'])*100:+.1f}%p")
        
        logger.info(f"\nTop 3 인기도:")
        logger.info(f"  기본:    {result1['top3_ratio']:.2%}")
        logger.info(f"  재조정:  {result2['top3_ratio']:.2%}")
        
        logger.info(f"\n💡 최종 권장:")
        if result2['diversity_ratio'] > result1['diversity_ratio'] * 1.2:
            logger.info("  ✅ 재조정 모델 사용 권장 (20% 이상 개선)")
        elif result2['diversity_ratio'] > result1['diversity_ratio']:
            logger.info("  ⚠️  재조정 모델 약간 개선")
        else:
            logger.info("  ❌ 기본 모델 유지")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Hybrid v2 모델 평가')
    parser.add_argument('--model', type=str, default='data/models/hybrid_v2_model.pkl',
                       help='평가할 모델 경로')
    parser.add_argument('--compare', action='store_true',
                       help='두 모델 비교')
    
    args = parser.parse_args()
    
    try:
        if args.compare:
            compare_models()
        else:
            result = evaluate_model(args.model)
            if result is None:
                sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 평가 실패: {e}", exc_info=True)
        sys.exit(1)