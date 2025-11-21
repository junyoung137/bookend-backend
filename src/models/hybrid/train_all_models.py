"""
전체 모델 학습 스크립트
기본 모델 + 재조정 모델을 순차적으로 학습

사용법:
    python src/models/hybrid/train_all_models.py
"""

import sys
import subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_script(script_path: str, description: str) -> bool:
    """스크립트를 subprocess로 실행"""
    logger.info(f"\n{'='*60}")
    logger.info(f"🚀 {description}")
    logger.info(f"{'='*60}\n")
    
    try:
        # Python 인터프리터 경로
        python_exe = sys.executable
        
        result = subprocess.run(
            [python_exe, script_path],
            check=True,
            capture_output=False,  # 실시간 출력
            text=True
        )
        
        logger.info(f"✅ {description} 완료!\n")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} 실패: {e}\n")
        return False
    except Exception as e:
        logger.error(f"❌ {description} 오류: {e}\n", exc_info=True)
        return False


def main():
    """전체 모델 학습"""
    
    logger.info("="*60)
    logger.info("🚀 하이브리드 추천 모델 학습 시작")
    logger.info("="*60)
    
    # 프로젝트 루트 경로
    project_root = Path(__file__).parent.parent.parent.parent
    
    # 1. 기본 모델 학습 (Popularity 40%)
    hybrid_v1_path = project_root / "src/models/hybrid/Hybrid_v1.py"
    
    if not hybrid_v1_path.exists():
        logger.error(f"❌ Hybrid_v1.py를 찾을 수 없습니다: {hybrid_v1_path}")
        return False
    
    success = run_script(
        str(hybrid_v1_path),
        "1️⃣  기본 모델 학습 (Popularity 40%)"
    )
    
    if not success:
        logger.error("기본 모델 학습 실패로 중단합니다.")
        return False
    
    # 2. 재조정 모델 학습 (Popularity 20%)
    retrain_path = project_root / "src/models/hybrid/Retrain_rebalanced.py"
    
    if not retrain_path.exists():
        logger.error(f"❌ Retrain_rebalanced.py를 찾을 수 없습니다: {retrain_path}")
        logger.info("\n💡 다음 파일을 생성하세요:")
        logger.info(f"   {retrain_path}")
        return False
    
    success = run_script(
        str(retrain_path),
        "2️⃣  재조정 모델 학습 (Popularity 20%)"
    )
    
    if not success:
        logger.error("재조정 모델 학습 실패로 중단합니다.")
        return False
    
    # 3. 결과 확인
    logger.info("\n" + "="*60)
    logger.info("📊 학습 결과")
    logger.info("="*60)
    
    models_dir = project_root / 'data/models'
    
    base_model = models_dir / 'hybrid_v2_model.pkl'
    rebalanced_model = models_dir / 'hybrid_v2_rebalanced.pkl'
    
    all_exist = True
    
    if base_model.exists():
        size_mb = base_model.stat().st_size / (1024 * 1024)
        logger.info(f"✅ 기본 모델: {base_model.name} ({size_mb:.2f} MB)")
    else:
        logger.error(f"❌ 기본 모델 없음: {base_model}")
        all_exist = False
    
    if rebalanced_model.exists():
        size_mb = rebalanced_model.stat().st_size / (1024 * 1024)
        logger.info(f"✅ 재조정 모델: {rebalanced_model.name} ({size_mb:.2f} MB)")
    else:
        logger.error(f"❌ 재조정 모델 없음: {rebalanced_model}")
        all_exist = False
    
    if all_exist:
        logger.info("\n" + "="*60)
        logger.info("🎉 모든 모델 학습 완료!")
        logger.info("="*60)
        
        logger.info("\n💡 다음 단계:")
        logger.info("   python src/models/hybrid/adaptive_hybrid.py")
    else:
        logger.error("\n⚠️  일부 모델 생성 실패")
    
    return all_exist


if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"❌ 학습 실패: {e}", exc_info=True)
        sys.exit(1)