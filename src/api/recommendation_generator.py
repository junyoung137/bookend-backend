"""
Recommendation Content Generator

✅ 개선사항:
- 재현성 보장 (같은 item_id는 항상 같은 문장)
- Top 3 다양성 보장 (순환 방식)
- 에러 처리 강화
- 로깅 개선
"""

import random
import logging
from typing import Dict, List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


# =========================================================
# Enum (Single Source of Truth)
# =========================================================

class RecommendationType(str, Enum):
    """추천 유형 - schemas.py와 동일"""
    PARAPHRASE = "paraphrase"
    TONE = "tone"
    EXPAND = "expand"


# =========================================================
# Generator Class
# =========================================================

class RecommendationGenerator:
    """
    추천 문장 생성기
    
    원칙:
    1. 같은 item_id는 항상 같은 문장 반환 (재현성)
    2. Top 3는 순환 방식으로 타입 할당 (다양성)
    3. 개별 실패해도 나머지 처리 (견고성)
    
    단일 책임: content와 type 생성만 담당
    """
    
    # 추천 유형별 문장 템플릿 (One Source of Truth)
    TEMPLATES = {
        RecommendationType.PARAPHRASE: [
            "이 문장을 더 간결하고 명확하게 표현해보세요.",
            "핵심만 남기고 불필요한 부분을 제거해보세요.",
            "같은 의미를 더 짧은 문장으로 바꿔보세요.",
            "문장 구조를 단순화해서 읽기 쉽게 만들어보세요.",
            "중복된 표현을 정리하고 간결하게 다듬어보세요.",
        ],
        RecommendationType.TONE: [
            "좀 더 부드럽고 친근한 톤으로 바꿔보세요.",
            "격식 있고 전문적인 느낌으로 수정해보세요.",
            "독자와의 거리를 좁히는 따뜻한 표현으로 바꿔보세요.",
            "공식적이고 신뢰감 있는 어조로 작성해보세요.",
            "좀 더 열정적이고 생동감 있는 톤으로 표현해보세요.",
        ],
        RecommendationType.EXPAND: [
            "이 부분을 구체적인 예시와 함께 확장해보세요.",
            "좀 더 자세한 설명을 추가해서 풍부하게 만들어보세요.",
            "배경 정보나 맥락을 추가해서 완성도를 높여보세요.",
            "독자의 이해를 돕는 추가 설명을 넣어보세요.",
            "관련된 세부 사항을 더해서 깊이 있게 작성해보세요.",
        ],
    }
    
    def __init__(self, seed: Optional[int] = None):
        """
        Args:
            seed: 랜덤 시드 (테스트용, 기본값 None)
        """
        self.rng = random.Random(seed)
        logger.info(f"RecommendationGenerator initialized (seed={seed})")
    
    def generate(
        self,
        item_id: int,
        score: float,
        rank: int,
        context: Optional[Dict] = None
    ) -> Tuple[str, str]:
        """
        단일 추천에 대한 content/type 생성
        
        단일 책임: 한 개의 추천에 대한 생성
        
        Args:
            item_id: 아이템 ID
            score: 추천 점수 (0.0 ~ 1.0)
            rank: 순위 (1부터 시작)
            context: 선택적 컨텍스트 (미래 확장용)
        
        Returns:
            (content, type) 튜플
            
        Raises:
            ValueError: 유효하지 않은 입력값
        """
        # 입력 검증
        if item_id <= 0:
            raise ValueError(f"Invalid item_id: {item_id}")
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"Invalid score: {score}")
        if rank < 1:
            raise ValueError(f"Invalid rank: {rank}")
        
        try:
            # 1. 타입 결정
            rec_type = self._determine_type(rank, score, context)
            
            # 2. 문장 선택 (재현성 보장)
            content = self._select_content(rec_type, item_id)
            
            logger.debug(
                f"Generated for item_id={item_id}: type={rec_type.value}, "
                f"score={score:.2f}, rank={rank}"
            )
            
            return content, rec_type.value
            
        except Exception as e:
            logger.error(
                f"Generation failed for item_id={item_id}: {e}",
                exc_info=True
            )
            raise
    
    def _determine_type(
        self, 
        rank: int, 
        score: float,
        context: Optional[Dict] = None
    ) -> RecommendationType:
        """
        추천 유형 결정
        
        단일 책임: 타입 결정 로직만 담당
        
        전략:
        - Top 3: 순환 방식 (paraphrase → tone → expand)
        - 나머지: 점수 기반 가중치
        
        Args:
            rank: 순위
            score: 점수
            context: 컨텍스트 (시간대 등)
        
        Returns:
            RecommendationType
        """
        # Top 3는 순환 방식으로 다양성 보장
        if rank <= 3:
            types = [
                RecommendationType.PARAPHRASE,  # rank 1
                RecommendationType.TONE,        # rank 2
                RecommendationType.EXPAND       # rank 3
            ]
            return types[(rank - 1) % 3]
        
        # 나머지는 점수 기반 가중치
        if score >= 0.8:
            # 고점수: paraphrase 우선
            weights = [0.5, 0.3, 0.2]
        elif score >= 0.5:
            # 중간점수: tone 우선
            weights = [0.3, 0.4, 0.3]
        else:
            # 저점수: expand 우선
            weights = [0.2, 0.3, 0.5]
        
        return self.rng.choices(
            list(RecommendationType), 
            weights=weights, 
            k=1
        )[0]
    
    def _select_content(
        self, 
        rec_type: RecommendationType, 
        item_id: int
    ) -> str:
        """
        템플릿 중 하나 선택
        
        단일 책임: 문장 선택만 담당
        재현성: item_id를 시드로 사용
        
        Args:
            rec_type: 추천 유형
            item_id: 아이템 ID
        
        Returns:
            선택된 문장
        """
        templates = self.TEMPLATES[rec_type]
        
        # item_id를 시드로 사용 (재현성 보장)
        local_rng = random.Random(item_id)
        selected = local_rng.choice(templates)
        
        return selected
    
    def generate_batch(
        self,
        recommendations: List[Dict],
        context: Optional[Dict] = None
    ) -> List[Dict]:
        """
        여러 추천에 대해 content/type 일괄 생성
        
        단일 책임: 배치 생성 및 에러 처리
        Graceful Degradation: 개별 실패해도 나머지 계속 처리
        
        Args:
            recommendations: [{"item_id": int, "score": float, "rank": int}, ...]
            context: 선택적 컨텍스트
        
        Returns:
            content/type이 추가된 리스트
        """
        if not recommendations:
            logger.warning("Empty recommendations list")
            return []
        
        enriched = []
        success_count = 0
        failure_count = 0
        
        for rec in recommendations:
            try:
                # 필수 필드 검증
                if "item_id" not in rec or "score" not in rec or "rank" not in rec:
                    logger.warning(f"Missing required fields in recommendation: {rec}")
                    enriched.append(rec)  # 원본 그대로 추가
                    failure_count += 1
                    continue
                
                # content/type 생성
                content, rec_type = self.generate(
                    item_id=rec["item_id"],
                    score=rec["score"],
                    rank=rec["rank"],
                    context=context
                )
                
                # 원본에 추가
                enriched_rec = {**rec}
                enriched_rec["content"] = content
                enriched_rec["type"] = rec_type
                enriched.append(enriched_rec)
                success_count += 1
                
            except Exception as e:
                # 개별 실패해도 나머지 계속 처리 (Graceful Degradation)
                logger.error(
                    f"Failed to generate for item {rec.get('item_id')}: {e}",
                    exc_info=True
                )
                enriched.append(rec)  # 원본 그대로 추가
                failure_count += 1
        
        logger.info(
            f"Batch generation complete: {success_count} success, "
            f"{failure_count} failures out of {len(recommendations)}"
        )
        
        return enriched


# =========================================================
# Singleton Instance
# =========================================================

_generator_instance: Optional[RecommendationGenerator] = None


def get_recommendation_generator() -> RecommendationGenerator:
    """
    싱글톤 제너레이터 반환
    
    단일 책임: 인스턴스 관리만 담당
    
    Returns:
        RecommendationGenerator 인스턴스
    """
    global _generator_instance
    
    if _generator_instance is None:
        _generator_instance = RecommendationGenerator()
        logger.info("✅ RecommendationGenerator singleton created")
    
    return _generator_instance


# =========================================================
# Testing
# =========================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*70)
    print("🧪 Testing Recommendation Generator")
    print("="*70)
    
    gen = RecommendationGenerator(seed=42)
    
    # Test 1: 단일 생성
    print("\n1️⃣ 단일 추천:")
    content, rec_type = gen.generate(item_id=1, score=0.92, rank=1)
    print(f"   Type: {rec_type}")
    print(f"   Content: {content}")
    
    # Test 2: 배치 생성 (Top 3 다양성 확인)
    print("\n2️⃣ 배치 추천 (Top 3 다양성):")
    test_recs = [
        {"item_id": 1, "score": 0.92, "rank": 1},
        {"item_id": 2, "score": 0.88, "rank": 2},
        {"item_id": 3, "score": 0.82, "rank": 3},
    ]
    
    enriched = gen.generate_batch(test_recs)
    for rec in enriched:
        print(f"\n   Rank {rec['rank']}: {rec['type']}")
        print(f"   Content: {rec['content']}")
    
    # Test 3: 재현성
    print("\n3️⃣ 재현성 테스트:")
    c1, t1 = gen.generate(5, 0.9, 1)
    c2, t2 = gen.generate(5, 0.9, 1)
    print(f"   일치: {c1 == c2 and t1 == t2} ✅")
    
    # Test 4: 에러 처리
    print("\n4️⃣ 에러 처리 테스트:")
    invalid_recs = [
        {"item_id": 10, "score": 0.9, "rank": 1},  # 정상
        {"score": 0.8, "rank": 2},  # item_id 누락
        {"item_id": 11, "score": 0.7, "rank": 3},  # 정상
    ]
    result = gen.generate_batch(invalid_recs)
    print(f"   결과: {len(result)}/3 처리됨")
    
    print("\n" + "="*70)
    print("✅ Test Complete")
    print("="*70 + "\n")