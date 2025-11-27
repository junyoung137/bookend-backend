# collector/rule_collector.py
from groq import Groq
import json
import os
import sys
from datetime import datetime
from collections import defaultdict

# ===== 유사도 계산용 라이브러리 추가 =====
try:
    from rapidfuzz import fuzz
    USE_RAPIDFUZZ = True
except ImportError:
    import difflib
    USE_RAPIDFUZZ = False
    print("⚠️  rapidfuzz 미설치")

# ===== feature별 유사도 임계값 =====
FEATURE_SIMILARITY_THRESHOLD = {
    'Paraphrase': 90.0,    # 다듬기: 엄격
    'Tone Adjust': 85.0,   # 톤 조절
    'Expand': 80.0,        # 확장: 유연
    'Compress': 85.0       # 압축
}

class Collector:
    """
    수집가 (Collector)
    
    역할:
    1. insights_queue.json 읽기
    2. 인사이트를 규칙(Rule)으로 변환
    3. 만족/불만족 분리 처리
    4. 중복 규칙 병합 & 충돌 해결
    5. rulebook.json에 저장
    """
    
    def __init__(self, api_key,
                 insights_path=None,
                 initial_rules_path=None,
                 rulebook_path=None):
        
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"
        self.use_rapidfuzz = USE_RAPIDFUZZ
        
        # ===== 경로 자동 설정 =====
        try:
            # 로컬 사용
            current_file = os.path.abspath(__file__)
            collector_dir = os.path.dirname(current_file)
            project_root = os.path.dirname(collector_dir)
            
        except NameError:
            # 주피터 기준
            current_dir = os.getcwd()
            if 'notebooks' in current_dir:
                project_root = os.path.dirname(current_dir)
            else:
                project_root = current_dir
            print(f"📍 주피터 환경 감지: {project_root}")
            
        if insights_path is None:
            insights_path = os.path.join(project_root, 'data', 'insights_queue.json')
    
        if initial_rules_path is None:
            initial_rules_path = os.path.join(project_root, 'data', 'initial_rulebook.json')
    
        if rulebook_path is None:
            rulebook_path = os.path.join(project_root, 'data', 'rulebook.json')
        
        self.insights_path = insights_path
        self.initial_rules_path = initial_rules_path
        self.rulebook_path = rulebook_path
        
        # 룰북 로드 (우선순위 적용)
        self.rulebook = self._load_rulebook()


    def _calculate_similarity(self, text1, text2):
        """
        두 guideline 문자열의 유사도 계산 (0~100)
        
        Args:
            text1: 첫 번째 guideline
            text2: 두 번째 guideline
        
        Returns:
            float: 유사도 (0~100)
        """
        if not text1 or not text2:
            return 0.0
        
        if self.use_rapidfuzz:
            # rapidfuzz 사용 (빠르고 정확)
            return float(fuzz.token_set_ratio(text1, text2))
        else:
            # difflib 사용 (fallback)
            ratio = difflib.SequenceMatcher(None, text1, text2).ratio()
            return ratio * 100.0
        

    def _convert_initial_rules(self, initial_data):
        """
        초기 규칙 (선택 데이터) → 룰북 형식 변환
    
        변환 내용:
        1. 중첩 구조 → 평탄 리스트
        2. R001 → paraphrase_formal_informative
        3. selection_count → statistics
        """
    
        converted_rules = []
    
        # context_rules 순회
        for context_key, features in initial_data.get('context_rules', {}).items():
            # "formal_informative" → tone="formal", genre="informative"
            parts = context_key.split('_')
            
            if len(parts) >= 2:
                tone = parts[0]   # formal, common, normal
                genre = parts[1]  # informative, narrative 
                
            else:
                # 오류 방지 ("_"가 없는 경우 -> 예외 처리)
                tone = parts[0]
                genre = 'general'
        
            # 각 feature 순회
            for feature_name, rule_data in features.items():
                # rule_id 재생성
                new_rule_id = self._generate_rule_id(feature_name, tone, genre)
            
                # 규칙 변환
                converted_rule = {
                    'rule_id': new_rule_id,  # ← "paraphrase_formal_informative"
                    'feature': feature_name,
                    'context': {
                        'tone': tone,
                        'genre': genre,
                        'complexity': 'any'
                    },
                
                    # 선택 데이터 기반 필드
                    'guideline': self._extract_guideline_from_examples(
                        rule_data.get('examples', [])
                    ),
                    'alternative_guidelines': [],
                    'why_good': [],
                
                    # AVOID 규칙 (선택 데이터에는 없음)
                    'avoid_guidelines': [],
                    'why_bad': [],
                
                    # 신뢰도 (역추론이므로 0.6)
                    'confidence': rule_data.get('confidence', 0.6),
                
                    # 통계 (실제 문장 데이터 기반)
                    'statistics': {
                        'total_feedback_count': rule_data.get('total_samples', 0),
                        'satisfied_count': rule_data.get('selection_count', 0),
                        'dissatisfied_count': 0,  # 선택 데이터에는 불만족 없음
                        'satisfied_weight': float(rule_data.get('selection_count', 0)),
                        'dissatisfied_weight': 0.0,
                        'positive_ratio': rule_data.get('selection_rate', 0.0)
                    },
                
                    # 메타 데이터 정보 추가
                    'source': 'initial_selection_data',  # ← 선택 데이터!
                    'original_rule_id': rule_data.get('rule_id', ''),  # R001 보존
                    'created_at': initial_data.get('created_at', ''),
                    'updated_at': datetime.now().isoformat()
                }
            
                converted_rules.append(converted_rule)
    
        return converted_rules

    def _extract_guideline_from_examples(self, examples):
        """
        예시들로부터 가이드라인 추출
    
        예시:
        - 원문보다 짧게
        - 핵심 정보 유지
        - 자연스러운 표현
        """
        if not examples:
            return "문장 데이터 기반 규칙 (구체적 가이드라인 추출 필요)"
    
        guidelines = []
    
        for ex in examples[:3]:  # 상위 3개만
            original = ex.get('original', '')
            selected = ex.get('selected', '')
        
            # 길이 비교
            if len(selected) < len(original) * 0.8:
                guidelines.append("원문보다 간결하게 표현")
                break
    
        if not guidelines:
            guidelines.append("사용자 선호 패턴 기반")
    
        return " | ".join(guidelines)
        
    
    def _load_rulebook(self):
        """
        룰북 로드 (우선순위 적용)
        
         1. rulebook.json (피드백 기반 기존 룰북)
         2. initial_rules.json (1순위 없을 시, 초기규칙모음집 적용)
            - 첫 피드백 처리 후 rulebook.json으로 저장
         3. 빈 룰북 생성 (에러 방지용)
        """
        
        # 1. 기존 룰북
        if os.path.exists(self.rulebook_path):
            with open(self.rulebook_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"📖 기존 룰북 로드: {len(data.get('rules', []))}개 규칙")
                return data
        
        # 2. 초기 규칙모음집
        elif os.path.exists(self.initial_rules_path):
            with open(self.initial_rules_path, 'r', encoding='utf-8') as f:
                initial_data = json.load(f)
                
                # 초기 규칙 형식 변환
                converted_rules = self._convert_initial_rules(initial_data)
                
                # 메타데이터 추가
                rulebook = {
                    'metadata': {
                        'created_at': initial_data.get('created_at', ''),
                        'total_rules': len(converted_rules),
                        'last_updated': None,
                        'source': 'initial_selection_data'    # 초기 규칙임을 명시
                    },
                    'rules': converted_rules
                }
                
                print(f"📖 초기 규칙 모음집 로드: {len(converted_rules)}개 규칙 (문장 데이터 기반)")
                
                return rulebook
        
        # 3. 빈 룰북 생성
        else:
            print("📖 새 룰북 생성")
            return {
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'total_rules': 0,
                    'last_updated': None,
                    'source': 'empty'
                },
                'rules': []
            }
    
    def _generate_rule_id(self, feature, tone, genre):
        """규칙 ID 생성"""
        key = f"{feature}_{tone}_{genre}".lower()
        return key
    
    def _calculate_confidence(self, satisfied_weight, dissatisfied_weight, 
                             satisfied_count, dissatisfied_count):
        """
        신뢰도 계산 (불만족 피드백에는 마이너스 가중치 적용)
        
        공식:
        - 기본: satisfied / total
        - 불만족 패널티: 0.2 * dissatisfied_weight
        - 만족 횟수가 많을수록 신뢰도에 보너스 적용: count에 따라 +0.02~0.1
        """
        total_count = satisfied_count + dissatisfied_count
        
        if total_count == 0:
            return 0.0
        
        # 기본 신뢰도 (만족 횟수 / 전체 피드백 횟수)
        base_confidence = satisfied_count / total_count
        
        # 불만족 패널티 (신뢰도 -20% 적용)
        penalty = 0.2 * dissatisfied_weight
        
        # 만족 횟수 카운트 (신뢰도 보너스 적용)
        if satisfied_count >= 10:   # 10번 이상 만족 -> 신뢰도 +10%
            bonus = 0.1
        elif satisfied_count >= 5:  # 5번 이상 만족 -> 신뢰도 +5%
            bonus = 0.05
        elif satisfied_count >= 3:  # 3번 이상 만족 -> 신뢰도 +2%
            bonus = 0.02
        else:
            bonus = 0.0  
        
        # 최종 계산 : 기본 신뢰도에 불만족 패널티는 -20%, 만족 횟수에 따른 보너스 적용
        confidence = base_confidence - penalty + bonus
        
        # -1 ~ 1 범위 (클리핑) : -1(완전 부정) ~ 1(완전 긍정=100% 신뢰)
        confidence = max(-1.0, min(confidence, 1.0))
        
        return round(confidence, 2)
    
    
    def _refine_guideline_with_llm(self, guidelines, feature, tone, genre):
        """
        LLM으로 여러 규칙을 하나의 명확한 규칙으로 병합 & 정제
    
        Args:
            guidelines: 원본 규칙 리스트
            feature: 기능 (Paraphrase, Tone Adjust, ...)
            tone: 톤 (formal, normal, ...)
            genre: 장르 (informative, narrative, ...)
    
        Returns:
            str: 병합되고 정제된 규칙
        """
    
        # 규칙이 1개면 그대로 사용
        if len(guidelines) == 1:
            return guidelines[0]
    
        # 여러 개면 LLM으로 병합
        guidelines_text = "\n".join([f"{i+1}. {g}" for i, g in enumerate(guidelines)])
    
        prompt = f"""

다음은 동일한 맥락({feature} 기능, {tone} 톤, {genre} 장르)에서 수집된 여러 규칙들입니다.

규칙들:
{guidelines_text}

요구사항:
1. 위 규칙들을 하나의 명확하고 실행 가능한 규칙으로 병합하세요.
2. 중복된 내용은 제거하고 핵심만 포함하세요.
3. 구체적이고 실행 가능해야 합니다.
4. 70자 이내로 작성하세요.
5. "~하라" 형식으로 끝내세요.

**중요: 반드시 한글로만 답변하세요. 영어, 한자, 기타 언어 사용 금지!**

병합된 규칙만 출력하세요. (설명 없이):"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )
        
            refined = response.choices[0].message.content.strip()
            print(f"   🔄 규칙 병합: {len(guidelines)}개 → 1개")
        
            return refined
        
        except Exception as e:
            print(f"   ⚠️  LLM 병합 실패, 원본 사용: {e}")
            # 실패 시 가장 긴 것 사용
            return max(guidelines, key=len)


    def _generate_avoid_with_llm(self, dissatisfied_rules, do_guideline, 
                             feature, tone, genre):
        """
        LLM으로 자연스러운 AVOID 규칙 생성
    
        Args:
            dissatisfied_rules: 불만족 기반 원본 규칙들
            do_guideline: DO 규칙 (참고용)
            feature, tone, genre: 맥락 정보
    
        Returns:
            list: AVOID 규칙 리스트
        """
    
        if not dissatisfied_rules:
            return []
    
        rules_text = "\n".join([f"- {r}" for r in dissatisfied_rules])
    
        prompt = f"""
다음은 사용자가 불만족한 교정 결과에서 도출된 규칙들입니다.

맥락:
- 기능: {feature}
- 톤: {tone}
- 장르: {genre}

DO 규칙 (참고):
{do_guideline}

불만족 규칙들:
{rules_text}

요구사항:
1. 위 불만족 규칙들을 분석하여 "피해야 할 것"을 명확히 하는 AVOID 규칙을 작성하세요.
2. 자연스러운 부정형 표현을 사용하세요. ("~하지 마라", "~을 피하라" 등)
3. DO 규칙과 상충되지 않으면서 보완하는 내용이어야 합니다.
4. 각 규칙은 50자 이내로 작성하세요.
5. 최대 3개의 AVOID 규칙을 작성하세요.
6. **중요: 반드시 한글로만 답변하세요. 영어, 한자, 기타 언어 사용 금지!**

출력 형식 (규칙만, 번호 없이):
규칙1
규칙2
규칙3"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3
            )
        
            avoid_text = response.choices[0].message.content.strip()
        
            # 줄바꿈으로 분리
            avoid_list = [line.strip() for line in avoid_text.split('\n') 
                         if line.strip() and not line.strip().startswith('#')]
        
            print(f"   🚫 AVOID 규칙 생성: {len(avoid_list)}개")
        
            return avoid_list[:3]  # 최대 3개
        
        except Exception as e:
            print(f"   ⚠️  LLM AVOID 생성 실패, 단순 변환 사용: {e}")
            # 실패 시 단순 치환
            return [r.replace("하라", "하지 마라").replace("사용", "사용 금지") 
                    for r in dissatisfied_rules[:3]]


    def _summarize_reasons_with_llm(self, reasons, sentiment):
        """
        LLM으로 이유들을 요약 & 정제 (선택 사항)
    
        Args:
            reasons: 원본 이유 리스트
            sentiment: "긍정" or "부정"
    
        Returns:
            list: 정제된 이유 리스트 (최대 3개)
        """
    
        if len(reasons) <= 3:
            return reasons
    
        reasons_text = "\n".join([f"- {r}" for r in reasons])
    
        prompt = f"""
다음은 {sentiment}적 피드백의 이유들입니다:

{reasons_text}

요구사항:
1. 위 이유들을 분석하여 핵심적인 3가지로 요약하세요
2. 중복을 제거하고 가장 중요한 것만 선택하세요
3. 각 이유는 30자 이내로 간결하게 작성하세요
4. 자연스러운 한국어로 작성하세요
5. **중요: 반드시 한글로만 답변하세요. 영어, 한자, 기타 언어 사용 금지!**

출력 형식 (이유만, 번호 없이):
이유1
이유2
이유3"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3
            )
        
            summary = response.choices[0].message.content.strip()
        
            # 줄바꿈으로 분리
            summarized = [line.strip() for line in summary.split('\n') 
                         if line.strip() and not line.strip().startswith('#')]
        
            return summarized[:3]
        
        except Exception as e:
            print(f"   ⚠️  이유 요약 실패, 원본 사용: {e}")
            return reasons[:3]
    
    
    def _merge_insights_to_rule(self, insights):
        """
        같은 맥락의 인사이트들을 하나의 규칙으로 병합
        
        만족/불만족 분리 처리:
        - 만족 → DO 규칙 (guideline, why_good) ← LLM으로 병합
        - 불만족 → AVOID 규칙 (avoid_guidelines, why_bad) ← LLM으로 생성
        """
        if not insights:
            return None
        
        first = insights[0]
        feature = first['selected_feature']
        tone = first['context']['tone']
        genre = first['context']['genre']
        
        # ===== 만족/불만족 분리 =====
        satisfied_insights = [i for i in insights if i['feedback'] == '만족']
        dissatisfied_insights = [i for i in insights if i['feedback'] == '불만족']
        
        satisfied_count = len(satisfied_insights)
        dissatisfied_count = len(dissatisfied_insights)
        total_count = len(insights)
        
        satisfied_weight = sum(
            i.get('weight', i.get('recommendation_score', 1.0))
                  for i in satisfied_insights
            )
        
        dissatisfied_weight = sum(
            i.get('weight', i.get('recommendation_score', 1.0))
                for i in dissatisfied_insights
            )
        
        # ===== DO 규칙 (만족 기반) =====
        do_guidelines_raw = []
        why_good_raw = []
        
        for insight in satisfied_insights:
            rule_text = insight['recommended_rule']
            if rule_text not in do_guidelines_raw:
                do_guidelines_raw.append(rule_text)
            
            # why_good 수집
            for reason in insight['why_good_or_bad']:
                if reason not in why_good_raw:
                    why_good_raw.append(reason)
        
        # ===== LLM으로 만족 기반 규칙 병합 및 정제 =====
        if do_guidelines_raw:
            main_guideline = self._refine_guideline_with_llm(
                do_guidelines_raw, 
                feature, 
                tone, 
                genre
            )
            alternative_guidelines = []  # 메인만 사용
        else:
            main_guideline = ""
            alternative_guidelines = []
        
        # 만족한 이유 LLM 활용해 정제
        if why_good_raw and len(why_good_raw) > 3:
            why_good = self._summarize_reasons_with_llm(why_good_raw, "긍정")[:3]
        else:
            why_good = why_good_raw[:3]
        
        
        # ===== AVOID 규칙 (불만족 기반) =====
        avoid_guidelines_raw = []
        why_bad_raw = []
        
        for insight in dissatisfied_insights:
            avoid_text = insight['recommended_rule']
            
            if avoid_text not in avoid_guidelines_raw:
                avoid_guidelines_raw.append(avoid_text)
            
            # why_bad 수집
            for reason in insight['why_good_or_bad']:
                if reason not in why_bad_raw:
                    why_bad_raw.append(reason)
                    
        # ===== LLM으로 불만족 기반 AVOID 규칙 생성 =====
        if avoid_guidelines_raw:
            avoid_guidelines = self._generate_avoid_with_llm(
                avoid_guidelines_raw,
                main_guideline,  # DO 규칙 참고
                feature,
                tone,
                genre
            )
        else:
            avoid_guidelines = []
            
        # 불만족한 이유 LLM 활용해 정제
        if why_bad_raw and len(why_bad_raw) > 3:
            why_bad = self._summarize_reasons_with_llm(why_bad_raw, "부정")[:3]
        else:
            why_bad = why_bad_raw[:3]
                    
        
        # ===== 최종 신뢰도 계산 (불만족에는 패널티, 만족 피드백에는 횟수 많을수록 보너스) =====
        confidence = self._calculate_confidence(
            satisfied_weight, dissatisfied_weight,
            satisfied_count, dissatisfied_count
        )
        
        # 예시 저장
        examples = []
        for insight in insights[:10]:  # 최대 10개만
            ex = {
                'original': insight['original'][:120],  # 120자 제한
                'corrected': insight['corrected_text'][:120],
                'feedback': insight['feedback'],
                'timestamp': insight['metadata']['timestamp']
            }
            examples.append(ex)

        # ===== 규칙 생성 =====
        rule = {
            'rule_id': self._generate_rule_id(feature, tone, genre),
            'feature': feature,
            'context': {
                'tone': tone,
                'genre': genre,
                'complexity': first['context'].get('complexity', 'any')
            },
            
            # DO 규칙
            'guideline': main_guideline,
            'alternative_guidelines': alternative_guidelines,
            'why_good': why_good,
            
            # AVOID 규칙
            'avoid_guidelines': avoid_guidelines,
            'why_bad': why_bad, 
            
            # 신뢰도
            'confidence': confidence,
            
            # 통계
            'statistics': {
                'total_feedback_count': total_count,
                'satisfied_count': satisfied_count,
                'dissatisfied_count': dissatisfied_count,
                'satisfied_weight': round(satisfied_weight, 2),
                'dissatisfied_weight': round(dissatisfied_weight, 2),
                'positive_ratio': round(satisfied_count / total_count if total_count > 0 else 0, 2)
            },

            'examples': examples,
            
            # 메타 데이터 정보 추가
            'source': 'user_feedback',  # ← 피드백 기반
            'created_at': first['metadata']['analyzed_at'],
            'updated_at': datetime.now().isoformat()
        }
        
        return rule
    
    def _resolve_conflict(self, new_rule, old_rule):
        """
        충돌 규칙 해결
        
        Returns:
            'update': 기존 규칙 업데이트
            'delete': 기존 규칙 삭제
            'keep': 기존 유지, 새 규칙 버림
        """
        
        # 1. 신뢰도 체크
        new_conf = new_rule['confidence']
        old_conf = old_rule['confidence']
        
        # 둘 다 마이너스 → 삭제
        if new_conf < 0 and old_conf < 0:
            print(f"   ⚠️  양쪽 모두 부정적 → 규칙 삭제")
            return 'delete'
        
        # 새 규칙만 마이너스 → 기존 유지
        if new_conf < 0 and old_conf >= 0:
            print(f"   ⏭️  새 규칙 부정적 → 기존 유지")
            return 'keep'
        
        # 기존만 마이너스 → 새 규칙으로 교체
        if new_conf >= 0 and old_conf < 0:
            print(f"   🔄 기존 규칙 부정적 → 새 규칙으로 교체")
            return 'update'
        
        # 2. 둘 다 긍정적 → 통계 합산
        return 'update'
    
    def process_insights(self):
        """
        메인 처리 로직
        
        1. insights_queue.json 읽기
        2. 맥락별로 그룹화
        3. 규칙 생성/업데이트/삭제 (하이브리드 매칭)
        4. rulebook.json 저장
        """
        
        print(f"\n{'='*60}")
        print("📚 수집가 실행 - 인사이트 → 규칙 변환")
        print(f"{'='*60}")
        
        # 1. 인사이트 로드
        if not os.path.exists(self.insights_path):
            print(f"⚠️  인사이트 파일 없음: {self.insights_path}")
            return False
        
        with open(self.insights_path, 'r', encoding='utf-8') as f:
            insights_data = json.load(f)
            
            # 파일 형식 자동 감지
            if isinstance(insights_data, list): # 리스트 형식일 경우
                insights = insights_data
            elif isinstance(insights_data, dict): # 딕셔너리 형식일 경우
                insights = insights_data.get('insights', [])
            else:
                insights = []
        
        if not insights:
            print("⚠️  처리할 인사이트 없음")
            return False
        
        print(f"📥 인사이트 {len(insights)}개 로드")
        
        # 2. 맥락별로 그룹화
        grouped = defaultdict(list)
        for insight in insights:
            feature = insight['selected_feature']
            tone = insight['context']['tone']
            genre = insight['context']['genre']
            
            rule_id = self._generate_rule_id(feature, tone, genre)
            grouped[rule_id].append(insight)
        
        print(f"🗂️  {len(grouped)}개 규칙 그룹 생성")
        
        # 3. 규칙 생성/업데이트/삭제
        new_rules = []
        updated_rules = []
        deleted_rules = []
        
        for rule_id, group_insights in grouped.items():
            print(f"\n📋 처리 중: {rule_id} ({len(group_insights)}개 인사이트)")
            
            # 규칙 생성
            new_rule = self._merge_insights_to_rule(group_insights)
            
            if not new_rule:
                continue
            
            # ===== 하이브리드 매칭: rule_id 우선 → 유사도 매칭 =====
            existing_idx = None
            match_method = None
            similarity_score = 0
        
            # Step 1: rule_id 정확 매칭 시도 (빠름)
            for idx, rule in enumerate(self.rulebook['rules']):
                if rule['rule_id'] == rule_id:
                    existing_idx = idx
                    match_method = 'exact'
                    print(f"   ✓ 정확 매칭 발견 (rule_id)")
                    break
        
            # Step 2: rule_id 매칭 실패 시 유사도 매칭 (유연함)
            if existing_idx is None:
                feature = new_rule['feature']
                tone = new_rule['context']['tone']
                genre = new_rule['context']['genre']
                new_guideline = new_rule['guideline']

                # feature별 threshold
                threshold = FEATURE_SIMILARITY_THRESHOLD.get(feature, 85.0)
            
                best_idx = None
                best_similarity = 0
            
                for idx, rule in enumerate(self.rulebook['rules']):
                    # 같은 feature/tone/genre만 비교
                    if (rule['feature'] != feature or
                        rule['context']['tone'] != tone or
                        rule['context']['genre'] != genre):
                        continue
                
                    # guideline 유사도 계산
                    old_guideline = rule.get('guideline', '')
                    if not old_guideline:
                        continue
                
                    similarity = self._calculate_similarity(new_guideline, old_guideline)

                    if similarity >= threshold and similarity > best_similarity:
                        best_similarity = similarity
                        best_idx = idx
            
                if best_idx is not None:
                    existing_idx = best_idx
                    match_method = 'similarity'
                    similarity_score = best_similarity
                    print(f"   ✓ 유사도 매칭 발견 ({best_similarity:.1f}% 유사)")

            # ===== 충돌 해결 =====
            if existing_idx is not None:
                old_rule = self.rulebook['rules'][existing_idx]
                action = self._resolve_conflict(new_rule, old_rule)
            
                if action == 'delete':
                    del self.rulebook['rules'][existing_idx]
                    deleted_rules.append(rule_id)
                    print(f"   🗑️  규칙 삭제")
                
                elif action == 'keep':
                    print(f"   ⏭️  기존 규칙 유지")
                
                elif action == 'update':
                    # 통계 합산
                    new_rule['statistics']['total_feedback_count'] += old_rule['statistics']['total_feedback_count']
                    new_rule['statistics']['satisfied_count'] += old_rule['statistics']['satisfied_count']
                    new_rule['statistics']['dissatisfied_count'] += old_rule['statistics']['dissatisfied_count']
                    new_rule['statistics']['satisfied_weight'] += old_rule['statistics']['satisfied_weight']
                    new_rule['statistics']['dissatisfied_weight'] += old_rule['statistics']['dissatisfied_weight']
                
                    # 신뢰도 재계산
                    new_rule['confidence'] = self._calculate_confidence(
                        new_rule['statistics']['satisfied_weight'],
                        new_rule['statistics']['dissatisfied_weight'],
                        new_rule['statistics']['satisfied_count'],
                        new_rule['statistics']['dissatisfied_count']
                    )

                    new_rule['statistics']['positive_ratio'] = round(
                        new_rule['statistics']['satisfied_count'] / new_rule['statistics']['total_feedback_count']
                        if new_rule['statistics']['total_feedback_count'] > 0 else 0, 2
                    )

                    # 생성 시간 유지
                    new_rule['created_at'] = old_rule['created_at']

                    # examples 병합
                    old_examples = old_rule.get('examples', [])
                    new_examples = new_rule.get('examples', [])
                    merged_examples = (old_examples + new_examples)[-10:]
                    new_rule['examples'] = merged_examples

                    # 매칭 정보 저장 (디버깅용)
                    new_rule['last_match'] = {
                        'method': match_method,
                        'similarity': similarity_score if match_method == 'similarity' else 100.0,
                        'matched_at': datetime.now().isoformat()
                    }
                
                    # 교체
                    self.rulebook['rules'][existing_idx] = new_rule
                    updated_rules.append(rule_id)
                    print(f"   ✏️  규칙 업데이트 (신뢰도: {new_rule['confidence']:.2f})")
            else:
                # 신규 추가
                self.rulebook['rules'].append(new_rule)
                new_rules.append(rule_id)
                print(f"   ✨ 신규 규칙 추가 (신뢰도: {new_rule['confidence']:.2f})")
        
        # 4. 메타데이터 업데이트
        self.rulebook['metadata']['total_rules'] = len(self.rulebook['rules'])
        self.rulebook['metadata']['last_updated'] = datetime.now().isoformat()
        
        # 5. 저장
        self._save_rulebook()
        
        print(f"\n{'='*60}")
        print(f"✅ 수집 완료!")
        print(f"   신규: {len(new_rules)}개")
        print(f"   업데이트: {len(updated_rules)}개")
        print(f"   삭제: {len(deleted_rules)}개")
        print(f"   전체: {len(self.rulebook['rules'])}개 규칙")
        print(f"{'='*60}\n")
        
        return True
    
    def _save_rulebook(self):
        """룰북 저장"""
        os.makedirs('data', exist_ok=True)
        
        with open(self.rulebook_path, 'w', encoding='utf-8') as f:
            json.dump(self.rulebook, f, ensure_ascii=False, indent=2)
        
        print(f"💾 룰북 저장: {self.rulebook_path}")


    def build_prompt_for_context(self, feature, tone, genre, min_confidence=0.5):
        """
        특정 맥락에 대한 LLM 프롬프트 블럭 생성
        
        Args:
            feature: 기능 (Paraphrase, Tone Adjust, Expand, Compress)
            tone: 톤 (formal, normal, common, terminal_word)
            genre: 장르 (informative, narrative, descriptive, dialogue)
            min_confidence: 최소 신뢰도 (기본 0.5)
        
        Returns:
            str: LLM에 넣을 프롬프트 블럭
        """
        do_guidelines = []
        avoid_guidelines = []
        
        # 해당 맥락의 규칙 필터링
        for rule in self.rulebook.get('rules', []):
            # 맥락 매칭
            if (rule.get('feature') != feature or
                rule.get('context', {}).get('tone') != tone or
                rule.get('context', {}).get('genre') != genre):
                continue
            
            confidence = rule.get('confidence', 0)
            
            # DO 규칙 (신뢰도 높음)
            if confidence >= min_confidence:
                guideline = rule.get('guideline', '')
                if guideline:
                    do_guidelines.append(f"- {guideline}")
                
                # alternative도 추가
                for alt in rule.get('alternative_guidelines', []):
                    if alt:
                        do_guidelines.append(f"  * {alt}")
            
            # AVOID 규칙 (신뢰도 낮음)
            elif confidence < 0:
                for avoid in rule.get('avoid_guidelines', []):
                    if avoid:
                        avoid_guidelines.append(f"- {avoid}")
        
        # 규칙이 없으면 빈 문자열
        if not do_guidelines and not avoid_guidelines:
            return ""
        
        # 프롬프트 블럭 생성
        blocks = []
        blocks.append(f"### [{feature} / {tone} / {genre}] 사용자 피드백 기반 규칙")
        
        if do_guidelines:
            blocks.append("\n[선호 패턴]")
            blocks.extend(do_guidelines)
        
        if avoid_guidelines:
            blocks.append("\n[피해야 할 패턴]")
            blocks.extend(avoid_guidelines)
        
        return "\n".join(blocks)
    
    def build_all_prompts(self):
        """
        모든 맥락의 프롬프트 블럭 생성 (테스트용)
        
        Returns:
            dict: {rule_id: prompt_block}
        """
        prompts = {}
        
        features = ['Paraphrase', 'Tone Adjust', 'Expand', 'Compress']
        tones = ['formal', 'normal', 'common', 'terminal_word']
        genres = ['informative', 'narrative', 'descriptive', 'dialogue']
        
        for feature in features:
            for tone in tones:
                for genre in genres:
                    prompt = self.build_prompt_for_context(feature, tone, genre)
                    if prompt:
                        rule_id = self._generate_rule_id(feature, tone, genre)
                        prompts[rule_id] = prompt
        
        return prompts