# evaluator/sentence_evaluator.py
from groq import Groq
import json
import os

class Evaluator:
    """
    평가자 (Evaluator)
    
    역할:
    1. rulebook.json에서 규칙 로드
    2. 맥락에 맞는 규칙 필터링
    3. 규칙을 프롬프트에 포함하여 LLM 교정
    4. 교정된 문장 반환
    """
    
    def __init__(self, api_key, rulebook_path=None):
        """
        초기화
        
        Args:
            api_key: Groq API 키
            rulebook_path: 룰북 파일 경로
        """
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"
        
        # ===== 경로 자동 설정 =====
        if rulebook_path is None:
            try: 
                # 로컬 사용 시
                current_file = os.path.abspath(__file__)
                evaluator_dir = os.path.dirname(current_file)
                project_root = os.path.dirname(evaluator_dir)
            except NameError:
                # 주피터 사용 시
                current_dir = os.getcwd()
                if 'notebooks' in current_dir:
                    project_root = os.path.dirname(current_dir)
                else:
                    project_root = current_dir
                print(f"📍 주피터 환경 감지: {project_root}")
                
            rulebook_path = os.path.join(project_root, 'data', 'rulebook.json')
        
        
        self.rulebook_path = rulebook_path
        
        # 룰북 로드
        self.rulebook = self._load_rulebook()
        
        print(f"✅ 평가자 초기화 완료")
        print(f"   규칙 수: {len(self.rulebook.get('rules', []))}개")
    
    def _load_rulebook(self):
        """룰북 로드"""
        if not os.path.exists(self.rulebook_path):
            print(f"⚠️  룰북 파일 없음: {self.rulebook_path}")
            return {'rules': []}
        
        with open(self.rulebook_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data
    
    def get_applicable_rules(self, feature, tone, genre, min_confidence=0.5, max_rules=5):
        """
        맥락에 맞는 규칙 찾기
        
        Args:
            feature: 기능 (Paraphrase, Tone Adjust, Expand, Compress)
            tone: 톤 (formal, normal, common, terminal_word)
            genre: 장르 (informative, narrative, descriptive, dialogue)
            min_confidence: 최소 신뢰도 (기본 0.5)
            max_rules: 최대 규칙 수 (기본 5개)
        
        Returns:
            list: 적용 가능한 규칙 리스트
        """
        
        # rule_id 생성
        rule_id = f"{feature}_{tone}_{genre}".lower()
        
        # 해당 규칙 찾기
        applicable = []
        
        for rule in self.rulebook.get('rules', []):
            # 정확히 일치하는 규칙
            if rule['rule_id'] == rule_id:
                # 신뢰도 체크
                if rule['confidence'] >= min_confidence:
                    applicable.append(rule)
        
        # 신뢰도 순으로 정렬 (높은 것부터)
        applicable.sort(key=lambda r: r['confidence'], reverse=True)
        
        # 최대 개수 제한
        return applicable[:max_rules]
    
    def _build_prompt(self, original_text, feature, tone, genre, rules):
        """
        규칙을 포함한 프롬프트 생성
        
        Args:
            original_text: 원문
            feature: 기능
            tone: 톤
            genre: 장르
            rules: 적용할 규칙 리스트
        
        Returns:
            str: 완성된 프롬프트
        """
        
        # 기능 설명
        feature_desc = {
            'Paraphrase': '문장을 간결하고 명확하게 다듬기',
            'Tone Adjust': '문장의 톤을 조정하기',
            'Expand': '문장을 구체적으로 확장하기',
            'Compress': '문장을 핵심만 남기고 압축하기'
        }
        
        # 톤 설명
        tone_desc = {
            'formal': '격식있는',
            'normal': '일상적인',
            'common': '평범한',
            'terminal_word': '명사형 종결'
        }
        
        # 장르 설명
        genre_desc = {
            'informative': '정보 전달',
            'narrative': '서사적',
            'descriptive': '묘사적',
            'dialogue': '대화체'
        }
        
        # DO 규칙 수집
        do_rules = []
        avoid_rules = []
        
        for rule in rules:
            # 메인 가이드라인
            if rule.get('guideline'):
                do_rules.append(rule['guideline'])
            
            # AVOID 규칙
            for avoid in rule.get('avoid_guidelines', []):
                if avoid not in avoid_rules:
                    avoid_rules.append(avoid)
        
        # 프롬프트 생성
        prompt = f"""당신은 한국어 문장 교정 전문가입니다.

[원문]
{original_text}

[교정 작업]
{feature_desc.get(feature, feature)}

[목표 스타일]
- 톤: {tone_desc.get(tone, tone)}
- 장르: {genre_desc.get(genre, genre)}
"""
        
        # DO 규칙 추가
        if do_rules:
            prompt += "\n[반드시 따라야 할 규칙]\n"
            for i, rule in enumerate(do_rules, 1):
                prompt += f"{i}. {rule}\n"
        
        # AVOID 규칙 추가
        if avoid_rules:
            prompt += "\n[피해야 할 것]\n"
            for i, rule in enumerate(avoid_rules, 1):
                prompt += f"{i}. {rule}\n"
        
        prompt += """
요구사항:
1. 위 규칙을 반드시 따르세요
2. 원문의 의미를 유지하세요
3. 자연스러운 한국어로 작성하세요
4. 교정된 문장만 출력하세요 (설명 없이)

교정된 문장:"""
        
        return prompt
    
    def correct(self, original_text, feature, tone='normal', genre='general', 
                min_confidence=0.5):
        """
        문장 교정 실행
        
        Args:
            original_text: 원문
            feature: 기능 (Paraphrase, Tone Adjust, Expand, Compress)
            tone: 톤 (formal, normal, common, terminal_word)
            genre: 장르 (informative, narrative, descriptive, dialogue)
            min_confidence: 최소 신뢰도
        
        Returns:
            dict: {
                'corrected': 교정된 문장,
                'rules_applied': 적용된 규칙 수,
                'confidence': 평균 신뢰도
            }
        """
        
        print(f"\n{'='*60}")
        print(f"📝 교정 시작")
        print(f"{'='*60}")
        print(f"기능: {feature}")
        print(f"톤: {tone}, 장르: {genre}")
        print(f"원문: {original_text[:50]}...")
        
        try:
            # 1. 적용 가능한 규칙 찾기
            rules = self.get_applicable_rules(feature, tone, genre, min_confidence)
            
            if not rules:
                print("⚠️  적용 가능한 규칙 없음 (기본 교정 실행)")
                # 규칙 없이 기본 교정
                prompt = f"""당신은 한국어 문장 교정 전문가입니다.

원문: {original_text}

작업: {feature}

자연스럽게 교정하세요. 교정된 문장만 출력하세요."""
                
                rules_count = 0
                avg_confidence = 0.0
            else:
                print(f"✅ {len(rules)}개 규칙 적용")
                for rule in rules:
                    print(f"   - {rule['rule_id']} (신뢰도: {rule['confidence']})")
                
                # 2. 프롬프트 생성
                prompt = self._build_prompt(original_text, feature, tone, genre, rules)
                
                rules_count = len(rules)
                avg_confidence = sum(r['confidence'] for r in rules) / len(rules)
            
            # 3. LLM 호출
            print("\n🤖 LLM 교정 중...")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            corrected = response.choices[0].message.content.strip()
            
            # 4. 결과 반환
            print(f"✨ 교정 완료!")
            print(f"   결과: {corrected[:50]}...")
            print(f"{'='*60}\n")
            
            return {
                'corrected': corrected,
                'rules_applied': rules_count,
                'confidence': round(avg_confidence, 2) if rules_count > 0 else 0.0
            }
            
        except Exception as e:
            print(f"❌ 교정 실패: {e}")
            return {
                'corrected': original_text,
                'rules_applied': 0,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def reload_rulebook(self):
        """룰북 다시 로드 (규칙이 업데이트된 경우)"""
        print("🔄 룰북 재로드 중...")
        self.rulebook = self._load_rulebook()
        print(f"✅ 재로드 완료: {len(self.rulebook.get('rules', []))}개 규칙")