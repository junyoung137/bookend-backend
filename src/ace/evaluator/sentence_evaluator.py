# src/ace/evaluator/sentence_evaluator.py

import json
import os
from datetime import datetime
import requests

class Evaluator:
    def __init__(self, api_key, rulebook_path=None):
        """평가자 초기화 (Groq API - 직접 HTTP)"""
        
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        
        self.rulebook_path = rulebook_path or '/app/src/ace/data/rulebook.json'
        self.rules = self._load_rulebook()
        
        print("평가자 초기화 완료 (직접 HTTP 요청)")
        print(f"   룰북 경로: {self.rulebook_path}")
        print(f"   로드된 규칙 수: {len(self.rules)}개")
    
    def _load_rulebook(self):
        """룰북 로드"""
        if not os.path.exists(self.rulebook_path):
            print(f"룰북 파일이 없습니다: {self.rulebook_path}")
            return []
        
        try:
            with open(self.rulebook_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('rules', [])
        except Exception as e:
            print(f"룰북 로드 실패: {e}")
            return []
    
    def correct(
        self,
        original_text,
        feature,
        tone='normal',
        genre='informative',
        min_confidence=0.5
    ):
        """문장 교정 실행"""
        print(f"\n{'='*60}")
        print(f"교정 시작 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        print(f"기능 → {feature} | 톤 → {tone} | 장르 → {genre}")
        print(f"원문 → {original_text[:100]}...")
        
        # 1. 적용 가능한 규칙 필터링
        applicable_rules = self._filter_rules(feature, tone, genre)
        
        if not applicable_rules:
            print("적용 가능한 규칙 없음 → 기본 교정 실행")
            return self._basic_correction(original_text, feature, tone, genre)
        
        # 2. 규칙 기반 교정
        print(f"📋 적용 가능한 규칙: {len(applicable_rules)}개")
        return self._apply_rules(original_text, applicable_rules, feature, tone, genre)
    
    def _filter_rules(self, feature, tone, genre):
        """적용 가능한 규칙 필터링"""
        applicable = []
        for rule in self.rules:
            if (rule.get('feature') == feature and
                rule.get('tone') == tone and
                rule.get('genre') == genre):
                applicable.append(rule)
        return applicable
    
    def _apply_rules(self, text, rules, feature, tone, genre):
        """규칙 적용 교정"""
        rules_text = "\n".join([
            f"- {r.get('pattern', '')}: {r.get('suggestion', '')}"
            for r in rules[:5]
        ])
        
        prompt = f"""다음 문장을 교정하세요.

**원문**: {text}

**교정 기능**: {feature}
**톤**: {tone}
**장르**: {genre}

**적용할 규칙**:
{rules_text}

**지침**:
1. 위 규칙을 최대한 반영
2. 원문의 의미 유지
3. 자연스러운 한국어
4. 교정된 문장만 출력 (설명 없이)
"""
        
        result = self._call_groq_direct(prompt, max_retries=3)
        
        if result:
            return {
                'corrected': result,
                'rules_applied': len(rules),
                'confidence': 0.8
            }
        else:
            return {
                'corrected': text,
                'rules_applied': 0,
                'confidence': 0.0
            }
    
    def _basic_correction(self, text, feature, tone, genre):
        """기본 교정 (규칙 없을 때)"""
        print("기본 교정 실행")
        
        feature_instructions = {
            'Paraphrase': '원문의 의미를 유지하면서 다르게 표현하세요.',
            'Expand': '원문을 더 상세하고 풍부하게 확장하세요.',
            'Shorten': '원문의 핵심만 간결하게 요약하세요.',
            'Formalize': '격식있고 전문적인 표현으로 변환하세요.',
        }
        
        instruction = feature_instructions.get(feature, '문장을 개선하세요.')
        
        prompt = f"""다음 문장을 교정하세요.

**원문**: {text}

**교정 방식**: {instruction}
**톤**: {tone}
**장르**: {genre}

**지침**:
1. {instruction}
2. 원문의 의미 유지
3. 자연스러운 한국어
4. 교정된 문장만 출력 (설명 없이)
"""
        
        result = self._call_groq_direct(prompt, max_retries=3)
        
        if result:
            return {
                'corrected': result,
                'rules_applied': 0,
                'confidence': 0.6
            }
        else:
            print("최종 교정 실패 → 원문 반환")
            return {
                'corrected': text,
                'rules_applied': 0,
                'confidence': 0.0
            }
    
    def _call_groq_direct(self, prompt, max_retries=3):
        """
        Groq API 직접 HTTP 호출 (가장 안정적)
        
        Args:
            prompt: 프롬프트
            max_retries: 최대 재시도 횟수
            
        Returns:
            str: 생성된 텍스트 또는 None
        """
        import time
        
        for attempt in range(max_retries):
            try:
                print(f"Groq 호출 중... (시도 {attempt + 1}/{max_retries})")
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": 1000
                }
                
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=120,  # 120초 타임아웃
                    verify=False  # SSL 검증 비활성화
                )
                
                if response.status_code == 200:
                    data = response.json()
                    result = data['choices'][0]['message']['content'].strip()
                    print(f"✅ Groq 호출 성공 (길이: {len(result)}자)")
                    return result
                else:
                    print(f"⚠️ Groq API 오류: {response.status_code} - {response.text}")
                    
            except Exception as e:
                print(f"Groq 호출 실패 (시도 {attempt + 1}): {e}")
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"{wait_time}초 후 재시도...")
                    time.sleep(wait_time)
                else:
                    print("최대 재시도 횟수 초과")
                    return None
        
        return None
