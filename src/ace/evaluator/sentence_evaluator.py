# evaluator/sentence_evaluator.py
import json
import os
import time
from datetime import datetime
from groq import Groq


class Evaluator:
    """
    Groq 기반 개인화 문장 교정기 (rulebook.json 활용)
    """

    def __init__(self, api_key, rulebook_path=None):
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"

        # ===== rulebook 경로 자동 탐색 =====
        if rulebook_path is None:
            try:
                current_file = os.path.abspath(__file__)
                evaluator_dir = os.path.dirname(current_file)
                project_root = os.path.dirname(evaluator_dir)
            except NameError:  # Jupyter 환경
                current_dir = os.getcwd()
                project_root = os.path.dirname(current_dir) if 'notebooks' in current_dir else current_dir
                print(f"주피터 환경 감지 → 프로젝트 루트: {project_root}")

            rulebook_path = os.path.join(project_root, 'data', 'rulebook.json')

        self.rulebook_path = rulebook_path
        self.rulebook = self._load_rulebook()

        print(f"평가자 초기화 완료")
        print(f"   룰북 경로: {self.rulebook_path}")
        print(f"   로드된 규칙 수: {len(self.rulebook.get('rules', []))}개")

    def _load_rulebook(self):
        if not os.path.exists(self.rulebook_path):
            print(f"룰북 파일이 없습니다: {self.rulebook_path}")
            return {'rules': []}

        with open(self.rulebook_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data

    def reload_rulebook(self):
        """룰북 갱신 시 호출"""
        print("룰북 재로드 중...")
        self.rulebook = self._load_rulebook()
        print(f"재로드 완료 → {len(self.rulebook.get('rules', []))}개 규칙")

    def get_applicable_rules(self, feature, tone, genre, min_confidence=0.5, max_rules=5):
        rule_id = f"{feature}_{tone}_{genre}".lower()
        applicable = []

        for rule in self.rulebook.get('rules', []):
            if rule.get('rule_id') == rule_id and rule.get('confidence', 0) >= min_confidence:
                applicable.append(rule)

        applicable.sort(key=lambda r: r.get('confidence', 0), reverse=True)
        return applicable[:max_rules]

    def _build_prompt(self, original_text, feature, tone, genre, rules):
        feature_desc = {
            'Paraphrase': '문장을 간결하고 명확하게 다듬기',
            'Tone Adjust': '문장의 톤을 조정하기',
            'Expand': '문장을 구체적으로 확장하기',
            'Compress': '문장을 핵심만 남기고 압축하기'
        }
        tone_desc = {'formal': '격식 있는', 'normal': '일상적인', 'common': '평범한', 'terminal_word': '명사형 종결'}
        genre_desc = {'informative': '정보 전달', 'narrative': '서사적', 'descriptive': '묘사적', 'dialogue': '대화체'}

        do_rules = [r['guideline'] for r in rules if r.get('guideline')]
        avoid_rules = []
        for r in rules:
            avoid_rules.extend(r.get('avoid_guidelines', []))

        prompt = f"""당신은 한국어 문장 교정 전문가입니다.

[원문]
{original_text}

[교정 작업]
{feature_desc.get(feature, feature)}

[목표 스타일]
- 톤: {tone_desc.get(tone, tone)}
- 장르: {genre_desc.get(genre, genre)}
"""

        if do_rules:
            prompt += "\n[반드시 따라야 할 규칙]\n"
            for i, rule in enumerate(do_rules, 1):
                prompt += f"{i}. {rule}\n"

        if avoid_rules:
            prompt += "\n[피해야 할 표현/행동]\n"
            for i, rule in enumerate(set(avoid_rules), 1):  # 중복 제거
                prompt += f"{i}. {rule}\n"

        prompt += """
요구사항:
1. 위 모든 규칙을 철저히 준수하세요.
2. 원문의 의미를 절대 왜곡하지 마세요.
3. 가장 자연스럽고 매끄러운 한국어로 작성하세요.
4. 교정된 문장만 출력하고, 그 외 어떤 설명도 추가하지 마세요.

교정된 문장:"""

        return prompt

    def correct(self, original_text, feature, tone='normal', genre='general',
                min_confidence=0.5, max_retries=3):
        print(f"\n{'='*60}")
        print(f"교정 시작 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        print(f"기능 → {feature} | 톤 → {tone} | 장르 → {genre}")
        print(f"원문 → {original_text[:70]}{'...' if len(original_text) > 70 else ''}")

        rules = self.get_applicable_rules(feature, tone, genre, min_confidence)

        if not rules:
            print("적용 가능한 규칙 없음 → 기본 교정 실행")
            prompt = f"""한국어 문장 교정 전문가로서 다음 문장을 자연스럽게 다듬어 주세요.
원문: {original_text}
작업: {feature}
교정된 문장만 출력하세요."""
            avg_confidence = 0.0
            rules_count = 0
        else:
            print(f"적용 규칙 {len(rules)}개")
            for r in rules:
                print(f"   • {r['rule_id']} (신뢰도: {r.get('confidence', 0):.2f})")
            prompt = self._build_prompt(original_text, feature, tone, genre, rules)
            avg_confidence = sum(r.get('confidence', 0) for r in rules) / len(rules)
            rules_count = len(rules)

        # Groq 호출 (재시도 로직 포함)
        for attempt in range(max_retries):
            try:
                print(f"Groq 호출 중... (시도 {attempt + 1}/{max_retries})")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=800,
                    temperature=0.3,
                    timeout=30.0
                )
                corrected = response.choices[0].message.content.strip()
                print("교정 완료!")
                print(f"{'='*60}\n")
                return {
                    'corrected': corrected,
                    'rules_applied': rules_count,
                    'confidence': round(avg_confidence, 3)
                }

            except Exception as e:
                print(f"Groq 호출 실패 (시도 {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    print(f"{wait}초 후 재시도...")
                    time.sleep(wait)

        # 최종 실패 시 원문 반환
        print("최종 교정 실패 → 원문 반환")
        print(f"{'='*60}\n")
        return {
            'corrected': original_text,
            'rules_applied': rules_count,
            'confidence': round(avg_confidence, 3),
            'error': 'Groq API timeout or error after retries'
        }
