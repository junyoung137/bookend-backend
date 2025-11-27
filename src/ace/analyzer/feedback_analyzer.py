from groq import Groq
import json
from datetime import datetime
import sys
import os

sys.path.append('..')
from config.definitions import FEATURE_DEFINITIONS, TONE_DEFINITIONS, GENRE_DEFINITIONS

class Analyzer:
    """
    분석가
    
    역할:
    1. 피드백 데이터 받아서 LLM으로 "왜 좋은지/나쁜지" 분석
    3. 인사이트 생성
    4. 저장 -> 수집가에게 전송
    
    Args:
        api_key: Groq API 키
    """
    
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"
        self.analysis_log = []
        self.feature_defs = FEATURE_DEFINITIONS  # 4가지 기능 정의
        self.tone_defs = TONE_DEFINITIONS        # 톤 정의 
        self.genre_defs = GENRE_DEFINITIONS      # 장르 정의


    def _get_user_segment(self, user_id):
        """
        피드백 개수로 세그먼트 자동 분류 (5단계)
        
        세그먼트 기준:
        - new (신규): 피드백 < 5개 → 3-5회만에 개인화!
        - regular (일반): 피드백 5-15개
        - growing (성장): 피드백 16-30개
        - engaged (열성): 피드백 31-50개
        - power (파워): 피드백 51개 이상
        """
        try:
            import psycopg2
            import os
        
            # PostgreSQL 연결 정보 (.env에서 읽기)
            conn = psycopg2.connect(
                host=os.getenv("ACE_DB_HOST", "localhost"),
                port=int(os.getenv("ACE_DB_PORT", "5432")),
                database=os.getenv("ACE_DB_NAME", "bookend_ace"),
                user=os.getenv("ACE_DB_USER", "ace_admin"),
                password=os.getenv("ACE_DB_PASSWORD", "")
            )
        
            cursor = conn.cursor()
        
            # 피드백 개수 조회 (processed=true만)
            cursor.execute("""
                SELECT COUNT(*) 
                FROM feedbacks
                WHERE user_id = %s AND processed = 1
            """, (user_id,))
        
            result = cursor.fetchone()
            feedback_count = result[0] if result else 0
        
            cursor.close()
            conn.close()
        
            # 세그먼트 분류
            if feedback_count < 5:
                segment = 'new'
            elif feedback_count <= 15:
                segment = 'regular'
            elif feedback_count <= 30:
                segment = 'growing'
            elif feedback_count <= 50:
                segment = 'engaged'
            else:
                segment = 'power'
        
            print(f"  📊 {user_id} → 피드백 {feedback_count}개 → {segment} 세그먼트")
        
            return segment
    
        except Exception as e:
            print(f"  ⚠️ 세그먼트 조회 실패: {e}")
            import traceback
            traceback.print_exc()
            return 'new'  # 기본값: 신규
    
    def analyze_feedback(self, feedback_data):
        """
        단일 피드백 분석
        
        Args:
            feedback_data: {
                'user_id': str,
                'original': str,          # 원문
                'corrected_text': str,    # 교정문
                'selected_feature': str,  # 추천 기능('Paraphrase', 'Tone Adjust', 'Expand', 'Compress')
                'feedback': '만족' or '불만족',
                'context': {              # 맥락 정보 (톤/장르/복잡도/신뢰도 점수)
                    'tone': str,
                    'genre': str,
                    'complexity': str,
                    'recommendation_score': float
                },
                'timestamp': str          # 사용자가 피드백 남긴 시간
            }
        
        Returns:
            insight: {
                'user_id': str,
                'original': str,              # 원문
                'corrected_text': str,        # 교정문
                'selected_feature': str,      # 4가지 추천 기능 분류
                'feedback': str,
                'segment': str,
                'context': dict,
                'why_good_or_bad': [str],     # 분석한 인사이트 이유(왜 만족/불만족인지 분석)
                'key_characteristics': [str], # 교정문의 특징 분석
                'key_insight': str,           # 핵심 인사이트
                'recommended_rule': str,      # 수집가용 규칙
                'metadata': {                 # 부가정보
                    'timestamp': str,         # 사용자가 피드백 남긴 시간
                    'analyzed_at': str        # 분석한 시간
                }
            }
        """
        
        print(f"\n{'='*60}")
        print(f"🔍 피드백 분석 중...")
        print(f"{'='*60}")
        
        # 기능 정보 가져오기
        feature = feedback_data['selected_feature']
        feature_info = self.feature_defs.get(feature, {}) # 기능 정보 value 가져오기
        
        # 맥락 정보
        tone = feedback_data['context']['tone']
        genre = feedback_data['context']['genre']
        complexity = feedback_data['context'].get('complexity', 'unknown')

        # 세그먼트 정보 (피드백 개수 기반 자동 분류)
        segment = self._get_user_segment(feedback_data['user_id'])
    
        # 세그먼트별 가중치
        segment_multiplier = {
            'new': 1.5,      # 신규 (<5개): 초기 데이터 귀중!
            'regular': 1.2,  # 일반 (5-15개): 패턴 형성 중
            'growing': 1.0,  # 성장 (16-30개): 패턴 안정화
            'engaged': 1.0,  # 열성 (31-50개): 충분한 데이터
            'power': 0.8     # 파워 (51+개): 패턴 완전 파악됨
        }
    
        segment_weight = segment_multiplier.get(segment, 1.0)
        
        # 프롬프트
        prompt = f"""당신은 문장 교정 피드백을 분석하는 전문가입니다.

**=== 원문 ===**
{feedback_data['original']}

**=== 적용된 기능 ===**
{feature} ({feature_info.get('name', feature)})

기능 목표: {feature_info.get('system_prompt', '').split('**목표**:')[1].split('**')[0].strip() if '**목표**:' in feature_info.get('system_prompt', '') else '정보 없음'}

기능 제약:
{chr(10).join([f"- {c}" for c in feature_info.get('constraints', [])])}

**=== 교정된 문장 ===**
{feedback_data['corrected_text']}

**=== 사용자 피드백 ===**
- 만족도: {feedback_data['feedback']}
- 추천 점수: {feedback_data['context'].get('recommendation_score', 'N/A')}

**=== 맥락 정보 ===**
- 톤(Tone): {tone}
  → {self.tone_defs.get(tone, '정의 없음').split('.')[0] if tone in self.tone_defs else '정의 없음'}
  
- 장르(Genre): {genre}
  → {self.genre_defs.get(genre, '정의 없음').split('.')[0] if genre in self.genre_defs else '정의 없음'}
  
- 복잡도(Complexity): {complexity}

---

**임무:**
이 교정 문장에 대한 사용자 피드백({feedback_data['feedback']})이 왜 나왔는지 심층 분석하세요.

**분석 포인트:**
1. **기능 적용 적절성**: {feature} 기능이 제약사항을 잘 지켰는가?
2. **맥락 부합성**: {tone} + {genre} + {complexity} 조합에 맞는 교정인가?
3. **구체적 장단점**: 어떤 표현/구조가 좋았거나 나빴는가?
4. **개선 가능성**: 이 피드백으로부터 어떤 규칙을 도출할 수 있는가?

**출력 형식 (JSON만, 마크다운 제거):**
{{
  "why_good_or_bad": [
    "구체적 이유1 (예: '간결성 30% 향상으로 가독성 증가')",
    "구체적 이유2 (예: 'formal 톤에 부적합한 구어체 사용')",
    "구체적 이유3"
  ],
  "key_characteristics": [
    "특징1 (예: '간결함 - 원문 대비 25% 단축')",
    "특징2 (예: '격식체 유지 - ~입니다 종결')",
    "특징3 (예: '정보성 강화 - 핵심 키워드 보존')"
  ],
  "key_insight": "한 문장 핵심 인사이트 (규칙으로 변환 가능하게 작성)",
  "recommended_rule": "수집가가 규칙으로 추가할 수 있는 구체적 지침 (예: '{genre}에서 {feature} 사용 시 ~하라')"
}}

**중요 지침:**
1. 다른 피드백과 차별화된 분석을 제공하라
2. 일반적인 표현("간결성 향상", "가독성 증가", "격식체 유지") 최소화
3. 이 맥락과 문장만의 고유한 특징에 집중하라
4. recommended_rule은 구체적이고 실행 가능해야 함
5. "~됨", "~함" 같은 명사형 종결 최소화, 능동적 표현 사용
6. **언어 제약:** 반드시 순한글만 사용. 한자, 영어, 러시아어, 기타 외국어 절대 금지
   - ❌ 나쁜 예: "正式한", "отсутств", "formal"
   - ✅ 좋은 예: "격식있는", "부족함", "격식"

**금지 표현 목록:**
- "원문 대비 XX% 단축/증가"
- "핵심 키워드 보존"
- "간결함"을 key_characteristics에 반복 사용

**중요: JSON만 출력, 백틱(```) 없이!**
"""

        try:
            # LLM 호출
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=1500,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result_text = response.choices[0].message.content.strip()
            result_text = result_text.replace('```json', '').replace('```', '').strip()
            analysis_result = json.loads(result_text)
            
            
            # 최종 인사이트 구조
            insight = {
                'user_id': feedback_data['user_id'],
                'original': feedback_data['original'],
                'corrected_text': feedback_data['corrected_text'],
                'selected_feature': feedback_data['selected_feature'],
                'feedback': feedback_data['feedback'],
                'segment': segment,
                'weight': segment_weight,
                'context': feedback_data['context'],
                
                'why_good_or_bad': analysis_result['why_good_or_bad'],
                'key_characteristics': analysis_result['key_characteristics'],
                'key_insight': analysis_result['key_insight'],
                'recommended_rule': analysis_result['recommended_rule'],
                'metadata': {
                    'timestamp': feedback_data['timestamp'],
                    'analyzed_at': datetime.now().isoformat()
                }
            }
            
            # 로그 저장
            self.analysis_log.append(insight)
            
            print("✅ 분석 완료!")
            print(f"사용자: {insight['user_id']}")
            print(f"기능: {insight['selected_feature']}")
            print(f"핵심: {insight['key_insight']}")
            print(f"추천 규칙: {insight['recommended_rule']}")
            
            return insight
            
        except Exception as e:
            print(f"❌ 오류: {e}")
            if 'result_text' in locals():
                print(f"응답 텍스트: {result_text[:200]}...")
            return None
    
    def batch_analyze(self, feedback_list):
        """
        여러 피드백 일괄 분석
        """
        insights = []
        processed_ids = []  # 대기열에 있던 피드백 처리완료 표시
        
        print(f"\n{'='*60}")
        print(f"📊 총 {len(feedback_list)}개 피드백 분석 시작")
        print(f"{'='*60}")
        
        for i, feedback in enumerate(feedback_list, 1):
            print(f"\n[{i}/{len(feedback_list)}]")
            insight = self.analyze_feedback(feedback)
            
            if insight:
                insights.append(insight)
                
                if 'id' in feedback:
                    processed_ids.append(feedback['id'])
        
        print(f"\n✨ 분석 완료: {len(insights)}/{len(feedback_list)} 성공")
        
        # DB 업데이트 실행
        if processed_ids:
            try:
                from src.ace.db.handler import mark_as_processed
                mark_as_processed(processed_ids)
                print(f"🔄 DB 업데이트: {len(processed_ids)}개")
            except Exception as e:
                print(f"🚨 DB 업데이트 실패: {e}")
        
        else:
            print("⏭️  처리할 ID 없음")
        
        return insights
    
    def export_for_collector(self, output_path=None):
        """
        수집가에게 전달할 형식으로 저장
        
        insights_queue.json에 저장 → 수집가가 주기적으로 읽어감
        
        Args:
            output_path: 출력 경로 (None이면 자동으로 프로젝트 루트/data/insights_queue.json)
        """
        
        if not self.analysis_log:
            print("⚠️ 분석 결과 없음")
            return False
        
        # 경로 자동 설정
        if output_path is None:
            try:
                # 로컬 -> 현 파일 위치에서 프로젝트 루트 탐색
                current_file = os.path.abspath(__file__)  # analyzer/feedback_analyzer.py
                analyzer_dir = os.path.dirname(current_file)  # analyzer/
                project_root = os.path.dirname(analyzer_dir)  # bookend-ace/
            except NameError:
                # 주피터 -> 현 작업 디렉토리 기준
                current_dir = os.getcwd()
                # notebooks/ 에서 실행 중이면 상위로
                if 'notebooks' in current_dir:
                    project_root = os.path.dirname(current_dir)
                else:
                    project_root = current_dir
                print(f"📍 주피터 환경 감지: {project_root}")
                
            # data 폴더 경로
            data_dir = os.path.join(project_root, 'data')
            output_path = os.path.join(data_dir, 'insights_queue.json')
            print(f"📁 자동 경로 설정: {output_path}")
        
        export_data = {
            'metadata': {
                'total_insights': len(self.analysis_log),
                'export_time': datetime.now().isoformat(),
                'note': '수집가는 이 인사이트를 받아 규칙 모음집 업데이트'
            },
            'insights': self.analysis_log
        }
        
        try:
            # data 폴더 확인
            data_dir = os.path.dirname(output_path)
            os.makedirs(data_dir, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            print(f"\n💾 전달용 파일 저장: {output_path}")
            print(f"   총 {len(self.analysis_log)}개 인사이트")
            
            # 기능별 통계
            feature_counts = {}
            for insight in self.analysis_log:
                feature = insight['selected_feature']
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
            
            print("\n📈 기능별 분석 결과:")
            for feature, count in feature_counts.items():
                print(f"   - {feature}: {count}개")
            
            return True
            
        except Exception as e:
            print(f"❌ 저장 실패: {e}")
            import traceback
            traceback.print_exc()
            return False