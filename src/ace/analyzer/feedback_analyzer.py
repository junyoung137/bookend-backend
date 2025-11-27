# analyzer/feedback_analyzer.py - Groq로 피드백 분석

import psycopg2
from datetime import datetime
import json
from config.settings import get_settings

settings = get_settings()

class Analyzer:
    def __init__(self, api_key):
        """Groq API 초기화"""
        self.api_key = api_key
        self.db_config = settings.ace.db_config
        
    def _get_user_segment(self, user_id):
        """사용자 세그먼트 조회"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # ✅ BOOLEAN 타입으로 수정
            cursor.execute("""
                SELECT COUNT(*) as feedback_count,
                       AVG(CASE WHEN feedback = '만족' THEN 1 ELSE 0 END) as satisfaction_rate
                FROM feedbacks
                WHERE user_id = %s AND processed = TRUE
            """, (user_id,))
            
            row = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if row:
                count, rate = row
                if count >= 10:
                    return 'power_user'
                elif rate and rate > 0.7:
                    return 'satisfied_user'
                else:
                    return 'new_user'
            
            return 'new_user'
            
        except Exception as e:
            print(f"  ⚠️ 세그먼트 조회 실패: {e}")
            return 'unknown'
    
    def batch_analyze(self, feedbacks):
        """
        피드백 배치 분석 (Groq API 사용)
        
        Args:
            feedbacks: 피드백 리스트
            
        Returns:
            list: 인사이트 리스트
        """
        print(f"\n{'='*60}")
        print(f"📊 총 {len(feedbacks)}개 피드백 분석 시작")
        print(f"{'='*60}")
        
        insights = []
        
        for i, fb in enumerate(feedbacks, 1):
            print(f"\n[{i}/{len(feedbacks)}]")
            print(f"{'='*60}")
            print(f"🔍 피드백 분석 중...")
            print(f"{'='*60}")
            
            try:
                # 사용자 세그먼트 조회
                segment = self._get_user_segment(fb['user_id'])
                
                # Groq API로 분석
                insight = self._analyze_with_groq(fb, segment)
                
                if insight:
                    insights.append(insight)
                    print(f"  ✅ 인사이트 생성 성공")
                else:
                    print(f"  ⚠️ 인사이트 생성 실패")
                    
            except Exception as e:
                print(f"  ❌ 오류: {e}")
                continue
        
        print(f"\n✨ 분석 완료: {len(insights)}/{len(feedbacks)} 성공")
        return insights
    
    def _analyze_with_groq(self, feedback, segment):
        """Groq API로 피드백 분석"""
        from groq import Groq
        import time
        
        client = Groq(api_key=self.api_key)
        
        prompt = f"""다음 피드백을 분석하고 인사이트를 추출하세요.

**피드백 정보**:
- 사용자: {feedback['user_id']}
- 세그먼트: {segment}
- 기능: {feedback['selected_feature']}
- 원문: {feedback['original']}
- 교정문: {feedback['corrected_text']}
- 만족도: {feedback['feedback']}

**분석 요구사항**:
1. 사용자가 왜 이 피드백을 남겼는지 추론
2. 교정문의 어떤 부분이 문제였는지 파악
3. 개선 방향 제시

JSON 형식으로 응답:
{{
  "issue": "문제점",
  "cause": "원인",
  "suggestion": "개선 방향"
}}
"""
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",  # Groq 최신 모델
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=500,
                    timeout=30.0  # ✅ 타임아웃 30초
                )
                
                result = response.choices[0].message.content
                
                # JSON 파싱
                try:
                    insight_data = json.loads(result)
                except:
                    # JSON이 아니면 텍스트 그대로 사용
                    insight_data = {
                        "issue": result[:100],
                        "cause": "파싱 실패",
                        "suggestion": result
                    }
                
                # 인사이트 객체 생성
                insight = {
                    'feedback_id': feedback.get('id'),
                    'user_id': feedback['user_id'],
                    'segment': segment,
                    'feature': feedback['selected_feature'],
                    'issue': insight_data.get('issue', ''),
                    'cause': insight_data.get('cause', ''),
                    'suggestion': insight_data.get('suggestion', ''),
                    'timestamp': datetime.now()
                }
                
                return insight
                
            except Exception as e:
                print(f"  ⚠️ Groq API 시도 {attempt + 1}/{max_retries} 실패: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 지수 백오프
                    print(f"  ⏳ {wait_time}초 대기 후 재시도...")
                    time.sleep(wait_time)
                else:
                    return None
        
        return None
    
    def export_for_collector(self, insights=None):
        """
        인사이트를 insights_queue에 저장
        
        Args:
            insights: 인사이트 리스트 (None이면 DB에서 조회)
        """
        if insights is None:
            # DB에서 미처리 인사이트 조회
            insights = self._get_pending_insights()
        
        if not insights:
            print("  ⏭️ 처리할 인사이트 없음")
            return
        
        # insights_queue 테이블에 저장
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        for insight in insights:
            cursor.execute("""
                INSERT INTO insights_queue 
                (feedback_id, user_id, segment, feature, issue, cause, suggestion, timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                insight['feedback_id'],
                insight['user_id'],
                insight['segment'],
                insight['feature'],
                insight['issue'],
                insight['cause'],
                insight['suggestion'],
                insight['timestamp']
            ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"  ✅ {len(insights)}개 인사이트 저장 완료")
    
    def _get_pending_insights(self):
        """DB에서 미처리 인사이트 조회"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        # ✅ BOOLEAN 타입으로 수정
        cursor.execute("""
            SELECT id, user_id, selected_feature, original, corrected_text, feedback
            FROM feedbacks
            WHERE processed = FALSE
            ORDER BY timestamp ASC
            LIMIT 10
        """)
        
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        
        feedbacks = []
        for row in rows:
            feedbacks.append({
                'id': row[0],
                'user_id': row[1],
                'selected_feature': row[2],
                'original': row[3],
                'corrected_text': row[4],
                'feedback': row[5]
            })
        
        return feedbacks
