# src/ace/db/handler.py (상단에 추가)

import psycopg2
from psycopg2.extras import RealDictCursor
import json
from datetime import datetime
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from config.settings import get_settings

settings = get_settings()
DB_CONFIG = settings.ace.db_config

def get_db_connection():
    """DB 연결 헬퍼 함수"""
    return psycopg2.connect(**DB_CONFIG)

def init_db():
    """DB 초기화 - feedback 테이블 생성"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # feedbacks 테이블 생성
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedbacks (
                id SERIAL PRIMARY KEY,
                user_id TEXT,
                original TEXT NOT NULL,
                selected_feature TEXT NOT NULL,
                corrected_text TEXT NOT NULL,
                feedback TEXT NOT NULL,
                context JSONB,
                timestamp TIMESTAMP NOT NULL,
                processed BOOLEAN DEFAULT FALSE,
                processed_at TIMESTAMP
            );
        ''')

        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON feedbacks(user_id);')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_processed ON feedbacks(processed);')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON feedbacks(timestamp);')
        
        # insights_queue 테이블 생성
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS insights_queue (
                id SERIAL PRIMARY KEY,
                feedback_id INTEGER REFERENCES feedbacks(id),
                user_id TEXT,
                segment TEXT,
                feature TEXT,
                issue TEXT,
                cause TEXT,
                suggestion TEXT,
                timestamp TIMESTAMP NOT NULL,
                processed BOOLEAN DEFAULT FALSE
            );
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_insights_processed ON insights_queue(processed);')
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print("✅ PostgreSQL 테이블 초기화 완료!")
        
    except Exception as e:
        print(f"⚠️ DB 초기화 실패 (이미 존재할 수 있음): {e}")

def has_user_feedback(user_id):
    """사용자가 피드백을 남긴 적 있는지 확인"""
    try:
        import time
        time.sleep(0.2)

        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT COUNT(*) FROM feedbacks WHERE user_id = %s",
            (user_id,)
        )
        
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        
        print(f"🔍 [{user_id}] 피드백 개수: {count}개")
        return count > 0
        
    except Exception as e:
        print(f"⚠️ DB 확인 실패: {e}")
        return False

def correct_with_personalization(
    user_id,
    text,
    feature,
    tone='normal',
    genre='informative',
    groq_api_key=None,
    rulebook_path=None,
):
    """
    개인화된 문장 교정 (통합 진입점)
    
    흐름:
    1. 피드백 없으면 → backend_skip 반환 (프론트엔드 처리)
    2. 피드백 있으면 → Groq 개인화 교정
    3. Groq 실패 → groq_failed 반환 (프론트엔드 폴백)
    """
    
    print(f"\n{'='*60}")
    print(f"🎯 교정 요청: user={user_id}, feature={feature}")
    print(f"{'='*60}")
    
    has_feedback = has_user_feedback(user_id)
    
    # ===== 1. 피드백 없음 → 백엔드 스킵 =====
    if not has_feedback:
        print(f"📝 피드백 없음 → 프론트엔드에서 HuggingFace 처리")
        return {
            'corrected': text,
            'method': 'backend_skip',
            'rules_applied': 0,
            'confidence': 0.0,
            'message': '프론트엔드에서 HuggingFace 사용 필요'
        }
    
    # ===== 2. 피드백 있음 → Groq 개인화 교정 =====
    print(f"✨ Groq 개인화 교정 실행 (피드백 기반)")
    
    try:
        from src.ace.evaluator.sentence_evaluator import Evaluator
        
        API_KEY_G = settings.ace.groq_api_key
        
        evaluator = Evaluator(
            api_key=API_KEY_G,
            rulebook_path=rulebook_path
        )
        
        result = evaluator.correct(
            original_text=text,
            feature=feature,
            tone=tone,
            genre=genre,
            min_confidence=0.5
        )
        
        return {
            'corrected': result['corrected'],
            'method': 'personalized',
            'rules_applied': result['rules_applied'],
            'confidence': result['confidence']
        }
        
    except Exception as e:
        print(f"⚠️ Groq 개인화 교정 실패: {e}")
        import traceback
        traceback.print_exc()
        
        # ===== 3. Groq 실패 → 프론트엔드 폴백 지시 =====
        return {
            'corrected': text,
            'method': 'groq_failed',
            'rules_applied': 0,
            'confidence': 0.0,
            'error': str(e),
            'message': '프론트엔드 HuggingFace 폴백 필요'
        }

def save_feedback(feedback_data):
    """DB에 피드백 저장"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO feedbacks 
        (user_id, original, selected_feature, corrected_text, 
         feedback, context, timestamp)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
    ''', (
        feedback_data['user_id'],
        feedback_data['original'],
        feedback_data['selected_feature'],
        feedback_data['corrected_text'],
        feedback_data['feedback'],
        json.dumps(feedback_data['context']),
        feedback_data['timestamp']
    ))
    
    feedback_id = cursor.fetchone()[0]
    conn.commit()
    cursor.close() 
    conn.close()
    
    return feedback_id

def get_new_feedbacks():
    """DB에서 미처리 피드백 조회"""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute('''
        SELECT id, user_id, original, selected_feature,
               corrected_text, feedback, context, timestamp
        FROM feedbacks
        WHERE processed = FALSE
        ORDER BY timestamp ASC
    ''')
    
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    
    feedbacks = [dict(row) for row in rows]
    print(f"📥 미처리 피드백 {len(feedbacks)}개 조회")
    return feedbacks

def mark_as_processed(feedback_ids):
    """피드백을 처리 완료 상태로 표시"""
    if not feedback_ids:
        return
        
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE feedbacks
        SET processed = TRUE,
            processed_at = %s
        WHERE id = ANY(%s)
    ''', (datetime.now(), feedback_ids))
    
    conn.commit()
    cursor.close()
    conn.close()
    
    print(f"✅ {len(feedback_ids)}개 피드백 처리 완료 표시")

def get_all_feedbacks():
    """DB에서 모든 피드백 조회 (테스트용)"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, user_id, feedback, processed, timestamp
        FROM feedbacks
        ORDER BY timestamp DESC
    ''')
    
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return rows

def save_feedback_and_process(feedback_data):
    """피드백 저장 + 즉시 ACE 파이프라인 실행"""
    
    try:
        feedback_id = save_feedback(feedback_data)
        print(f"✅ 피드백 저장 (ID: {feedback_id})")
        
        print("🔍 Groq 분석 시작...")
        
        API_KEY_G = settings.ace.groq_api_key
        
        from src.ace.analyzer.feedback_analyzer import Analyzer
        
        analyzer = Analyzer(api_key=API_KEY_G)
        
        feedback_data['id'] = feedback_id
        insights = analyzer.batch_analyze([feedback_data])
        
        if insights:
            print(f"💡 인사이트 생성: {len(insights)}개")
            analyzer.export_for_collector()

            from src.ace.collector.rule_collector import Collector
            collector = Collector(api_key=API_KEY_G)
            collector.process_insights()

            mark_as_processed([feedback_id])
            return True
        else:
            print("⚠️ 인사이트 생성 실패")
            return False
            
    except Exception as e:
        print(f"❌ 처리 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return False
