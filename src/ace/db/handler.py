# src/ace/db/handler.py

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
        print(f"⚠️ DB 초기화 실패: {e}")

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
    
    ⚠️ 임시: Groq API 문제로 모든 요청을 프론트엔드로 리다이렉트
    """
    
    print(f"\n{'='*60}")
    print(f"🎯 교정 요청: user={user_id}, feature={feature}")
    print(f"{'='*60}")
    
    has_feedback = has_user_feedback(user_id)
    
    if has_feedback:
        print(f"✅ 피드백 {has_feedback}개 있음")
    else:
        print(f"📝 피드백 없음")
    
    print("⚠️ 백엔드 스킵 → 프론트엔드 HuggingFace 사용")
    
    # ===== 모든 경우 프론트엔드 처리 (Groq 비활성화) =====
    return {
        'corrected': text,
        'method': 'backend_skip',
        'use_frontend': True,  # ✅ 프론트엔드 사용 플래그
        'rules_applied': 0,
        'confidence': 0.0,
        'has_feedback': has_feedback,
        'message': 'Use HuggingFace on frontend'
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
    """피드백 저장 (분석은 임시 비활성화)"""
    
    try:
        feedback_id = save_feedback(feedback_data)
        print(f"✅ 피드백 저장 (ID: {feedback_id})")
        print("⚠️ Groq API 문제로 분석 스킵")
        return True
            
    except Exception as e:
        print(f"❌ 처리 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return False
