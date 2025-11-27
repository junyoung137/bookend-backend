# db/handler.py - 피드백 남긴 옵션만 DB 저장/조회하기

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

# DB 설정 (feedback DB 정보 입력)
DB_CONFIG = settings.ace.db_config

def get_db_connection():
    """DB 연결 헬퍼 함수"""
    return psycopg2.connect(**DB_CONFIG)

def init_db():
    """DB 초기화 - feedback 테이블 생성 (최초 1회만 실행)"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # feedback 테이블 생성 (processed를 BOOLEAN으로 변경)
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

    # 인덱스 생성
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_user_id
        ON feedbacks(user_id);
    ''')

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_processed
        ON feedbacks(processed);
    ''')

    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_timestamp
        ON feedbacks(timestamp);
    ''')
    
    conn.commit()
    cursor.close()
    conn.close()

    print("✅ PostgreSQL 테이블 초기화 완료!")

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
    """DB에서 미처리
