# db/handler.py - 피드백 DB 저장/조회 (BOOLEAN 타입 수정)

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
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # ✅ processed를 BOOLEAN으로 변경
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
    """DB에서 미처리 피드백 조회"""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    # ✅ BOOLEAN 타입으로 비교
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
    
    # ✅ BOOLEAN 타입으로 업데이트
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

def has_user_feedback(user_id):
    """사용자가 피드백을 남긴 적 있는지 확인"""
    try:
        import time
        time.sleep(0.2)   # DB 커밋 완료 대기

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
    1. 피드백 있으면 → Groq 개인화 교정 (백엔드)
    2. 피드백 없으면 → HuggingFace 기본 교정 (프론트엔드에서 이미 처리됨)
    """
    
    print(f"\n{'='*60}")
    print(f"🎯 교정 요청: user={user_id}, feature={feature}")
    print(f"{'='*60}")
    
    has_feedback = has_user_feedback(user_id)
    
    if has_feedback:
        # ===== Groq 개인화 교정 (백엔드) =====
        from src.ace.evaluator.sentence_evaluator import Evaluator
        print(f"✨ Groq 개인화 교정 실행 (피드백 기반)")
        
        try:
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
            print(f"   → HuggingFace 기본 교정으로 폴백")
            
            # ✅ Groq 실패 시 HuggingFace 폴백
            from src.ace.services.basic_llm import correct_sentence
            
            result = correct_sentence(
                text=text,
                feature=feature.lower().replace(' ', '_'),
                tone=tone,
                genre=genre,
                hf_api_key=settings.ace.huggingface_api_key
            )
            
            return {
                'corrected': result.get('corrected', text),
                'method': 'default_fallback',
                'rules_applied': 0,
                'confidence': 0.0,
                'error': str(e)
            }
    
    else:
        # ===== HuggingFace 기본 교정 (프론트엔드) =====
        print(f"📝 피드백 없음 → 프론트엔드에서 HuggingFace 처리")
        
        # 프론트엔드에서 이미 처리되므로 원문 그대로 반환
        return {
            'corrected': text,
            'method': 'frontend_huggingface',
            'rules_applied': 0,
            'confidence': 0.0,
            'message': '프론트엔드에서 HuggingFace API 사용'
        }

def save_feedback_and_process(feedback_data):
    """
    피드백 저장 + 즉시 ACE 파이프라인 실행
    
    Returns:
        bool: 성공 여부
    """
    
    try:
        # 1. DB 저장
        feedback_id = save_feedback(feedback_data)
        print(f"✅ 피드백 저장 (ID: {feedback_id})")
        
        # 2. 즉시 분석 실행
        print("🔍 Groq 분석 시작...")
        
        API_KEY_G = settings.ace.groq_api_key
        
        from src.ace.analyzer.feedback_analyzer import Analyzer
        
        analyzer = Analyzer(api_key=API_KEY_G)
        
        feedback_data['id'] = feedback_id
        insights = analyzer.batch_analyze([feedback_data])
        
        if insights:
            print(f"💡 인사이트 생성: {len(insights)}개")

            # 3. 인사이트를 insights_queue에 저장
            analyzer.export_for_collector()

            # 4. 규칙 생성
            from src.ace.collector.rule_collector import Collector
            collector = Collector(api_key=API_KEY_G)
            collector.process_insights()

            # 5. 처리 완료 표시
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
