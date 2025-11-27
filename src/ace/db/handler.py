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
    
    # feedback 테이블 생성
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedbacks (
            id SERIAL PRIMARY KEY,
            user_id TEXT,
            original TEXT NOT NULL,
            selected_feature TEXT NOT NULL,
            corrected_text TEXT NOT NULL,
            feedback TEXT NOT NULL,   -- '만족' or '불만족'
            context JSONB,            -- JSON
            timestamp TIMESTAMP NOT NULL,
            processed INTEGER DEFAULT 0,
            processed_at TIMESTAMP
        );
    ''')

    # 인덱스 생성 (검색 성능 최적화)
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
    
    feedback_id = cursor.fetchone()[0]  # RETURNING으로 받기

    conn.commit()
    cursor.close() 
    conn.close()
    
    return feedback_id

def get_new_feedbacks():
    """DB에서 미처리 피드백 조회
        
    Returns:
        list: 미처리 피드백 리스트 (각 피드백은 딕셔너리)
    """
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)   # 딕셔너리로 변환
    
    cursor.execute('''
        SELECT id, user_id, original, selected_feature,
               corrected_text, feedback, context, timestamp
        FROM feedbacks
        WHERE processed = 0    -- 미처리 피드백인 경우만 조회
        ORDER BY timestamp ASC -- 오래된 것부터 조회
    ''')
    
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    
    # 딕셔너리 리스트로 변환
    feedbacks = []
    for row in rows:
        feedback = dict(row)
        feedbacks.append(feedback)
    
    print(f"📥 미처리 피드백 {len(feedbacks)}개 조회")
    return feedbacks

def mark_as_processed(feedback_ids):
    """
    피드백을 처리 완료 상태로 표시 (DB 업데이트)
    
    Args:
        feedback_ids: 처리 완료한 피드백 ID 리스트
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 배치 업데이트
    cursor.execute('''
        UPDATE feedbacks
            SET processed = 1,    -- 처리완료로 변경
                processed_at = %s -- 처리 시각
            WHERE id = ANY(%s)    -- 여러 개 한번에 처리
        ''', (datetime.now(), feedback_ids))
    
    conn.commit()
    cursor.close()
    conn.close()
    
    print(f"✅ {len(feedback_ids)}개 피드백 처리 완료 표시")

def get_all_feedbacks():
    """
    DB에서 모든 피드백 조회 (테스트용)
    
    Returns:
        list: 모든 피드백 리스트
    """
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
    """
    사용자가 피드백을 남긴 적 있는지 확인 
    -> 피드백 1회 이상부터 더 개인화된 문장 제공
    
    Args:
        user_id: 사용자 ID
    
    Returns:
        bool: 피드백 존재 여부
    """
    # 실제 DB 확인
    try:
        import time
        time.sleep(0.2)   # 0.2초 대기 (DB 커밋 완료 대기)

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
    
    웹사이트에서는 이 함수만 호출하면 됨
    
    흐름:
    1. 사용자 피드백 여부 확인
    2. 피드백 있으면 → 평가자 (개인화)
    3. 피드백 없으면 → 기본 LLM (웹사이트 내 기존 반환 함수)
    
    Args:
        user_id: 사용자 ID
        text: 원문
        feature: 기능 (Paraphrase, Tone Adjust, Expand, Compress)
        tone: 톤 (formal, normal, common, terminal_word)
        genre: 장르 (informative, narrative, descriptive, dialogue)
        groq_api_key: Groq API 키 (None이면 파일에서 읽기)
        rulebook_path: 룰북 경로
    
    Returns:
        dict: {
            'corrected': 교정된 문장,
            'method': 'personalized' or 'default',
            'rules_applied': 적용된 규칙 수 (개인화일 때만),
            'confidence': 평균 신뢰도 (개인화일 때만)
        }
    """
    
    print(f"\n{'='*60}")
    print(f"🎯 교정 요청: user={user_id}, feature={feature}")
    print(f"{'='*60}")
    
    # 1. 피드백 여부 확인
    has_feedback = has_user_feedback(user_id)
    
    if has_feedback:
        # ===== 개인화 교정 (평가자 사용) =====
        from src.ace.evaluator.sentence_evaluator import Evaluator
        print(f"✨ 개인화 교정 실행 (피드백 기반)")
        
        try:
            # API 키 로드
            API_KEY_G = settings.ace.groq_api_key

            # 평가자 실행
            evaluator = Evaluator(
                api_key=API_KEY_G,
                rulebook_path=rulebook_path
            )
            
            # 교정 실행
            result = evaluator.correct(
                original_text=text,
                feature=feature,
                tone=tone,
                genre=genre,
                min_confidence=0.5
            )
            
            # 결과 반환
            return {
                'corrected': result['corrected'],
                'method': 'personalized',
                'rules_applied': result['rules_applied'],
                'confidence': result['confidence']
            }
            
        except Exception as e:
            print(f"⚠️  개인화 교정 실패: {e}")
            print(f"   → 기본 교정으로 폴백")
            
            # 실패 시 기본 교정으로 폴백
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
        # ===== 기본 교정 (웹사이트 내 기존 LLM) =====
        from src.ace.services.basic_llm import correct_sentence
        print(f"📝 기본 교정 실행 (피드백 없음)")
        
        if correct_sentence:
            result = correct_sentence(
                text=text,
                feature=feature.lower().replace(' ', '_'),
                tone=tone,
                genre=genre,
                hf_api_key=settings.ace.huggingface_api_key
            )

        else:
            # 기본 함수가 없으면 원문 그대로
            print("⚠️  기본 교정 함수 없음 → 원문 반환")
            result = text

        return result

def save_feedback_and_process(feedback_data):
    """
    피드백 저장 + 즉시 ACE 파이프라인 실행
    
    웹사이트에서 피드백 받으면 즉시 호출

    Args:
        feedback_data: 피드백 데이터
    
    Returns:
        bool: 성공 여부
    """
    
    # 1. DB 저장
    feedback_id = save_feedback(feedback_data)
    print(f"✅ 피드백 저장 (ID: {feedback_id})")
    
    # 2. 즉시 분석 실행 (동기)
    try:
        print("🔍 즉시 분석 시작...")
        
        # API 키 로드
        API_KEY_G = settings.ace.groq_api_key
        
        # 분석가 실행 (import)
        from src.ace.analyzer.feedback_analyzer import Analyzer
        
        analyzer = Analyzer(api_key=API_KEY_G)
        
        # 피드백 분석 (ID 포함)
        feedback_data['id'] = feedback_id
        insights = analyzer.batch_analyze([feedback_data])
        
        if insights:
            print(f"💡 인사이트 생성: {len(insights)}개")

            # 3. 분석한 인사이트를 insights_queue에 저장
            analyzer.export_for_collector()

            # 4. 인사이트 파일을 읽어 규칙 생성
            from src.ace.collector.rule_collector import Collector
            collector = Collector(api_key=API_KEY_G)
            collector.process_insights()

            # 5. 처리 완료 표시
            mark_as_processed([feedback_id])
            return True
            
    except Exception as e:
        print(f"❌ 처리 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return False