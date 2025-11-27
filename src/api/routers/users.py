"""
유저 관련 API 라우터
"""
from fastapi import APIRouter, HTTPException
import psycopg2
import os
from typing import Dict, Any

router = APIRouter(prefix="/api/users", tags=["users"])


def get_bookend_db_connection():
    """bookend_db 연결"""
    try:
        conn = psycopg2.connect(
            host=os.getenv("BOOKEND_DB_HOST", "localhost"),
            port=int(os.getenv("BOOKEND_DB_PORT", "5432")),
            database=os.getenv("BOOKEND_DB_NAME", "bookend_db"),
            user=os.getenv("BOOKEND_DB_USER", "lunaflow"),
            password=os.getenv("BOOKEND_DB_PASSWORD", "")
        )
        return conn
    except Exception as e:
        print(f"❌ DB 연결 실패: {e}")
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")


@router.get("/{user_id}/is-new")
async def check_if_new_user(user_id: str) -> Dict[str, Any]:
    """
    신규 유저인지 확인 (interaction_count <= 2)
    
    Returns:
        {
            "success": bool,
            "is_new_user": bool,
            "interaction_count": int
        }
    """
    try:
        conn = get_bookend_db_connection()
        cursor = conn.cursor()
        
        # TODO: C의 테이블 구조에 맞게 수정 필요!
        # 테이블 이름: users (확인 필요)
        # 컬럼 이름: user_id, interaction_count (확인 필요)
        cursor.execute("""
            SELECT interaction_count 
            FROM users 
            WHERE user_id = %s
        """, (user_id,))
        
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not row:
            # 유저 없으면 신규로 간주
            print(f"⚠️ 유저 없음: {user_id} → 신규로 처리")
            return {
                "success": True,
                "is_new_user": True,
                "interaction_count": 0
            }
        
        interaction_count = row[0]
        is_new_user = interaction_count <= 2
        
        print(f"✅ 유저 확인: {user_id}, 인터랙션={interaction_count}, 신규={is_new_user}")
        
        return {
            "success": True,
            "is_new_user": is_new_user,
            "interaction_count": interaction_count
        }
        
    except Exception as e:
        print(f"❌ 신규 유저 확인 실패: {e}")
        # 에러 시 안전하게 true 반환 (피드백 수집 우선)
        return {
            "success": False,
            "is_new_user": True,
            "interaction_count": 0,
            "error": str(e)
        }


@router.get("/list")
async def get_users():
    """실제 유저 목록 조회 (개발용)"""
    try:
        conn = get_bookend_db_connection()
        cursor = conn.cursor()
        
        # TODO: C의 테이블 구조에 맞게 수정
        cursor.execute("""
            SELECT user_id, username, interaction_count 
            FROM users