# ACE Database 설정 가이드

## 1. PostgreSQL 유저 생성 (없으면)
```bash
psql -U postgres

CREATE USER ace_admin WITH PASSWORD 'ace-project1';
ALTER USER ace_admin CREATEDB;
\q
```

## 2. DB 생성
```bash
createdb -U ace_admin bookend_ace
```

## 3. 스키마 복원
```bash
psql -U ace_admin -h localhost bookend_ace < bookend_ace_schema.sql
```

## 4. 확인
```bash
psql -U ace_admin bookend_ace

\dt                              # feedbacks 테이블 있는지
\d feedbacks                     # 테이블 구조 확인
SELECT COUNT(*) FROM feedbacks;  # 0개여야 정상!
\q
```

## 5. 환경변수 설정

`.env` 파일 생성:
```env
ACE_DB_HOST=localhost
ACE_DB_PORT=5432
ACE_DB_NAME=bookend_ace
ACE_DB_USER=ace_admin
ACE_DB_PASSWORD=ace-project1
```

## 6. 백엔드 실행

## 7. 프론트엔드 실행