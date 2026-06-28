-- Migration 002: trip_session 테이블 생성
-- 목적: 사용자와의 대화형 추천 세션을 저장
--   (한 번의 대화 = 하나의 session_id. 그 안에서 여러 턴이 오가며
--    날씨/위치 등 맥락이 누적되고, 추천된 장소들이 기록됨)
--
-- 설계 메모:
--   - context, recommended_places는 JSON 문자열로 저장
--     (SQLite는 JSON 타입이 별도로 없으므로 TEXT에 저장)
--   - status로 진행 중/종료된 세션을 구분해서, 나중에 "이어서 대화하기" 같은
--     기능을 만들 때 활용 가능
--   - user_id에 대한 FK 제약은 아직 user 테이블 스키마가 백엔드 쪽 책임
--     영역이라 일단 비워둠 (TEXT로만 저장, 추후 FK 연결 검토)

CREATE TABLE IF NOT EXISTS trip_session (
    session_id         TEXT PRIMARY KEY,             -- 세션 고유 ID (예: UUID)
    user_id            TEXT,                         -- 사용자 식별자 (추후 FK 연결 가능)
    created_at         TEXT NOT NULL,                -- 세션 시작 시각 (ISO 8601)
    updated_at         TEXT NOT NULL,                -- 마지막 턴 갱신 시각 (ISO 8601)
    status             TEXT NOT NULL DEFAULT 'active', -- 'active' | 'completed' | 'abandoned'
    context            TEXT,                         -- 누적 대화 맥락 (JSON: 날씨, 위치, 이전 턴 요약 등)
    recommended_places TEXT,                         -- 추천된 장소 contentid 목록 (JSON 배열)
    turn_count         INTEGER NOT NULL DEFAULT 0    -- 누적 대화 턴 수
);

-- 자주 조회될 패턴: 특정 사용자의 최근 세션 찾기
CREATE INDEX IF NOT EXISTS idx_trip_session_user_id ON trip_session(user_id);
CREATE INDEX IF NOT EXISTS idx_trip_session_status  ON trip_session(status);
