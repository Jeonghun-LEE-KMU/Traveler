-- Migration 003: places 테이블에 is_active 컬럼 추가
-- 목적: 축제/행사(typeId=15)의 기간 만료 여부를 표시하기 위함
--   - 1 (기본값): 활성 상태 (현재 유효/노출 대상)
--   - 0: 비활성 상태 (event_end_date가 지나 만료된 축제, RAG 추천에서 제외)
--
-- 완전 삭제 대신 비활성화 방식을 택한 이유:
--   추후 "작년 인기 축제" 같은 통계/분석 활용 가능성을 보존하기 위함
--
-- 사용 예: 활성 장소만 조회
--   SELECT * FROM places WHERE is_active = 1;
--
-- 적용 대상은 우선 축제(typeId=15)지만, 컬럼 자체는 전체 places에 적용
-- (다른 타입도 추후 폐업 등으로 비활성화 처리할 가능성을 고려)

ALTER TABLE places ADD COLUMN is_active INTEGER DEFAULT 1;

CREATE INDEX IF NOT EXISTS idx_places_is_active ON places(is_active);