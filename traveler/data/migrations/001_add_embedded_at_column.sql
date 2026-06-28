-- Migration 001: places 테이블에 embedded_at 컬럼 추가
-- 목적: Pinecone 임베딩 적재 여부를 추적하기 위함
--   - NULL: 아직 Pinecone에 임베딩되지 않은 장소 (신규/미처리)
--   - 타임스탬프(ISO 8601 문자열): 해당 시각에 임베딩이 완료된 장소
--
-- 사용 예: 미처리 장소만 골라서 임베딩 파이프라인에 넘기기
--   SELECT * FROM places WHERE embedded_at IS NULL;
--
-- 주의: 기존 noKidsZone, iconic_score 컬럼은 이미 존재하므로 이 마이그레이션에서 다루지 않음.

ALTER TABLE places ADD COLUMN embedded_at TEXT DEFAULT NULL;
