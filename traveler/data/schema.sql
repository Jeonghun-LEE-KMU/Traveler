-- ============================================================
-- places 테이블 스키마
-- 프로젝트: LLM-RAG 트래블 큐레이터
-- 작성자: gayoung
-- Sprint 1 | 2026-05-22
-- ============================================================

CREATE TABLE IF NOT EXISTS places (

    -- ── 공통 (detailCommon2) ──────────────────────────────────
    content_id          TEXT PRIMARY KEY,   -- 장소 고유 ID
    content_type_id     TEXT NOT NULL,      -- 타입 (12=관광지, 14=문화시설, 15=축제, 25=여행코스, 28=레포츠, 38=쇼핑, 39=음식점)
    name                TEXT NOT NULL,      -- 장소명
    address             TEXT,               -- 전체 주소
    area_code           TEXT,               -- 도 단위 (TourAPI areaCode, 예: 1=서울, 6=부산)
    sigungu_code        TEXT,               -- 시/군/구 단위 (TourAPI sigunguCode, 예: 680=강남구)
    lat                 REAL,               -- 위도 (mapy)
    lng                 REAL,               -- 경도 (mapx)
    overview            TEXT,               -- 소개글 (Pinecone 임베딩용)
    image_url           TEXT,               -- 대표 이미지 URL
    homepage            TEXT,               -- 공식 홈페이지
    tel                 TEXT,               -- 전화번호
    category            TEXT,               -- 분류 코드 (lclsSystm1)
    modified_time       TEXT,               -- 데이터 수정일
    iconic_score        REAL DEFAULT 0.0,   -- 인기도 점수 (readcount 정규화, 0.0 ~ 1.0)

    -- ── 관광지 (typeId=12) ───────────────────────────────────
    parking             TEXT,               -- 주차 가능 여부
    rest_date           TEXT,               -- 휴무일
    use_time            TEXT,               -- 운영시간
    baby_carriage       TEXT,               -- 유모차 대여

    -- ── 문화시설 (typeId=14) ─────────────────────────────────
    use_fee             TEXT,               -- 입장료
    use_time_culture    TEXT,               -- 운영시간
    rest_date_culture   TEXT,               -- 휴무일
    parking_culture     TEXT,               -- 주차

    -- ── 축제/행사 (typeId=15) ────────────────────────────────
    event_start_date    TEXT,               -- 행사 시작일
    event_end_date      TEXT,               -- 행사 종료일
    event_place         TEXT,               -- 행사 장소
    play_time           TEXT,               -- 공연시간
    use_time_festival   TEXT,               -- 이용요금
    program             TEXT,               -- 프로그램 내용

    -- ── 여행코스 (typeId=25) ─────────────────────────────────
    schedule            TEXT,               -- 코스 일정
    theme               TEXT,               -- 테마
    distance            TEXT,               -- 거리
    take_time           TEXT,               -- 소요시간

    -- ── 레포츠 (typeId=28) ───────────────────────────────────
    use_time_leports    TEXT,               -- 운영시간
    rest_date_leports   TEXT,               -- 휴무일
    parking_leports     TEXT,               -- 주차
    reservation         TEXT,               -- 예약 필요 여부
    exp_age_range       TEXT,               -- 체험 가능 연령

    -- ── 쇼핑 (typeId=38) ─────────────────────────────────────
    open_time           TEXT,               -- 운영시간
    sale_item           TEXT,               -- 판매품목
    rest_date_shopping  TEXT,               -- 휴무일
    parking_shopping    TEXT,               -- 주차
    restroom            TEXT,               -- 화장실

    -- ── 음식점 (typeId=39) ───────────────────────────────────
    first_menu          TEXT,               -- 대표메뉴
    treat_menu          TEXT,               -- 전체메뉴
    open_time_food      TEXT,               -- 운영시간
    rest_date_food      TEXT,               -- 휴무일
    parking_food        TEXT,               -- 주차
    kids_facility       TEXT,               -- 키즈시설

    -- ── 반려동물 (detailPetTour2) ────────────────────────────
    is_pet_allowed      INTEGER DEFAULT 0,  -- 반려동물 동반 가능 여부 (0/1)
    pet_acmpy_type      TEXT,               -- 동반 가능 유형
    pet_condition       TEXT,               -- 동반 조건 (목줄 착용 등)
    pet_info            TEXT,               -- 기타 안내

    -- ── 노키즈존 ─────────────────────────────────────────────
    is_nokidszone       INTEGER DEFAULT 0   -- 노키즈존 여부 (0/1)
);

-- ── 인덱스 ────────────────────────────────────────────────────
-- 지역 필터링용
CREATE INDEX IF NOT EXISTS idx_area        ON places(area_code);
CREATE INDEX IF NOT EXISTS idx_sigungu     ON places(area_code, sigungu_code);

-- 타입 필터링용
CREATE INDEX IF NOT EXISTS idx_type        ON places(content_type_id);

-- RRF Re-ranking용 인기도 정렬
CREATE INDEX IF NOT EXISTS idx_iconic      ON places(iconic_score DESC);

-- 반려동물 / 노키즈존 필터링용
CREATE INDEX IF NOT EXISTS idx_pet         ON places(is_pet_allowed);
CREATE INDEX IF NOT EXISTS idx_nokids      ON places(is_nokidszone);
