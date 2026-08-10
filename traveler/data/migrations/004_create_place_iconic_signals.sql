-- Iconic Score v2 component signal table.
--
-- places.iconic_score는 검색 runtime에서 읽는 최종 점수만 보관한다.
-- place_iconic_signals는 raw/normalized component score, 가중치, 버전, 상태를
-- 보관해 가중치 재조정과 결과 설명이 가능하도록 한다.

CREATE TABLE IF NOT EXISTS place_iconic_signals (
    content_id TEXT PRIMARY KEY,

    tourism100_raw REAL,
    datalab_search_raw REAL,
    tar_rlte_raw REAL,
    regional_demand_raw REAL,

    tourism100_score REAL,
    datalab_search_score REAL,
    tar_rlte_score REAL,
    regional_demand_score REAL,

    iconic_score_v2 REAL,
    weights_json TEXT,
    signal_version TEXT,

    missing_signal_count INTEGER DEFAULT 0,
    status TEXT,
    computed_at TEXT,

    FOREIGN KEY (content_id) REFERENCES places(content_id)
);

CREATE INDEX IF NOT EXISTS idx_place_iconic_signals_score
ON place_iconic_signals(iconic_score_v2);

CREATE INDEX IF NOT EXISTS idx_place_iconic_signals_status
ON place_iconic_signals(status);
