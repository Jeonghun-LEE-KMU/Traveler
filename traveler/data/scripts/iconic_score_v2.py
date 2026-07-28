"""
Iconic Score ETL v2.

Sprint 3 목표:
  4개 신호를 결합해 장소별 iconic_score_v2 preview/snapshot을 만든다.

신호:
  1. tourism100_score      : 한국관광 100선 exact 매칭 여부
  2. datalab_search_score  : Naver DataLab 검색량/관심도
  3. tar_rlte_score        : TarRlteTar 연관 빈도
  4. regional_demand_score : 지역별 관광 자원 수요

주의:
  이 스크립트는 기본적으로 places.db를 수정하지 않는다.
  active 장소 snapshot, raw signal CSV, normalized preview/snapshot만 저장한다.
  DB component table 저장은 --write-signals-db를 명시했을 때만 수행한다.
  places.iconic_score 업데이트와 Pinecone metadata sync는 preview 검토 후 별도 단계에서 수행한다.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from iconic_score import (
    DEFAULT_TOURISM_100_CSV,
    load_tourism_100_entries,
    normalize_name,
    split_tourism_keywords,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR / "data"
DEFAULT_DB_PATH = DATA_DIR / "places.db"

DEFAULT_WEIGHTS = {
    "tourism100_score": 0.25,
    "datalab_search_score": 0.35,
    "tar_rlte_score": 0.30,
    "regional_demand_score": 0.10,
}

REQUIRED_RAW_SIGNALS = {
    "tourism100_raw",
    "datalab_search_raw",
    "tar_rlte_raw",
    "regional_demand_raw",
}


@dataclass(frozen=True)
class ActivePlace:
    content_id: str
    name: str
    address: str
    area_code: str
    sigungu_code: str
    content_type_id: str


def today_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row[1]) for row in rows}


def load_active_places(
    db_path: Path,
    *,
    limit: int | None = None,
    content_types: set[str] | None = None,
) -> list[ActivePlace]:
    if not db_path.exists():
        raise FileNotFoundError(f"places.db를 찾을 수 없습니다: {db_path}")

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        columns = table_columns(conn, "places")
        required = {
            "content_id",
            "name",
            "address",
            "area_code",
            "sigungu_code",
            "content_type_id",
            "is_active",
        }
        missing = required - columns
        if missing:
            raise RuntimeError(f"places 테이블 필수 컬럼 누락: {sorted(missing)}")

        where = ["is_active = 1", "name IS NOT NULL", "trim(name) != ''"]
        params: list[str | int] = []
        if content_types:
            placeholders = ", ".join("?" for _ in content_types)
            where.append(f"CAST(content_type_id AS TEXT) IN ({placeholders})")
            params.extend(sorted(content_types))

        sql = f"""
            SELECT
                CAST(content_id AS TEXT) AS content_id,
                COALESCE(name, '') AS name,
                COALESCE(address, '') AS address,
                COALESCE(CAST(area_code AS TEXT), '') AS area_code,
                COALESCE(CAST(sigungu_code AS TEXT), '') AS sigungu_code,
                COALESCE(CAST(content_type_id AS TEXT), '') AS content_type_id
            FROM places
            WHERE {' AND '.join(where)}
            ORDER BY CAST(content_id AS TEXT)
        """
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)

        rows = conn.execute(sql, params).fetchall()

    return [
        ActivePlace(
            content_id=str(row["content_id"]),
            name=str(row["name"]),
            address=str(row["address"]),
            area_code=str(row["area_code"]),
            sigungu_code=str(row["sigungu_code"]),
            content_type_id=str(row["content_type_id"]),
        )
        for row in rows
    ]


def write_active_snapshot(path: Path, places: list[ActivePlace]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "place_id",
                "title",
                "area_code",
                "sigungu_code",
                "content_type_id",
                "addr1",
                "is_active",
            ],
        )
        writer.writeheader()
        for place in places:
            writer.writerow(
                {
                    "place_id": place.content_id,
                    "title": place.name,
                    "area_code": place.area_code,
                    "sigungu_code": place.sigungu_code,
                    "content_type_id": place.content_type_id,
                    "addr1": place.address,
                    "is_active": 1,
                }
            )


def build_tourism100_exact_map(
    places: list[ActivePlace],
    tourism_csv_path: Path,
) -> dict[str, int]:
    entries = load_tourism_100_entries(tourism_csv_path)
    tourism_keywords_by_region: list[tuple[str, set[str]]] = []
    for entry in entries:
        keyword_norms = {
            normalize_name(keyword)
            for keyword in split_tourism_keywords(entry.place_name)
        }
        tourism_keywords_by_region.append((entry.region, keyword_norms))

    result: dict[str, int] = {}
    for place in places:
        place_norm = normalize_name(place.name)
        is_exact = any(
            place_norm in keyword_norms and region in place.address
            for region, keyword_norms in tourism_keywords_by_region
        )
        result[place.content_id] = 1 if is_exact else 0
    return result


def read_signal_csv(
    path: Path | None,
    *,
    value_columns: list[str],
) -> dict[str, float]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"signal CSV를 찾을 수 없습니다: {path}")

    values: dict[str, float] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path}에 header가 없습니다.")

        id_column = "content_id" if "content_id" in reader.fieldnames else "place_id"
        if id_column not in reader.fieldnames:
            raise ValueError(f"{path}에는 content_id 또는 place_id 컬럼이 필요합니다.")

        selected_value_column = next(
            (column for column in value_columns if column in reader.fieldnames),
            None,
        )
        if selected_value_column is None:
            raise ValueError(
                f"{path}에는 다음 중 하나의 값 컬럼이 필요합니다: {value_columns}"
            )

        for row in reader:
            content_id = str(row.get(id_column) or "").strip()
            raw_value = str(row.get(selected_value_column) or "").strip()
            if not content_id or raw_value == "":
                continue
            try:
                values[content_id] = float(raw_value)
            except ValueError:
                continue
    return values


def read_place_filter_csv(path: Path | None) -> set[str]:
    """
    pilot 계산 대상을 CSV의 content_id/place_id 집합으로 제한한다.

    WHY:
      regional_demand_raw처럼 특정 지역 pilot CSV만 넣고 전체 DB를 계산하면,
      CSV에 없는 전국 장소들이 missing_required 상태로 섞여 preview 품질을
      판단하기 어렵다. 이 필터는 "이번 실험 대상 장소"를 명시적으로 고정한다.
    """
    if path is None:
        return set()
    if not path.exists():
        raise FileNotFoundError(f"places filter CSV를 찾을 수 없습니다: {path}")

    content_ids: set[str] = set()
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path}에 header가 없습니다.")

        id_column = "content_id" if "content_id" in reader.fieldnames else "place_id"
        if id_column not in reader.fieldnames:
            raise ValueError(f"{path}에는 content_id 또는 place_id 컬럼이 필요합니다.")

        for row in reader:
            content_id = str(row.get(id_column) or "").strip()
            if content_id:
                content_ids.add(content_id)

    if not content_ids:
        raise ValueError(f"{path}에서 content_id/place_id를 찾지 못했습니다.")
    return content_ids


def read_tar_signal_csv(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"TarRlteTar CSV를 찾을 수 없습니다: {path}")

    values: dict[str, float] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path}에 header가 없습니다.")

        id_column = "content_id" if "content_id" in reader.fieldnames else "place_id"
        if id_column not in reader.fieldnames:
            raise ValueError(f"{path}에는 content_id 또는 place_id 컬럼이 필요합니다.")

        has_direct_raw = "tar_rlte_raw" in reader.fieldnames
        has_counts = {
            "inbound_count",
            "outbound_count",
        }.issubset(set(reader.fieldnames))
        if not has_direct_raw and not has_counts:
            raise ValueError(
                f"{path}에는 tar_rlte_raw 또는 inbound_count/outbound_count 컬럼이 필요합니다."
            )

        for row in reader:
            content_id = str(row.get(id_column) or "").strip()
            if not content_id:
                continue
            try:
                if has_direct_raw:
                    raw = float(str(row.get("tar_rlte_raw") or "0").strip() or 0)
                else:
                    inbound = float(str(row.get("inbound_count") or "0").strip() or 0)
                    outbound = float(str(row.get("outbound_count") or "0").strip() or 0)
                    raw = inbound + 0.5 * outbound
            except ValueError:
                continue
            values[content_id] = raw
    return values


def normalize_count_signal(raw_values: dict[str, float]) -> dict[str, float]:
    if not raw_values:
        return {}

    logged = {
        key: math.log1p(max(float(value), 0.0))
        for key, value in raw_values.items()
    }
    minimum = min(logged.values())
    maximum = max(logged.values())
    if math.isclose(minimum, maximum):
        normalized_value = 0.0 if math.isclose(maximum, 0.0) else 1.0
        return {key: normalized_value for key in logged}

    return {
        key: (value - minimum) / (maximum - minimum)
        for key, value in logged.items()
    }


def score_status(raw_row: dict[str, float | None]) -> tuple[int, str]:
    missing_required = [
        signal
        for signal in sorted(REQUIRED_RAW_SIGNALS)
        if raw_row.get(signal) is None
    ]
    missing_count = len(missing_required)
    if missing_required:
        return missing_count, "missing_required:" + "|".join(missing_required)
    return missing_count, "ok"


def compute_weighted_score(score_row: dict[str, float | None]) -> float | None:
    """
    사용 가능한 신호의 weight만 재정규화한다.
    필수 신호가 비어 있는 pilot 단계에서도 preview를 만들 수 있게 하기 위함이다.
    """
    weighted_sum = 0.0
    available_weight = 0.0
    for signal, weight in DEFAULT_WEIGHTS.items():
        value = score_row.get(signal)
        if value is None:
            continue
        weighted_sum += weight * float(value)
        available_weight += weight

    if math.isclose(available_weight, 0.0):
        return None
    return round(weighted_sum / available_weight, 4)


def build_raw_rows(
    places: list[ActivePlace],
    *,
    tourism_raw: dict[str, int],
    datalab_raw: dict[str, float],
    tar_raw: dict[str, float],
    regional_demand_raw: dict[str, float],
) -> list[dict]:
    collected_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict] = []
    for place in places:
        row = {
            "place_id": place.content_id,
            "title": place.name,
            "area_code": place.area_code,
            "sigungu_code": place.sigungu_code,
            "content_type_id": place.content_type_id,
            "tourism100_raw": tourism_raw.get(place.content_id),
            "datalab_search_raw": datalab_raw.get(place.content_id),
            "tar_rlte_raw": tar_raw.get(place.content_id),
            "regional_demand_raw": regional_demand_raw.get(place.content_id),
            "collected_at": collected_at,
        }
        missing_count, status = score_status(row)
        row["missing_signal_count"] = missing_count
        row["status"] = status
        row["error_message"] = ""
        rows.append(row)
    return rows


def build_preview_rows(raw_rows: list[dict]) -> list[dict]:
    raw_by_signal = {
        "datalab_search_raw": {},
        "tar_rlte_raw": {},
        "regional_demand_raw": {},
    }
    for row in raw_rows:
        place_id = str(row["place_id"])
        for signal in raw_by_signal:
            value = row.get(signal)
            if value is None:
                continue
            raw_by_signal[signal][place_id] = float(value)

    normalized = {
        signal: normalize_count_signal(values)
        for signal, values in raw_by_signal.items()
    }

    rows: list[dict] = []
    for raw in raw_rows:
        place_id = str(raw["place_id"])
        score_row: dict[str, float | None] = {
            "tourism100_score": (
                float(raw["tourism100_raw"])
                if raw.get("tourism100_raw") is not None
                else None
            ),
            "datalab_search_score": normalized["datalab_search_raw"].get(place_id),
            "tar_rlte_score": normalized["tar_rlte_raw"].get(place_id),
            "regional_demand_score": normalized["regional_demand_raw"].get(place_id),
        }
        iconic_score_v2 = compute_weighted_score(score_row)
        rows.append(
            {
                "place_id": place_id,
                "title": raw["title"],
                "area_code": raw["area_code"],
                "sigungu_code": raw["sigungu_code"],
                "content_type_id": raw["content_type_id"],
                **score_row,
                "missing_signal_count": raw["missing_signal_count"],
                "iconic_score_v2": iconic_score_v2,
                "status": raw["status"],
            }
        )
    rows.sort(
        key=lambda row: (
            row["iconic_score_v2"] is None,
            -(row["iconic_score_v2"] or 0.0),
            row["title"],
        )
    )
    return rows


def write_dict_rows(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ensure_signal_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
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
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_place_iconic_signals_score
        ON place_iconic_signals(iconic_score_v2)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_place_iconic_signals_status
        ON place_iconic_signals(status)
        """
    )


def write_signals_to_db(
    db_path: Path,
    raw_rows: list[dict],
    preview_rows: list[dict],
    *,
    signal_version: str,
) -> None:
    raw_by_id = {str(row["place_id"]): row for row in raw_rows}
    computed_at = datetime.now(timezone.utc).isoformat()
    weights_json = json.dumps(DEFAULT_WEIGHTS, ensure_ascii=False, sort_keys=True)

    with sqlite3.connect(db_path) as conn:
        ensure_signal_table(conn)
        conn.executemany(
            """
            INSERT INTO place_iconic_signals (
                content_id,
                tourism100_raw,
                datalab_search_raw,
                tar_rlte_raw,
                regional_demand_raw,
                tourism100_score,
                datalab_search_score,
                tar_rlte_score,
                regional_demand_score,
                iconic_score_v2,
                weights_json,
                signal_version,
                missing_signal_count,
                status,
                computed_at
            )
            VALUES (
                :content_id,
                :tourism100_raw,
                :datalab_search_raw,
                :tar_rlte_raw,
                :regional_demand_raw,
                :tourism100_score,
                :datalab_search_score,
                :tar_rlte_score,
                :regional_demand_score,
                :iconic_score_v2,
                :weights_json,
                :signal_version,
                :missing_signal_count,
                :status,
                :computed_at
            )
            ON CONFLICT(content_id) DO UPDATE SET
                tourism100_raw = excluded.tourism100_raw,
                datalab_search_raw = excluded.datalab_search_raw,
                tar_rlte_raw = excluded.tar_rlte_raw,
                regional_demand_raw = excluded.regional_demand_raw,
                tourism100_score = excluded.tourism100_score,
                datalab_search_score = excluded.datalab_search_score,
                tar_rlte_score = excluded.tar_rlte_score,
                regional_demand_score = excluded.regional_demand_score,
                iconic_score_v2 = excluded.iconic_score_v2,
                weights_json = excluded.weights_json,
                signal_version = excluded.signal_version,
                missing_signal_count = excluded.missing_signal_count,
                status = excluded.status,
                computed_at = excluded.computed_at
            """,
            [
                {
                    "content_id": str(row["place_id"]),
                    "tourism100_raw": raw_by_id[str(row["place_id"])].get(
                        "tourism100_raw"
                    ),
                    "datalab_search_raw": raw_by_id[str(row["place_id"])].get(
                        "datalab_search_raw"
                    ),
                    "tar_rlte_raw": raw_by_id[str(row["place_id"])].get(
                        "tar_rlte_raw"
                    ),
                    "regional_demand_raw": raw_by_id[str(row["place_id"])].get(
                        "regional_demand_raw"
                    ),
                    "tourism100_score": row.get("tourism100_score"),
                    "datalab_search_score": row.get("datalab_search_score"),
                    "tar_rlte_score": row.get("tar_rlte_score"),
                    "regional_demand_score": row.get("regional_demand_score"),
                    "iconic_score_v2": row.get("iconic_score_v2"),
                    "weights_json": weights_json,
                    "signal_version": signal_version,
                    "missing_signal_count": row.get("missing_signal_count"),
                    "status": row.get("status"),
                    "computed_at": computed_at,
                }
                for row in preview_rows
            ],
        )
        conn.commit()


def print_summary(raw_rows: list[dict], preview_rows: list[dict]) -> None:
    total = len(raw_rows)
    status_counts: dict[str, int] = {}
    for row in raw_rows:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1

    scored = [
        float(row["iconic_score_v2"])
        for row in preview_rows
        if row.get("iconic_score_v2") is not None
    ]
    print("[요약]")
    print(f"- active 장소 수: {total}")
    print(f"- iconic_score_v2 계산 가능: {len(scored)}")
    print(f"- status: {status_counts}")
    if scored:
        print(
            "- iconic_score_v2 분포: "
            f"min={min(scored):.4f}, "
            f"max={max(scored):.4f}, "
            f"mean={sum(scored) / len(scored):.4f}"
        )
        print("[Top 10]")
        for row in preview_rows[:10]:
            print(
                f"  {row['iconic_score_v2']} | "
                f"{row['title']}({row['place_id']}) | "
                f"type={row['content_type_id']} | status={row['status']}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iconic Score v2 raw/preview snapshot 생성"
    )
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="places.db 경로")
    parser.add_argument(
        "--tourism-csv",
        default=DEFAULT_TOURISM_100_CSV,
        help="한국관광 100선 CSV 경로",
    )
    parser.add_argument(
        "--datalab-raw",
        default=None,
        help="DataLab raw CSV. content_id/place_id + datalab_search_raw 컬럼",
    )
    parser.add_argument(
        "--tar-raw",
        default=None,
        help="TarRlteTar raw CSV. content_id/place_id + tar_rlte_raw 또는 inbound_count/outbound_count",
    )
    parser.add_argument(
        "--regional-demand-raw",
        default=None,
        help="지역 관광 자원 수요 raw CSV. content_id/place_id + regional_demand_raw 컬럼",
    )
    parser.add_argument(
        "--places-filter-csv",
        default=None,
        help=(
            "계산 대상 장소를 이 CSV의 content_id/place_id로 제한. "
            "pilot 실행 시 regional_demand_raw CSV 등을 지정"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=DATA_DIR,
        help="snapshot/preview CSV 저장 디렉터리",
    )
    parser.add_argument("--limit", type=int, default=None, help="pilot 처리 건수")
    parser.add_argument(
        "--content-types",
        nargs="+",
        default=None,
        help="처리할 content_type_id 제한. 예: --content-types 12 14",
    )
    parser.add_argument(
        "--signal-version",
        default="iconic_v2_4signal_202607",
        help="DB 저장 시 사용할 signal version",
    )
    parser.add_argument(
        "--write-signals-db",
        action="store_true",
        help="place_iconic_signals 테이블에 raw/component/final score를 UPSERT",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_path = Path(args.db)
    tourism_csv_path = Path(args.tourism_csv)
    out_dir = Path(args.out_dir)
    tag = today_tag()

    content_types = set(args.content_types) if args.content_types else None
    places = load_active_places(
        db_path,
        limit=args.limit,
        content_types=content_types,
    )
    if args.places_filter_csv:
        filter_ids = read_place_filter_csv(Path(args.places_filter_csv))
        before_count = len(places)
        places = [place for place in places if place.content_id in filter_ids]
        print(
            "[places filter] "
            f"{before_count}건 → {len(places)}건 "
            f"(filter ids={len(filter_ids)})"
        )
        if not places:
            raise RuntimeError(
                "--places-filter-csv와 매칭되는 active 장소가 없습니다."
            )
    print(f"[active places] {len(places)}건 로드")

    active_snapshot_path = out_dir / f"active_places_snapshot_{tag}.csv"
    raw_path = out_dir / f"iconic_raw_signals_{tag}.csv"
    preview_path = out_dir / "iconic_score_v2_preview.csv"
    snapshot_path = out_dir / f"iconic_score_v2_snapshot_{tag}.csv"

    write_active_snapshot(active_snapshot_path, places)
    print(f"[active snapshot] {active_snapshot_path}")

    tourism_raw = build_tourism100_exact_map(places, tourism_csv_path)
    datalab_raw = read_signal_csv(
        Path(args.datalab_raw) if args.datalab_raw else None,
        value_columns=[
            "datalab_search_raw",
            "adjusted_search_ratio",
            "search_ratio",
            "search_raw",
        ],
    )
    tar_raw = read_tar_signal_csv(Path(args.tar_raw) if args.tar_raw else None)
    regional_demand_raw = read_signal_csv(
        Path(args.regional_demand_raw) if args.regional_demand_raw else None,
        value_columns=[
            "regional_demand_raw",
            "tourism_resource_demand_raw",
            "demand_raw",
        ],
    )

    raw_rows = build_raw_rows(
        places,
        tourism_raw=tourism_raw,
        datalab_raw=datalab_raw,
        tar_raw=tar_raw,
        regional_demand_raw=regional_demand_raw,
    )
    write_dict_rows(
        raw_path,
        raw_rows,
        [
            "place_id",
            "title",
            "area_code",
            "sigungu_code",
            "content_type_id",
            "tourism100_raw",
            "datalab_search_raw",
            "tar_rlte_raw",
            "regional_demand_raw",
            "collected_at",
            "missing_signal_count",
            "status",
            "error_message",
        ],
    )
    print(f"[raw signals] {raw_path}")

    preview_rows = build_preview_rows(raw_rows)
    preview_fieldnames = [
        "place_id",
        "title",
        "area_code",
        "sigungu_code",
        "content_type_id",
        "tourism100_score",
        "datalab_search_score",
        "tar_rlte_score",
        "regional_demand_score",
        "missing_signal_count",
        "iconic_score_v2",
        "status",
    ]
    write_dict_rows(preview_path, preview_rows, preview_fieldnames)
    write_dict_rows(snapshot_path, preview_rows, preview_fieldnames)
    print(f"[preview] {preview_path}")
    print(f"[snapshot] {snapshot_path}")
    if args.write_signals_db:
        write_signals_to_db(
            db_path,
            raw_rows,
            preview_rows,
            signal_version=args.signal_version,
        )
        print(
            "[DB] place_iconic_signals UPSERT 완료 "
            f"(version={args.signal_version})"
        )
    else:
        print("[안전 모드] places.db는 수정하지 않았습니다.")
    print_summary(raw_rows, preview_rows)


if __name__ == "__main__":
    main()
