"""
Naver DataLab search raw signal 수집기.

역할:
  1. places.db 또는 입력 CSV에서 iconic score 대상 장소를 읽는다.
  2. 모든 DataLab 요청에 같은 anchor keyword를 포함한다. (naver datalab 검색량 자체가 이런식으로만 제공이 됨)
     - 기본 anchor: 경복궁
     - 1회 요청: anchor 1개 + 대상 장소 최대 4개 = 총 5개 keywordGroups
  3. DataLab 응답의 상대 ratio를 anchor 기준으로 보정한다.
     - datalab_search_raw = place_average_ratio / anchor_average_ratio
  4. iconic_score_v2.py 입력용 datalab_search_raw CSV를 저장한다.

주의:
  Naver DataLab은 실제 검색 횟수가 아니라 "요청 안에서의 상대 비율"을 반환한다.
  그래서 batch마다 anchor를 넣어야 서로 다른 batch의 값을 비교할 수 있다.

예시:
  # 구로구 pilot: DB에서 주소에 서울특별시+구로구가 포함된 active 장소 조회
  ./venv/bin/python traveler/data/scripts/collector_datalab_raw.py \
    --region 구로구 \
    --area 서울특별시

  # regional_demand_raw 파일과 같은 장소 집합을 대상으로 조회
  ./venv/bin/python traveler/data/scripts/collector_datalab_raw.py \
    --input-places traveler/data/regional_demand_raw_20260721.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import requests
from dotenv import load_dotenv


NAVER_DATALAB_URL = "https://openapi.naver.com/v1/datalab/search"
TARGETS_PER_BATCH = 4
ANCHOR_GROUP_ID = "__anchor__"


def find_project_root() -> Path:
    """스크립트 위치와 실행 위치가 달라도 프로젝트 루트를 찾는다."""
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / ".env").exists() or (parent / "traveler").exists():
            return parent
    return current.parent


ROOT_DIR = find_project_root()
DATA_DIR = ROOT_DIR / "traveler" / "data"
DEFAULT_DB_PATH = DATA_DIR / "places.db"
DEFAULT_OUT_DIR = DATA_DIR

load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class Place:
    content_id: str
    name: str
    keyword: str
    address: str = ""
    content_type_id: str = ""


def clean_datalab_keyword(name: str) -> str:
    """
    TourAPI 장소명의 설명 표기를 제거해 DataLab 검색어를 만든다.

    예:
      "수원 화성 [유네스코 세계유산]" -> "수원 화성"
    """
    keyword = re.sub(r"\[[^\]]*]", "", name)
    keyword = re.sub(r"\([^)]*전체[^)]*\)", "", keyword)
    keyword = re.sub(r"\s+", " ", keyword).strip()
    return keyword


def get_naver_credentials() -> tuple[str, str]:
    client_id = (
        os.getenv("NAVER_CLIENT_ID")
        or os.getenv("NAVER_DATALAB_CLIENT_ID")
        or os.getenv("NAVER_CLIENT")
    )
    client_secret = (
        os.getenv("NAVER_CLIENT_SECRET")
        or os.getenv("NAVER_DATALAB_CLIENT_SECRET")
    )
    if not client_id or not client_secret:
        raise RuntimeError(
            ".env에 NAVER_CLIENT_ID/NAVER_CLIENT_SECRET 또는 "
            "NAVER_DATALAB_CLIENT_ID/NAVER_DATALAB_CLIENT_SECRET이 필요합니다."
        )
    return client_id, client_secret


def load_places_from_db(
    db_path: Path,
    *,
    area: str | None,
    region: str | None,
    limit: int | None,
) -> list[Place]:
    if not db_path.exists():
        raise FileNotFoundError(f"places.db를 찾을 수 없습니다: {db_path}")

    conditions = [
        "is_active = 1",
        "name IS NOT NULL",
        "trim(name) != ''",
    ]
    params: list[str] = []

    # 시군구(region)가 있으면 광역지역(area) + 시군구를 모두 만족하도록 엄격 매칭.
    if area:
        conditions.append("COALESCE(address, '') LIKE ?")
        params.append(f"%{area}%")
    if region:
        conditions.append("COALESCE(address, '') LIKE ?")
        params.append(f"%{region}%")

    limit_sql = ""
    if limit is not None and limit > 0:
        limit_sql = "LIMIT ?"
        params.append(str(limit))

    query = f"""
        SELECT
            CAST(content_id AS TEXT) AS content_id,
            COALESCE(name, '') AS name,
            COALESCE(address, '') AS address,
            COALESCE(CAST(content_type_id AS TEXT), '') AS content_type_id
        FROM places
        WHERE {' AND '.join(conditions)}
        ORDER BY content_type_id, name
        {limit_sql}
    """

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, params).fetchall()

    return [
        Place(
            content_id=str(row["content_id"]),
            name=str(row["name"]),
            keyword=clean_datalab_keyword(str(row["name"])),
            address=str(row["address"]),
            content_type_id=str(row["content_type_id"]),
        )
        for row in rows
    ]


def load_places_from_csv(path: Path, *, limit: int | None) -> list[Place]:
    if not path.exists():
        raise FileNotFoundError(f"입력 CSV를 찾을 수 없습니다: {path}")

    places: list[Place] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("입력 CSV header가 없습니다.")

        if "content_id" not in reader.fieldnames or "name" not in reader.fieldnames:
            raise ValueError("입력 CSV에는 content_id, name 컬럼이 필요합니다.")

        for row in reader:
            content_id = str(row.get("content_id") or "").strip()
            name = str(row.get("name") or "").strip()
            if not content_id or not name or content_id in seen:
                continue

            seen.add(content_id)
            places.append(
                Place(
                    content_id=content_id,
                    name=name,
                    keyword=clean_datalab_keyword(name),
                    address=str(row.get("address") or "").strip(),
                    content_type_id=str(row.get("content_type_id") or "").strip(),
                )
            )

            if limit is not None and limit > 0 and len(places) >= limit:
                break

    return places


def chunked(values: list[Place], size: int) -> Iterable[list[Place]]:
    for index in range(0, len(values), size):
        yield values[index:index + size]


def average_ratio(result: dict) -> float:
    ratios = [
        float(point["ratio"])
        for point in result.get("data", [])
        if point.get("ratio") is not None
    ]
    return sum(ratios) / len(ratios) if ratios else 0.0


def fetch_datalab_batch(
    target_batch: list[Place],
    *,
    anchor_keyword: str,
    start_date: str,
    end_date: str,
    client_id: str,
    client_secret: str,
    max_retries: int = 3,
) -> dict[str, float]:
    keyword_groups = [
        {
            "groupName": ANCHOR_GROUP_ID,
            "keywords": [anchor_keyword],
        },
        *[
            {
                # 응답을 content_id와 안전하게 연결하기 위해 groupName에 content_id 사용.
                "groupName": place.content_id,
                "keywords": [place.keyword],
            }
            for place in target_batch
        ],
    ]
    payload = {
        "startDate": start_date,
        "endDate": end_date,
        "timeUnit": "week",
        "keywordGroups": keyword_groups,
    }
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
        "Content-Type": "application/json",
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                NAVER_DATALAB_URL,
                headers=headers,
                json=payload,
                timeout=(5, 30),
            )
            response.raise_for_status()
            data = response.json()
            return {
                str(result["title"]): average_ratio(result)
                for result in data.get("results", [])
            }
        except (
            requests.Timeout,
            requests.ConnectionError,
            requests.HTTPError,
            requests.JSONDecodeError,
        ):
            if attempt == max_retries:
                raise
            time.sleep(2 * attempt)

    raise RuntimeError("DataLab 최대 재시도 횟수를 초과했습니다.")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def collect_datalab_raw(
    places: list[Place],
    *,
    anchor_keyword: str,
    weeks: int,
    sleep_seconds: float,
) -> tuple[list[dict], dict]:
    client_id, client_secret = get_naver_credentials()

    end = date.today()
    start = end - timedelta(weeks=weeks)
    start_date = start.isoformat()
    end_date = end.isoformat()

    raw_rows: list[dict] = []
    snapshot_batches: list[dict] = []

    for batch_number, target_batch in enumerate(
        chunked(places, TARGETS_PER_BATCH),
        start=1,
    ):
        batch_ratios = fetch_datalab_batch(
            target_batch,
            anchor_keyword=anchor_keyword,
            start_date=start_date,
            end_date=end_date,
            client_id=client_id,
            client_secret=client_secret,
        )
        anchor_ratio = batch_ratios.get(ANCHOR_GROUP_ID, 0.0)
        batch_status = "ok"
        if not math.isfinite(anchor_ratio) or anchor_ratio <= 0:
            batch_status = "anchor_ratio_zero"

        for place in target_batch:
            place_ratio = batch_ratios.get(place.content_id, 0.0)
            if batch_status != "ok":
                datalab_search_raw = None
                status = batch_status
            elif not math.isfinite(place_ratio) or place_ratio < 0:
                datalab_search_raw = None
                status = "invalid_place_ratio"
            else:
                datalab_search_raw = place_ratio / anchor_ratio
                status = "ok"

            raw_rows.append(
                {
                    "content_id": place.content_id,
                    "name": place.name,
                    "keyword": place.keyword,
                    "content_type_id": place.content_type_id,
                    "datalab_search_raw": datalab_search_raw,
                    "place_ratio": place_ratio,
                    "anchor_ratio": anchor_ratio,
                    "anchor_keyword": anchor_keyword,
                    "batch_number": batch_number,
                    "status": status,
                }
            )

        snapshot_batches.append(
            {
                "batch_number": batch_number,
                "anchor_keyword": anchor_keyword,
                "places": [
                    {
                        "content_id": place.content_id,
                        "name": place.name,
                        "keyword": place.keyword,
                    }
                    for place in target_batch
                ],
                "average_ratios": batch_ratios,
                "status": batch_status,
            }
        )
        print(
            f"[DataLab] batch={batch_number}, "
            f"targets={len(target_batch)}, anchor_ratio={anchor_ratio:.4f}"
        )
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    snapshot = {
        "start_date": start_date,
        "end_date": end_date,
        "time_unit": "week",
        "anchor": {
            "group_id": ANCHOR_GROUP_ID,
            "keyword": anchor_keyword,
        },
        "target_count": len(places),
        "batches": snapshot_batches,
    }
    return raw_rows, snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Naver DataLab 검색 관심도를 content_id별 raw CSV로 변환"
    )
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="places.db 경로")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="출력 디렉터리")
    parser.add_argument(
        "--input-places",
        default=None,
        help="content_id,name 컬럼을 가진 입력 CSV. 지정하면 DB region 필터 대신 사용",
    )
    parser.add_argument(
        "--area",
        default="서울특별시",
        help="DB 조회 시 address에 포함될 광역지역명. 빈 문자열이면 미사용",
    )
    parser.add_argument(
        "--region",
        default="구로구",
        help="DB 조회 시 address에 포함될 시군구명. 빈 문자열이면 미사용",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="대상 장소 수 제한. 생략하면 조건에 맞는 전체 장소",
    )
    parser.add_argument(
        "--anchor-keyword",
        default="경복궁",
        help="모든 DataLab batch에 넣을 기준 검색어",
    )
    parser.add_argument(
        "--weeks",
        type=int,
        default=52,
        help="조회 기간. 오늘 기준 최근 N주",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.2,
        help="DataLab batch 사이 대기 시간",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_path = Path(args.db)
    out_dir = Path(args.out_dir)
    tag = datetime.now(timezone.utc).strftime("%Y%m%d")

    if args.input_places:
        places = load_places_from_csv(Path(args.input_places), limit=args.limit)
        source_label = Path(args.input_places).stem
    else:
        area = str(args.area).strip() or None
        region = str(args.region).strip() or None
        places = load_places_from_db(
            db_path,
            area=area,
            region=region,
            limit=args.limit,
        )
        source_label = "_".join(value for value in [area, region] if value) or "all"

    if not places:
        raise RuntimeError("DataLab 수집 대상 장소가 없습니다.")

    print(f"[target] places={len(places)}")
    print(f"[anchor] {args.anchor_keyword}")

    raw_rows, snapshot = collect_datalab_raw(
        places,
        anchor_keyword=args.anchor_keyword,
        weeks=args.weeks,
        sleep_seconds=args.sleep_seconds,
    )

    safe_source = re.sub(r"[^0-9A-Za-z가-힣_-]+", "_", source_label).strip("_")
    raw_path = out_dir / f"datalab_search_raw_{safe_source}_{tag}.csv"
    snapshot_path = out_dir / f"datalab_search_snapshot_{safe_source}_{tag}.json"

    write_csv(
        raw_path,
        raw_rows,
        [
            "content_id",
            "name",
            "keyword",
            "content_type_id",
            "datalab_search_raw",
            "place_ratio",
            "anchor_ratio",
            "anchor_keyword",
            "batch_number",
            "status",
        ],
    )
    write_json(snapshot_path, snapshot)

    ok_count = sum(1 for row in raw_rows if row["status"] == "ok")
    print("\n[완료]")
    print(f"- raw: {raw_path}")
    print(f"- snapshot: {snapshot_path}")
    print(f"- raw row 수: {len(raw_rows)}")
    print(f"- ok row 수: {ok_count}")
    print("[안전 모드] places.db는 수정하지 않았습니다.")


if __name__ == "__main__":
    main()
