"""
Iconic Score ETL v1

두 신호를 결합해 장소별 iconic_score(0~1)를 계산한다.

1. 한국관광 100선 선정 여부
   - 공식 자료는 1~100위 순위가 아니라 100개 장소의 선정 목록이다.
   - 선정 장소는 1점, 미선정 장소는 0점으로 처리한다.

2. Naver DataLab 검색 트렌드
   - DataLab 값은 실제 검색 횟수가 아니라 요청 안에서의 상대 비율이다.
   - 여러 번 나누어 조회할 때는 모든 요청에 같은 기준 장소(anchor)를 넣고,
     각 장소의 값을 기준 장소 값으로 나누어 배치 간 척도를 맞춘다.
   - 보정한 값에는 log1p와 Min-Max 정규화를 적용한다.

전체 장소 Iconic Score 실행 모드

1. datalab-collect
   - SQLite의 is_active=1 장소 전체를 대상으로 한다.
   - 경복궁 anchor 1개와 대상 장소 최대 4개를 한 배치로 조회한다.
   - 각 배치가 끝날 때마다 datalab_full_checkpoint.json에 저장한다.
   - 중간에 종료되어도 같은 명령을 실행하면 남은 장소부터 재개한다.
   - 이 단계에서는 places.db를 수정하지 않는다.

2. score-preview
   - 체크포인트에 수집된 검색 비율을 정규화한다.
   - 한국관광 100선 신호와 검색 점수를 결합한다.
   - iconic_score_full_preview.csv를 생성한다.
   - 미리보기일 뿐 places.db는 수정하지 않는다.

3. apply
   - DataLab 전체 수집이 완료된 경우에만 실행할 수 있다.
   - 계산된 iconic_score를 places.db에 저장한다.
   - Iconic Score는 Pinecone metadata이므로 embedded_at을 변경하지 않는다.
   - 따라서 점수 변경만으로 장소를 재임베딩하지 않는다.

권장 실행 순서:
  datalab-collect 반복 실행
  → score-preview 결과 검증
  → apply로 SQLite 저장
  → embed_pipeline에서 Pinecone metadata만 동기화

실행 예시:
  python3 traveler/data/scripts/iconic_score.py \
    --mode datalab-collect --max-batches 5

주의:
  전체 수집이 끝나기 전에는 apply를 실행하지 않는다.
  기존 체크포인트는 해당 수집 기간과 장소 목록을 고정하여 사용한다.
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
import unicodedata
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import requests
from dotenv import load_dotenv


WEIGHT_TOURISM = 0.2
WEIGHT_SEARCH = 0.8
NAVER_DATALAB_URL = "https://openapi.naver.com/v1/datalab/search"
DEFAULT_DATALAB_WEEKS = 12
DEFAULT_DATALAB_TEST_LIMIT = 20
TARGETS_PER_BATCH = 4

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR / "data"
DEFAULT_TOURISM_100_CSV = DATA_DIR / "korea_tourism_100_2025_2026.csv"
DEFAULT_DB_PATH = DATA_DIR / "places.db"
DEFAULT_MATCH_REPORT = DATA_DIR / "iconic_match_report.csv"
DEFAULT_SCORE_PREVIEW = DATA_DIR / "iconic_score_preview.csv"
DEFAULT_DATALAB_SNAPSHOT = DATA_DIR / "datalab_test_snapshot.json"
DEFAULT_MANUAL_MAPPING = DATA_DIR / "iconic_manual_mapping.csv"
DEFAULT_FULL_CHECKPOINT = DATA_DIR / "datalab_full_checkpoint.json"
DEFAULT_FULL_PREVIEW = DATA_DIR / "iconic_score_full_preview.csv"

load_dotenv()


SPECIAL_ALIASES = {
    "5대 고궁": ["경복궁", "창덕궁", "창경궁", "덕수궁", "경희궁"],
}


@dataclass(frozen=True)
class TourismEntry:
    source_order: int
    region_group: str
    region: str
    place_name: str


@dataclass(frozen=True)
class Place:
    content_id: str
    name: str
    address: str


@dataclass(frozen=True)
class MatchedPlace:
    place: Place
    tourism_place_name: str
    region: str


@dataclass(frozen=True)
class IconicScoreResult:
    place_id: str
    is_tourism_100: bool
    tourism_score: float
    search_ratio: float | None
    search_score: float | None
    iconic_score: float | None
    status: str


def load_tourism_100_entries(
    csv_path: str | Path = DEFAULT_TOURISM_100_CSV,
) -> list[TourismEntry]:
    """한국관광 100선 CSV를 순서를 보존한 목록으로 읽는다."""
    entries: list[TourismEntry] = []

    with Path(csv_path).open(encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"rank", "region_group", "region", "place_name"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                "CSV에 rank, region_group, region, place_name 컬럼이 필요합니다."
            )

        for row in reader:
            # 원본 CSV의 일부 복합 장소명에는 따옴표로 감싸지 않은 쉼표가 있다.
            # DictReader가 남은 조각을 None 키에 넣으므로 다시 결합한다.
            place_name_parts = [row.get("place_name") or ""]
            place_name_parts.extend(row.get(None) or [])
            place_name = ",".join(place_name_parts).strip()
            if not place_name:
                continue
            entries.append(
                TourismEntry(
                    # rank는 점수가 아니라 CSV의 목록 순서로만 보존한다.
                    source_order=int(row["rank"]),
                    region_group=(row.get("region_group") or "").strip(),
                    region=(row.get("region") or "").strip(),
                    place_name=place_name,
                )
            )

    return entries


def load_tourism_100_names(
    csv_path: str | Path = DEFAULT_TOURISM_100_CSV,
) -> set[str]:
    """
    한국관광 100선 CSV에서 선정된 장소 이름을 읽는다.

    필요한 CSV 컬럼: place_name
    기존 CSV에 rank 컬럼이 있어도 사용하지 않는다.

    반환 예시:
        {"경복궁", "불국사&석굴암", "전주 한옥마을"}
    """
    return {
        entry.place_name
        for entry in load_tourism_100_entries(csv_path)
    }


def normalize_name(value: str) -> str:
    """
    장소명 비교용 문자열을 만든다.

    예: "전주 한옥마을"과 "전주한옥마을"을 모두 "전주한옥마을"로 변환한다.
    원본 이름은 수정하지 않고 비교할 때만 사용한다.
    """
    value = unicodedata.normalize("NFKC", value or "").lower()
    return re.sub(r"[^0-9a-z가-힣]", "", value)


def split_tourism_keywords(place_name: str) -> list[str]:
    """
    복합 관광지명을 DB와 비교할 수 있는 개별 검색어로 나눈다.

    예:
      "불국사&석굴암" -> ["불국사", "석굴암"]
      "경주대릉원(동궁과 월지, 첨성대&황리단길)"
        -> ["경주대릉원", "동궁과 월지", "첨성대", "황리단길"]
    """
    if place_name in SPECIAL_ALIASES:
        return SPECIAL_ALIASES[place_name]

    separated = re.sub(r"[()（）&,·/]", "|", place_name)
    keywords = [
        keyword.strip()
        for keyword in separated.split("|")
        if len(normalize_name(keyword)) >= 2
    ]
    return list(dict.fromkeys(keywords))


def load_places(db_path: str | Path = DEFAULT_DB_PATH) -> list[Place]:
    """places.db에서 매칭에 필요한 필드만 읽는다."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"places.db를 찾을 수 없습니다: {db_path}")

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT content_id, name, COALESCE(address, '')
            FROM places
            WHERE name IS NOT NULL
              AND trim(name) != ''
            """
        ).fetchall()

    return [
        Place(content_id=str(row[0]), name=str(row[1]), address=str(row[2]))
        for row in rows
    ]

def load_manual_tourism_ids(
    mapping_path: str | Path = DEFAULT_MANUAL_MAPPING,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> set[str]:
    """수동으로 확인한 한국관광 100선 장소 ID를 읽는다."""
    mapping_path = Path(mapping_path)

    if not mapping_path.exists():
        print(f"[수동 매핑] 파일 없음: {mapping_path}")
        return set()

    confirmed_ids: set[str] = set()

    with mapping_path.open(
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        required = {
            "content_id",
            "decision",
        }
        if (
            reader.fieldnames is None
            or not required.issubset(reader.fieldnames)
        ):
            raise ValueError(
                "수동 매핑 CSV에 content_id, decision 컬럼이 필요합니다."
            )

        for row in reader:
            if (row.get("decision") or "").strip() != "confirmed":
                continue

            content_id = (row.get("content_id") or "").strip()
            if content_id:
                confirmed_ids.add(content_id)

    # 수동 매핑 ID가 실제 활성 DB 장소인지 확인한다.
    with sqlite3.connect(db_path) as conn:
        placeholders = ", ".join(
            "?" for _ in confirmed_ids
        )

        if confirmed_ids:
            rows = conn.execute(
                f"""
                SELECT content_id
                FROM places
                WHERE is_active = 1
                  AND content_id IN ({placeholders})
                """,
                sorted(confirmed_ids),
            ).fetchall()

            existing_ids = {
                str(row[0])
                for row in rows
            }
        else:
            existing_ids = set()

    missing_ids = confirmed_ids - existing_ids
    if missing_ids:
        raise RuntimeError(
            "수동 매핑 ID가 활성 places DB에 없습니다: "
            f"{sorted(missing_ids)}"
        )

    return confirmed_ids


def load_auto_exact_tourism_ids(
    db_path: str | Path = DEFAULT_DB_PATH,
    tourism_csv_path: str | Path = DEFAULT_TOURISM_100_CSV,
) -> set[str]:
    """이름과 지역이 모두 정확히 일치한 100선 장소 ID를 반환한다."""
    entries = load_tourism_100_entries(tourism_csv_path)
    places = load_places(db_path)
    confirmed_ids: set[str] = set()

    for entry in entries:
        exact, _ = find_place_candidates(
            entry,
            places,
        )

        for place in exact:
            # 동명 장소 오매칭 방지를 위해 지역도 확인한다.
            if entry.region not in place.address:
                continue

            confirmed_ids.add(place.content_id)

    return confirmed_ids


def load_confirmed_tourism_ids(
    db_path: str | Path = DEFAULT_DB_PATH,
    tourism_csv_path: str | Path = DEFAULT_TOURISM_100_CSV,
    manual_mapping_path: str | Path = DEFAULT_MANUAL_MAPPING,
) -> set[str]:
    """자동 exact와 수동 confirmed 매핑을 합친다."""
    auto_ids = load_auto_exact_tourism_ids(
        db_path=db_path,
        tourism_csv_path=tourism_csv_path,
    )
    manual_ids = load_manual_tourism_ids(
        mapping_path=manual_mapping_path,
        db_path=db_path,
    )

    confirmed_ids = auto_ids | manual_ids

    print(
        "[100선 확정 장소] "
        f"자동={len(auto_ids)}, "
        f"수동={len(manual_ids)}, "
        f"중복제거 후 전체={len(confirmed_ids)}"
    )

    return confirmed_ids

def find_place_candidates(
    entry: TourismEntry,
    places: list[Place],
) -> tuple[list[Place], list[Place]]:
    """
    정확 매칭 후보와 부분 매칭 후보를 반환한다.

    이 함수는 후보만 제안하고 자동으로 DB 점수를 수정하지 않는다.
    복합 관광지명과 동명 장소 때문에 리포트 검토가 먼저 필요하다.
    """
    keyword_norms = {
        normalize_name(keyword)
        for keyword in split_tourism_keywords(entry.place_name)
    }

    exact: list[Place] = []
    partial: list[Place] = []

    for place in places:
        place_norm = normalize_name(place.name)
        if not place_norm:
            continue

        if place_norm in keyword_norms:
            exact.append(place)
            continue

        if any(
            len(keyword_norm) >= 4
            and (
                keyword_norm in place_norm
                or place_norm in keyword_norm
            )
            for keyword_norm in keyword_norms
        ):
            partial.append(place)

    # CSV 지역과 주소가 일치하는 후보를 앞쪽에 보여줘 검토하기 쉽게 한다.
    exact.sort(key=lambda place: entry.region not in place.address)
    partial.sort(key=lambda place: entry.region not in place.address)
    return exact, partial


def format_places_for_report(places: list[Place]) -> str:
    return " | ".join(
        f"{place.content_id}:{place.name}({place.address})"
        for place in places
    )


def write_match_report(
    db_path: str | Path = DEFAULT_DB_PATH,
    tourism_csv_path: str | Path = DEFAULT_TOURISM_100_CSV,
    report_path: str | Path = DEFAULT_MATCH_REPORT,
) -> dict[str, int]:
    """
    100선과 places.db의 장소명 매칭 후보를 CSV로 저장한다.

    이 단계에서는 iconic_score를 업데이트하지 않는다.
    """
    entries = load_tourism_100_entries(tourism_csv_path)
    places = load_places(db_path)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    counts = {"exact": 0, "review": 0, "unmatched": 0}

    with report_path.open("w", encoding="utf-8-sig", newline="") as file:
        fieldnames = [
            "source_order",
            "region_group",
            "region",
            "tourism_place_name",
            "split_keywords",
            "status",
            "exact_count",
            "exact_candidates",
            "partial_count",
            "partial_candidates",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for entry in entries:
            exact, partial = find_place_candidates(entry, places)
            if exact:
                status = "exact"
            elif partial:
                status = "review"
            else:
                status = "unmatched"
            counts[status] += 1

            writer.writerow(
                {
                    "source_order": entry.source_order,
                    "region_group": entry.region_group,
                    "region": entry.region,
                    "tourism_place_name": entry.place_name,
                    "split_keywords": " | ".join(
                        split_tourism_keywords(entry.place_name)
                    ),
                    "status": status,
                    "exact_count": len(exact),
                    "exact_candidates": format_places_for_report(exact),
                    "partial_count": len(partial),
                    "partial_candidates": format_places_for_report(partial),
                }
            )

    print(f"[매칭 리포트] 저장: {report_path}")
    print(
        "[매칭 결과] "
        f"exact={counts['exact']}, "
        f"review={counts['review']}, "
        f"unmatched={counts['unmatched']}"
    )
    return counts


def select_exact_matched_places(
    db_path: str | Path,
    tourism_csv_path: str | Path,
    limit: int = DEFAULT_DATALAB_TEST_LIMIT,
) -> list[MatchedPlace]:
    """
    exact 이름 매칭과 CSV 지역-DB 주소가 모두 일치하는 장소만 선택한다.

    매칭 리포트의 review 후보는 오탐 가능성이 있으므로 테스트에서 제외한다.
    """
    entries = load_tourism_100_entries(tourism_csv_path)
    places = load_places(db_path)
    selected: list[MatchedPlace] = []
    seen_ids: set[str] = set()

    for entry in entries:
        exact, _ = find_place_candidates(entry, places)
        for place in exact:
            if entry.region not in place.address:
                continue
            if place.content_id in seen_ids:
                continue

            selected.append(
                MatchedPlace(
                    place=place,
                    tourism_place_name=entry.place_name,
                    region=entry.region,
                )
            )
            seen_ids.add(place.content_id)

            if len(selected) >= limit:
                return selected

    return selected


def clean_datalab_keyword(name: str) -> str:
    """
    TourAPI 장소명의 설명 표기를 제거해 DataLab 검색어를 만든다.

    예:
      "수원 화성 [유네스코 세계유산]" -> "수원 화성"
    """
    keyword = re.sub(r"\[[^\]]*]", "", name)
    keyword = re.sub(r"\s+", " ", keyword).strip()
    return keyword


def chunked(values: list[MatchedPlace], size: int) -> Iterable[list[MatchedPlace]]:
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
    places: list[MatchedPlace],
    start_date: str,
    end_date: str,
    client_id: str,
    client_secret: str,
    max_retries: int = 3,
) -> dict[str, float]:
    """anchor 1개와 대상 최대 4개의 최근 주간 평균 ratio를 조회한다."""
    keyword_groups = [
        {
            # 응답을 content_id와 안전하게 연결하기 위해 ID를 그룹명으로 사용한다.
            "groupName": matched.place.content_id,
            "keywords": [clean_datalab_keyword(matched.place.name)],
        }
        for matched in places
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


def write_score_preview(
    matched_places: list[MatchedPlace],
    results: list[IconicScoreResult],
    preview_path: str | Path,
) -> None:
    matched_by_id = {
        matched.place.content_id: matched
        for matched in matched_places
    }
    preview_path = Path(preview_path)
    preview_path.parent.mkdir(parents=True, exist_ok=True)

    with preview_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "content_id",
                "name",
                "address",
                "tourism_place_name",
                "tourism_score",
                "adjusted_search_ratio",
                "search_score",
                "iconic_score",
                "status",
            ],
        )
        writer.writeheader()

        for result in results:
            matched = matched_by_id[result.place_id]
            writer.writerow(
                {
                    "content_id": result.place_id,
                    "name": matched.place.name,
                    "address": matched.place.address,
                    "tourism_place_name": matched.tourism_place_name,
                    "tourism_score": result.tourism_score,
                    "adjusted_search_ratio": result.search_ratio,
                    "search_score": result.search_score,
                    "iconic_score": result.iconic_score,
                    "status": result.status,
                }
            )


def run_datalab_test(
    db_path: str | Path = DEFAULT_DB_PATH,
    tourism_csv_path: str | Path = DEFAULT_TOURISM_100_CSV,
    preview_path: str | Path = DEFAULT_SCORE_PREVIEW,
    snapshot_path: str | Path = DEFAULT_DATALAB_SNAPSHOT,
    limit: int = DEFAULT_DATALAB_TEST_LIMIT,
    anchor_name: str = "경복궁",
    weeks: int = DEFAULT_DATALAB_WEEKS,
) -> list[IconicScoreResult]:
    """
    exact 매칭 장소 최대 20개로 DataLab 보정과 Iconic Score를 시험한다.

    places.db는 수정하지 않고 CSV 미리보기와 JSON 원본 스냅샷만 저장한다.
    """
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
            "NAVER_DATALAB_CLIENT_ID/NAVER_DATALAB_CLIENT_SECRET이 필요합니다. "
            "ID는 NAVER_CLIENT 이름도 지원합니다."
        )

    matched_places = select_exact_matched_places(
        db_path=db_path,
        tourism_csv_path=tourism_csv_path,
        limit=limit,
    )
    if not matched_places:
        raise RuntimeError("DataLab 테스트에 사용할 exact 매칭 장소가 없습니다.")

    anchor_norm = normalize_name(anchor_name)
    anchor = next(
        (
            matched
            for matched in matched_places
            if normalize_name(matched.place.name) == anchor_norm
        ),
        None,
    )
    if anchor is None:
        raise RuntimeError(
            f"exact 매칭 장소 안에서 anchor {anchor_name!r}를 찾지 못했습니다."
        )

    targets = [
        matched
        for matched in matched_places
        if matched.place.content_id != anchor.place.content_id
    ]
    end = date.today()
    start = end - timedelta(weeks=weeks)
    start_date = start.isoformat()
    end_date = end.isoformat()

    batches: list[dict[str, float]] = []
    batch_details: list[dict] = []

    for batch_number, target_batch in enumerate(
        chunked(targets, TARGETS_PER_BATCH),
        start=1,
    ):
        request_places = [anchor, *target_batch]
        batch_ratios = fetch_datalab_batch(
            places=request_places,
            start_date=start_date,
            end_date=end_date,
            client_id=client_id,
            client_secret=client_secret,
        )
        batches.append(batch_ratios)
        batch_details.append(
            {
                "batch_number": batch_number,
                "places": [
                    {
                        "content_id": matched.place.content_id,
                        "name": matched.place.name,
                        "keyword": clean_datalab_keyword(matched.place.name),
                    }
                    for matched in request_places
                ],
                "average_ratios": batch_ratios,
            }
        )
        print(
            f"[DataLab] batch={batch_number}, "
            f"anchor=1, targets={len(target_batch)}"
        )
        time.sleep(0.2)

    adjusted_ratios = adjust_datalab_batches(
        batches,
        anchor_place_id=anchor.place.content_id,
    )
    place_ids = [matched.place.content_id for matched in matched_places]
    tourism_ids = set(place_ids)
    results = compute_iconic_scores(
        place_ids=place_ids,
        tourism_100_place_ids=tourism_ids,
        adjusted_search_ratios=adjusted_ratios,
    )

    write_score_preview(matched_places, results, preview_path)

    snapshot_path = Path(snapshot_path)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    with snapshot_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "start_date": start_date,
                "end_date": end_date,
                "time_unit": "week",
                "anchor": {
                    "content_id": anchor.place.content_id,
                    "name": anchor.place.name,
                },
                "batches": batch_details,
                "adjusted_ratios": adjusted_ratios,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[미리보기] {preview_path}")
    print(f"[원본 스냅샷] {snapshot_path}")
    print("[안전 모드] places.db는 수정하지 않았습니다.")
    return results


def adjust_datalab_batches(
    batches: Iterable[dict[str, float]],
    anchor_place_id: str,
) -> dict[str, float]:
    """
    DataLab 요청별 상대값을 동일한 기준으로 보정한다.

    각 batch는 이미 기간별 ratio를 평균 낸 {place_id: ratio}라고 가정한다.
    모든 batch에는 같은 anchor_place_id가 반드시 들어 있어야 한다.

    예시:
        batch 1 = {"anchor": 80, "p1": 40} -> p1 = 40 / 80 = 0.5
        batch 2 = {"anchor": 50, "p2": 75} -> p2 = 75 / 50 = 1.5
    """
    adjusted: dict[str, float] = {}

    for batch_number, batch in enumerate(batches, start=1):
        if anchor_place_id not in batch:
            raise ValueError(
                f"{batch_number}번째 batch에 기준 장소 "
                f"{anchor_place_id!r}가 없습니다."
            )

        anchor_ratio = batch[anchor_place_id]
        if not math.isfinite(anchor_ratio) or anchor_ratio <= 0:
            raise ValueError(
                f"{batch_number}번째 batch의 기준 장소 값은 "
                "0보다 큰 유한한 숫자여야 합니다."
            )

        for place_id, ratio in batch.items():
            if place_id == anchor_place_id:
                continue
            if not math.isfinite(ratio) or ratio < 0:
                raise ValueError(
                    f"{place_id!r}의 DataLab 값은 0 이상의 "
                    "유한한 숫자여야 합니다."
                )
            if place_id in adjusted:
                raise ValueError(
                    f"{place_id!r}가 여러 batch에 중복되어 있습니다."
                )

            adjusted[place_id] = ratio / anchor_ratio

    # 기준 장소는 자기 자신과 비교하므로 항상 1이다.
    adjusted[anchor_place_id] = 1.0
    return adjusted


def search_ratios_to_scores(
    adjusted_ratios: dict[str, float],
) -> dict[str, float]:
    """
    기준 장소로 보정한 검색 비율을 0~1 점수로 바꾼다.

    log1p:
        검색량이 압도적으로 큰 장소 하나 때문에 나머지 장소의 점수가
        모두 0 근처로 몰리는 현상을 완화한다.

    Min-Max:
        가장 작은 값을 0, 가장 큰 값을 1로 변환한다.
    """
    if not adjusted_ratios:
        return {}

    logged: dict[str, float] = {}
    for place_id, ratio in adjusted_ratios.items():
        if not math.isfinite(ratio) or ratio < 0:
            raise ValueError(
                f"{place_id!r}의 보정 검색 비율은 0 이상의 "
                "유한한 숫자여야 합니다."
            )
        logged[place_id] = math.log1p(ratio)

    minimum = min(logged.values())
    maximum = max(logged.values())

    # 전부 같은 값이라면 누구도 더 높거나 낮지 않으므로 중립값을 부여한다.
    if math.isclose(maximum, minimum):
        return {place_id: 0.5 for place_id in logged}

    return {
        place_id: (value - minimum) / (maximum - minimum)
        for place_id, value in logged.items()
    }


def compute_iconic_scores(
    place_ids: Iterable[str],
    tourism_100_place_ids: set[str],
    adjusted_search_ratios: dict[str, float],
    weight_tourism: float = WEIGHT_TOURISM,
    weight_search: float = WEIGHT_SEARCH,
) -> list[IconicScoreResult]:
    """
    장소별 iconic_score를 계산한다.

    adjusted_search_ratios에는 adjust_datalab_batches()로 보정한 값을 넣는다.
    DataLab 값이 누락된 장소는 인기 없음(0점)으로 간주하지 않고,
    iconic_score=None과 status="missing_search_data"를 반환한다.
    """
    if weight_tourism < 0 or weight_search < 0:
        raise ValueError("가중치는 0 이상이어야 합니다.")
    if not math.isclose(weight_tourism + weight_search, 1.0):
        raise ValueError("weight_tourism + weight_search는 1이어야 합니다.")

    search_scores = search_ratios_to_scores(adjusted_search_ratios)
    results: list[IconicScoreResult] = []

    for place_id in place_ids:
        is_selected = place_id in tourism_100_place_ids
        tourism_score = 1.0 if is_selected else 0.0
        search_ratio = adjusted_search_ratios.get(place_id)
        search_score = search_scores.get(place_id)

        if search_score is None:
            iconic_score = None
            status = "missing_search_data"
        else:
            iconic_score = (
                weight_tourism * tourism_score
                + weight_search * search_score
            )
            iconic_score = round(iconic_score, 4)
            status = "ok"

        results.append(
            IconicScoreResult(
                place_id=place_id,
                is_tourism_100=is_selected,
                tourism_score=tourism_score,
                search_ratio=(
                    round(search_ratio, 4)
                    if search_ratio is not None
                    else None
                ),
                search_score=(
                    round(search_score, 4)
                    if search_score is not None
                    else None
                ),
                iconic_score=iconic_score,
                status=status,
            )
        )

    return results


def load_active_places(
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[Place]:
    """DataLab 전체 수집 대상인 활성 장소를 읽는다."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"places.db를 찾을 수 없습니다: {db_path}")

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT content_id, name, COALESCE(address, '')
            FROM places
            WHERE is_active = 1
              AND name IS NOT NULL
              AND trim(name) != ''
            ORDER BY content_id
            """
        ).fetchall()

    return [
        Place(str(content_id), str(name), str(address))
        for content_id, name, address in rows
    ]


def get_datalab_credentials() -> tuple[str, str]:
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
            ".env에 NAVER_CLIENT_ID와 NAVER_CLIENT_SECRET이 필요합니다."
        )
    return client_id, client_secret


def place_to_dict(place: Place) -> dict[str, str]:
    return {
        "content_id": place.content_id,
        "name": place.name,
        "address": place.address,
    }


def place_from_dict(data: dict) -> Place:
    return Place(
        content_id=str(data["content_id"]),
        name=str(data["name"]),
        address=str(data.get("address") or ""),
    )


def write_json_atomic(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    temporary_path.replace(path)


def create_full_checkpoint(
    db_path: str | Path,
    checkpoint_path: str | Path,
    anchor_name: str,
    weeks: int,
) -> dict:
    places = load_active_places(db_path)
    anchor_norm = normalize_name(anchor_name)
    anchor = next(
        (
            place
            for place in places
            if normalize_name(place.name) == anchor_norm
        ),
        None,
    )
    if anchor is None:
        raise RuntimeError(
            f"활성 장소에서 anchor {anchor_name!r}를 찾지 못했습니다."
        )

    end = date.today()
    start = end - timedelta(weeks=weeks)
    checkpoint = {
        "version": 1,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "weeks": weeks,
        "anchor": place_to_dict(anchor),
        # 수집 도중 DB가 변해도 이번 주 대상은 고정한다.
        "places": [place_to_dict(place) for place in places],
        "adjusted_ratios": {anchor.content_id: 1.0},
        "completed_batches": 0,
    }
    write_json_atomic(checkpoint_path, checkpoint)
    print(
        f"[체크포인트 생성] 대상={len(places)}, "
        f"기간={checkpoint['start_date']}~{checkpoint['end_date']}"
    )
    return checkpoint


def load_or_create_full_checkpoint(
    db_path: str | Path,
    checkpoint_path: str | Path,
    anchor_name: str,
    weeks: int,
) -> dict:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return create_full_checkpoint(
            db_path,
            checkpoint_path,
            anchor_name,
            weeks,
        )

    with checkpoint_path.open(encoding="utf-8") as file:
        checkpoint = json.load(file)

    required = {
        "start_date",
        "end_date",
        "anchor",
        "places",
        "adjusted_ratios",
    }
    missing = required - set(checkpoint)
    if missing:
        raise RuntimeError(
            f"체크포인트 필수 값이 없습니다: {sorted(missing)}"
        )

    saved_anchor = str(checkpoint["anchor"].get("name") or "")
    if normalize_name(saved_anchor) != normalize_name(anchor_name):
        raise RuntimeError(
            "기존 체크포인트의 anchor가 다릅니다. "
            "다른 체크포인트 경로를 사용하세요."
        )
    return checkpoint


def run_full_datalab_collect(
    db_path: str | Path = DEFAULT_DB_PATH,
    checkpoint_path: str | Path = DEFAULT_FULL_CHECKPOINT,
    anchor_name: str = "경복궁",
    weeks: int = DEFAULT_DATALAB_WEEKS,
    max_batches: int = 5,
) -> dict:
    """활성 장소 전체를 수집하며 batch마다 체크포인트를 저장한다."""
    if max_batches <= 0:
        raise ValueError("--max-batches는 1 이상이어야 합니다.")

    client_id, client_secret = get_datalab_credentials()
    checkpoint = load_or_create_full_checkpoint(
        db_path,
        checkpoint_path,
        anchor_name,
        weeks,
    )
    places = [
        place_from_dict(item)
        for item in checkpoint["places"]
    ]
    anchor = place_from_dict(checkpoint["anchor"])
    adjusted_ratios = {
        str(content_id): float(ratio)
        for content_id, ratio
        in checkpoint["adjusted_ratios"].items()
    }
    pending = [
        place
        for place in places
        if place.content_id not in adjusted_ratios
    ]

    if not pending:
        print("[DataLab 전체 수집] 이미 완료되었습니다.")
        return checkpoint

    executed_batches = 0
    pending_wrapped = [
        MatchedPlace(place, "", "")
        for place in pending
    ]
    for target_batch in chunked(
        pending_wrapped,
        TARGETS_PER_BATCH,
    ):
        if executed_batches >= max_batches:
            break

        request_places = [
            MatchedPlace(anchor, "anchor", ""),
            *target_batch,
        ]
        ratios = fetch_datalab_batch(
            places=request_places,
            start_date=str(checkpoint["start_date"]),
            end_date=str(checkpoint["end_date"]),
            client_id=client_id,
            client_secret=client_secret,
        )
        anchor_ratio = ratios.get(anchor.content_id)
        if (
            anchor_ratio is None
            or not math.isfinite(anchor_ratio)
            or anchor_ratio <= 0
        ):
            raise RuntimeError(
                "DataLab anchor 값이 없거나 0입니다. "
                "이전 batch까지의 결과는 보존되었습니다."
            )

        for matched in target_batch:
            content_id = matched.place.content_id
            if content_id not in ratios:
                raise RuntimeError(
                    f"DataLab 응답에 {content_id} 결과가 없습니다."
                )
            ratio = float(ratios[content_id])
            if not math.isfinite(ratio) or ratio < 0:
                raise RuntimeError(
                    f"{content_id}의 DataLab 값이 올바르지 않습니다."
                )
            adjusted_ratios[content_id] = ratio / anchor_ratio

        executed_batches += 1
        checkpoint["adjusted_ratios"] = adjusted_ratios
        checkpoint["completed_batches"] = (
            int(checkpoint.get("completed_batches") or 0) + 1
        )
        write_json_atomic(checkpoint_path, checkpoint)

        print(
            "[DataLab 전체] "
            f"이번={executed_batches}/{max_batches}, "
            f"누적={len(adjusted_ratios)}/{len(places)}"
        )
        time.sleep(0.2)

    print(
        "[수집 상태] "
        f"완료={len(adjusted_ratios)}/{len(places)}, "
        f"남음={len(places) - len(adjusted_ratios)}"
    )
    return checkpoint


def load_full_checkpoint(
    checkpoint_path: str | Path,
    require_complete: bool,
) -> tuple[list[Place], dict[str, float], bool]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"체크포인트가 없습니다: {checkpoint_path}"
        )

    with checkpoint_path.open(encoding="utf-8") as file:
        checkpoint = json.load(file)

    places = [
        place_from_dict(item)
        for item in checkpoint["places"]
    ]
    adjusted_ratios = {
        str(content_id): float(ratio)
        for content_id, ratio
        in checkpoint["adjusted_ratios"].items()
    }
    complete = all(
        place.content_id in adjusted_ratios
        for place in places
    )
    if require_complete and not complete:
        raise RuntimeError(
            "DataLab 전체 수집이 완료되지 않았습니다. "
            f"완료={len(adjusted_ratios)}/{len(places)}"
        )
    return places, adjusted_ratios, complete


def build_full_score_results(
    db_path: str | Path,
    tourism_csv_path: str | Path,
    manual_mapping_path: str | Path,
    checkpoint_path: str | Path,
    require_complete: bool,
) -> tuple[list[Place], list[IconicScoreResult], bool]:
    places, adjusted_ratios, complete = load_full_checkpoint(
        checkpoint_path,
        require_complete,
    )
    tourism_ids = load_confirmed_tourism_ids(
        db_path=db_path,
        tourism_csv_path=tourism_csv_path,
        manual_mapping_path=manual_mapping_path,
    )
    results = compute_iconic_scores(
        place_ids=[place.content_id for place in places],
        tourism_100_place_ids=tourism_ids,
        adjusted_search_ratios=adjusted_ratios,
    )
    return places, results, complete


def run_full_score_preview(
    db_path: str | Path = DEFAULT_DB_PATH,
    tourism_csv_path: str | Path = DEFAULT_TOURISM_100_CSV,
    manual_mapping_path: str | Path = DEFAULT_MANUAL_MAPPING,
    checkpoint_path: str | Path = DEFAULT_FULL_CHECKPOINT,
    preview_path: str | Path = DEFAULT_FULL_PREVIEW,
) -> list[IconicScoreResult]:
    places, results, complete = build_full_score_results(
        db_path,
        tourism_csv_path,
        manual_mapping_path,
        checkpoint_path,
        False,
    )
    places_by_id = {
        place.content_id: place
        for place in places
    }
    preview_path = Path(preview_path)
    preview_path.parent.mkdir(parents=True, exist_ok=True)

    with preview_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "content_id",
                "name",
                "address",
                "is_tourism_100",
                "tourism_score",
                "adjusted_search_ratio",
                "search_score",
                "iconic_score",
                "status",
                "collection_complete",
            ],
        )
        writer.writeheader()
        for result in results:
            place = places_by_id[result.place_id]
            writer.writerow(
                {
                    "content_id": result.place_id,
                    "name": place.name,
                    "address": place.address,
                    "is_tourism_100": int(result.is_tourism_100),
                    "tourism_score": result.tourism_score,
                    "adjusted_search_ratio": result.search_ratio,
                    "search_score": result.search_score,
                    "iconic_score": result.iconic_score,
                    "status": result.status,
                    "collection_complete": int(complete),
                }
            )

    print(
        f"[전체 미리보기] {preview_path} "
        f"(수집 완료 여부={complete})"
    )
    print("[안전 모드] places.db는 수정하지 않았습니다.")
    return results


def apply_full_scores_to_db(
    db_path: str | Path = DEFAULT_DB_PATH,
    tourism_csv_path: str | Path = DEFAULT_TOURISM_100_CSV,
    manual_mapping_path: str | Path = DEFAULT_MANUAL_MAPPING,
    checkpoint_path: str | Path = DEFAULT_FULL_CHECKPOINT,
) -> int:
    _, results, complete = build_full_score_results(
        db_path,
        tourism_csv_path,
        manual_mapping_path,
        checkpoint_path,
        True,
    )
    if not complete:
        raise RuntimeError("완료되지 않은 점수는 저장할 수 없습니다.")

    with sqlite3.connect(db_path) as conn:
        columns = {
            str(row[1])
            for row in conn.execute(
                "PRAGMA table_info(places)"
            ).fetchall()
        }
        if "iconic_score" not in columns:
            raise RuntimeError(
                "places 테이블에 iconic_score 컬럼이 없습니다."
            )

        updated = 0
        for result in results:
            if result.iconic_score is None:
                continue
            cursor = conn.execute(
                """
                UPDATE places
                SET iconic_score = ?
                WHERE content_id = ?
                  AND is_active = 1
                """,
                (result.iconic_score, result.place_id),
            )
            updated += cursor.rowcount
        conn.commit()

    print(
        f"[DB 저장 완료] iconic_score={updated}건, "
        "embedded_at은 변경하지 않았습니다."
    )
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="한국관광 100선 매칭 및 Iconic Score ETL"
    )
    parser.add_argument(
        "--mode",
        choices=[
            "match",
            "datalab-test",
            "datalab-collect",
            "score-preview",
            "apply",
        ],
        default="match",
        help=(
            "match=매칭 리포트, datalab-test=20개 파일럿, "
            "datalab-collect=전체 수집, score-preview=전체 미리보기, "
            "apply=완료 점수 DB 저장"
        ),
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"places.db 경로 (기본값: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--tourism-csv",
        type=Path,
        default=DEFAULT_TOURISM_100_CSV,
        help=f"한국관광 100선 CSV 경로 (기본값: {DEFAULT_TOURISM_100_CSV})",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_MATCH_REPORT,
        help=f"매칭 리포트 저장 경로 (기본값: {DEFAULT_MATCH_REPORT})",
    )
    parser.add_argument(
        "--preview",
        type=Path,
        default=DEFAULT_SCORE_PREVIEW,
        help=f"점수 미리보기 경로 (기본값: {DEFAULT_SCORE_PREVIEW})",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=DEFAULT_DATALAB_SNAPSHOT,
        help=f"DataLab 스냅샷 경로 (기본값: {DEFAULT_DATALAB_SNAPSHOT})",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_FULL_CHECKPOINT,
        help=f"전체 수집 체크포인트 (기본값: {DEFAULT_FULL_CHECKPOINT})",
    )
    parser.add_argument(
        "--manual-mapping",
        type=Path,
        default=DEFAULT_MANUAL_MAPPING,
        help=f"수동 매핑 CSV (기본값: {DEFAULT_MANUAL_MAPPING})",
    )
    parser.add_argument(
        "--full-preview",
        type=Path,
        default=DEFAULT_FULL_PREVIEW,
        help=f"전체 점수 미리보기 (기본값: {DEFAULT_FULL_PREVIEW})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_DATALAB_TEST_LIMIT,
        help="DataLab 테스트 대상 최대 장소 수",
    )
    parser.add_argument(
        "--anchor-name",
        default="경복궁",
        help="모든 DataLab 배치에 포함할 기준 장소명",
    )
    parser.add_argument(
        "--weeks",
        type=int,
        default=DEFAULT_DATALAB_WEEKS,
        help="DataLab 검색 추이를 조회할 최근 주 수",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=5,
        help="datalab-collect 한 번에 실행할 최대 batch 수 (batch당 대상 4개)",
    )
    args = parser.parse_args()

    if args.mode == "match":
        write_match_report(
            db_path=args.db,
            tourism_csv_path=args.tourism_csv,
            report_path=args.report,
        )

        load_confirmed_tourism_ids(
            db_path=args.db,
            tourism_csv_path=args.tourism_csv,
            manual_mapping_path=DEFAULT_MANUAL_MAPPING,
        )
    elif args.mode == "datalab-test":
        run_datalab_test(
            db_path=args.db,
            tourism_csv_path=args.tourism_csv,
            preview_path=args.preview,
            snapshot_path=args.snapshot,
            limit=args.limit,
            anchor_name=args.anchor_name,
            weeks=args.weeks,
        )
    elif args.mode == "datalab-collect":
        run_full_datalab_collect(
            db_path=args.db,
            checkpoint_path=args.checkpoint,
            anchor_name=args.anchor_name,
            weeks=args.weeks,
            max_batches=args.max_batches,
        )
    elif args.mode == "score-preview":
        run_full_score_preview(
            db_path=args.db,
            tourism_csv_path=args.tourism_csv,
            manual_mapping_path=args.manual_mapping,
            checkpoint_path=args.checkpoint,
            preview_path=args.full_preview,
        )
    else:
        apply_full_scores_to_db(
            db_path=args.db,
            tourism_csv_path=args.tourism_csv,
            manual_mapping_path=args.manual_mapping,
            checkpoint_path=args.checkpoint,
        )


if __name__ == "__main__":
    main()
