"""
TarRlteTar 연관 관광지 raw signal 수집기.
관광정보 api에서 받아오는 content_id는 숫자로 되어있음.
iconic_score_v2 를 계산하기 위한 연관 관광지 정보는 해시값으로 들어옴. 그래서 장소명을 바탕으로 매칭해주는 작업이 필요함.
그 매칭 작업을 하는 스크립트.
우선적으로 장소 수집이 많이된 서울, 경기권을 바탕으로 파일럿 실행 진행. DB에 장소가 충분히 쌓이면 그 때 확장 예정.

역할:
  1. 한국관광공사 관광지별 연관 관광지 정보 API 호출
  2. 응답의 tAtsNm → rlteTatsNm 관계를 edge로 해석
  3. API 장소명/지역명을 places.db의 name/address와 매칭
  4. inbound_count / outbound_count 계산
  5. iconic_score_v2.py 입력용 tar_rlte_raw CSV 저장
  6. 매칭 실패 리포트 저장

주의:
  TarRlteTar API의 tAtsCd/rlteTatsCd는 places.content_id와 다르다.
  따라서 ID 조인이 아니라 장소명 + 지역명 기반 매칭을 사용한다.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sqlite3
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT_DIR / "traveler" / "data"
DEFAULT_DB_PATH = DATA_DIR / "places.db"
DEFAULT_OUT_DIR = DATA_DIR

load_dotenv(ROOT_DIR / ".env")

API_KEY = (
    os.getenv("TARRITETARSERVICE_API_KEY")
    or os.getenv("TAR_RLTE_TAR_API_KEY")
    or os.getenv("KORSERVICE_API_KEY")
)

BASE_URL = "https://apis.data.go.kr/B551011/TarRlteTarService1"
PAGE_SIZE = 100
REQUEST_INTERVAL_SECONDS = 0.3


@dataclass(frozen=True)
class Place:
    content_id: str
    name: str
    address: str
    area_code: str
    sigungu_code: str


@dataclass(frozen=True)
class ApiArea:
    area_cd: str
    signgu_cd: str
    label: str


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").lower()
    return re.sub(r"[^0-9a-z가-힣]", "", value)


def load_places(db_path: Path) -> list[Place]:
    if not db_path.exists():
        raise FileNotFoundError(f"places.db를 찾을 수 없습니다: {db_path}")

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT
                CAST(content_id AS TEXT) AS content_id,
                COALESCE(name, '') AS name,
                COALESCE(address, '') AS address,
                COALESCE(CAST(area_code AS TEXT), '') AS area_code,
                COALESCE(CAST(sigungu_code AS TEXT), '') AS sigungu_code
            FROM places
            WHERE is_active = 1
              AND name IS NOT NULL
              AND trim(name) != ''
            """
        ).fetchall()

    return [
        Place(
            content_id=str(row["content_id"]),
            name=str(row["name"]),
            address=str(row["address"]),
            area_code=str(row["area_code"]),
            sigungu_code=str(row["sigungu_code"]),
        )
        for row in rows
    ]


def build_name_index(places: list[Place]) -> dict[str, list[Place]]:
    index: dict[str, list[Place]] = defaultdict(list)
    for place in places:
        key = normalize_name(place.name)
        if key:
            index[key].append(place)
    return index


def region_matches(place: Place, region_name: str, signgu_name: str) -> bool:
    address = place.address or ""
    candidates = [
        region_name,
        signgu_name,
        region_name.replace("특별시", ""),
        region_name.replace("광역시", ""),
        region_name.replace("특별자치시", ""),
        region_name.replace("특별자치도", ""),
        region_name.replace("도", ""),
    ]
    candidates = [value for value in candidates if value]

    # 시군구가 맞으면 가장 강한 지역 매칭으로 본다.
    if signgu_name and signgu_name in address:
        return True

    return any(value and value in address for value in candidates)


def match_place(
    *,
    name: str,
    region_name: str,
    signgu_name: str,
    name_index: dict[str, list[Place]],
) -> Place | None:
    key = normalize_name(name)
    candidates = name_index.get(key, [])
    if not candidates:
        return None

    region_matched = [
        place
        for place in candidates
        if region_matches(place, region_name, signgu_name)
    ]
    if len(region_matched) == 1:
        return region_matched[0]
    if len(region_matched) > 1:
        return region_matched[0]

    # 이름 exact 후보가 하나뿐이면 지역 문자열이 조금 달라도 fallback 허용.
    if len(candidates) == 1:
        return candidates[0]

    return None


def fetch_area_based_list(
    *,
    area_cd: str,
    signgu_cd: str,
    base_ym: str,
    page_no: int,
    num_of_rows: int = PAGE_SIZE,
) -> tuple[list[dict], int]:
    if not API_KEY:
        raise RuntimeError("TARRITETARSERVICE_API_KEY 환경변수가 필요합니다.")

    url = f"{BASE_URL}/areaBasedList1"
    params = {
        "serviceKey": API_KEY,
        "MobileOS": "ETC",
        "MobileApp": "TravelCurator",
        "_type": "json",
        "numOfRows": num_of_rows,
        "pageNo": page_no,
        "areaCd": area_cd,
        "signguCd": signgu_cd,
        "baseYm": base_ym,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    if "response" not in data:
        raise RuntimeError(f"예상하지 못한 응답: {data}")

    header = data["response"].get("header", {})
    result_code = str(header.get("resultCode", ""))
    if result_code != "0000":
        raise RuntimeError(f"TarRlteTar API 오류: {header}")

    body = data["response"].get("body", {})
    total_count = int(body.get("totalCount") or 0)
    items = body.get("items") or {}

    if isinstance(items, str):
        return [], total_count

    item = items.get("item") if isinstance(items, dict) else []
    if isinstance(item, dict):
        return [item], total_count
    if isinstance(item, list):
        return item, total_count

    return [], total_count


def read_area_plan(path: Path | None) -> list[ApiArea]:
    if path is None:
        # 공식 문서 샘플로 확인된 pilot.
        return [ApiArea(area_cd="11", signgu_cd="11530", label="서울 구로구")]

    areas: list[ApiArea] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"areaCd", "signguCd"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("area plan CSV에는 areaCd, signguCd 컬럼이 필요합니다.")

        for row in reader:
            area_cd = str(row.get("areaCd") or "").strip()
            signgu_cd = str(row.get("signguCd") or "").strip()
            label = str(row.get("label") or f"{area_cd}/{signgu_cd}").strip()
            if area_cd and signgu_cd:
                areas.append(ApiArea(area_cd=area_cd, signgu_cd=signgu_cd, label=label))

    if not areas:
        raise ValueError("area plan CSV에 수집 대상이 없습니다.")
    return areas


def collect_edges(
    *,
    areas: list[ApiArea],
    base_ym: str,
    limit_edges: int | None,
) -> list[dict]:
    edges: list[dict] = []

    for area in areas:
        print(f"\n[API 수집] {area.label} areaCd={area.area_cd}, signguCd={area.signgu_cd}")
        page_no = 1
        area_edges = 0

        while True:
            items, total_count = fetch_area_based_list(
                area_cd=area.area_cd,
                signgu_cd=area.signgu_cd,
                base_ym=base_ym,
                page_no=page_no,
            )
            if page_no == 1:
                print(f"  totalCount={total_count}")

            if not items:
                print(f"  [종료] 데이터 없음 page={page_no}")
                break

            for item in items:
                edges.append(item)
                area_edges += 1
                if limit_edges is not None and len(edges) >= limit_edges:
                    print(f"  [limit] 전체 edge {len(edges)}건 도달")
                    return edges

            if len(items) < PAGE_SIZE:
                print(f"  [종료] 마지막 페이지 page={page_no}")
                break

            page_no += 1
            time.sleep(REQUEST_INTERVAL_SECONDS)

        print(f"  수집 edge={area_edges}")

    return edges


def build_tar_counts(
    edges: list[dict],
    name_index: dict[str, list[Place]],
) -> tuple[Counter, Counter, list[dict]]:
    inbound: Counter[str] = Counter()
    outbound: Counter[str] = Counter()
    report_rows: list[dict] = []

    for item in edges:
        center = match_place(
            name=str(item.get("tAtsNm") or ""),
            region_name=str(item.get("areaNm") or ""),
            signgu_name=str(item.get("signguNm") or ""),
            name_index=name_index,
        )
        related = match_place(
            name=str(item.get("rlteTatsNm") or ""),
            region_name=str(item.get("rlteRegnNm") or ""),
            signgu_name=str(item.get("rlteSignguNm") or ""),
            name_index=name_index,
        )

        if center:
            outbound[center.content_id] += 1
        if related:
            inbound[related.content_id] += 1

        if center and related:
            status = "both"
        elif center:
            status = "center_only"
        elif related:
            status = "related_only"
        else:
            status = "unmatched"

        report_rows.append(
            {
                "baseYm": item.get("baseYm", ""),
                "tAtsNm": item.get("tAtsNm", ""),
                "areaNm": item.get("areaNm", ""),
                "signguNm": item.get("signguNm", ""),
                "rlteTatsNm": item.get("rlteTatsNm", ""),
                "rlteRegnNm": item.get("rlteRegnNm", ""),
                "rlteSignguNm": item.get("rlteSignguNm", ""),
                "rlteCtgryLclsNm": item.get("rlteCtgryLclsNm", ""),
                "rlteCtgryMclsNm": item.get("rlteCtgryMclsNm", ""),
                "rlteRank": item.get("rlteRank", ""),
                "center_match_id": center.content_id if center else "",
                "related_match_id": related.content_id if related else "",
                "match_status": status,
            }
        )

    return inbound, outbound, report_rows


def write_tar_raw(
    path: Path,
    places: list[Place],
    inbound: Counter[str],
    outbound: Counter[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    place_by_id = {place.content_id: place for place in places}
    all_ids = sorted(set(inbound) | set(outbound))

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "content_id",
                "name",
                "inbound_count",
                "outbound_count",
                "tar_rlte_raw",
            ],
        )
        writer.writeheader()

        for content_id in all_ids:
            inbound_count = int(inbound.get(content_id, 0))
            outbound_count = int(outbound.get(content_id, 0))
            tar_rlte_raw = inbound_count + 0.5 * outbound_count
            place = place_by_id.get(content_id)
            writer.writerow(
                {
                    "content_id": content_id,
                    "name": place.name if place else "",
                    "inbound_count": inbound_count,
                    "outbound_count": outbound_count,
                    "tar_rlte_raw": tar_rlte_raw,
                }
            )


def write_match_report(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "baseYm",
                "tAtsNm",
                "areaNm",
                "signguNm",
                "rlteTatsNm",
                "rlteRegnNm",
                "rlteSignguNm",
                "rlteCtgryLclsNm",
                "rlteCtgryMclsNm",
                "rlteRank",
                "center_match_id",
                "related_match_id",
                "match_status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_unmatched_top(path: Path, report_rows: list[dict]) -> None:
    counter: Counter[tuple[str, str, str, str]] = Counter()

    for row in report_rows:
        if not row["center_match_id"]:
            counter[
                (
                    str(row["tAtsNm"]),
                    str(row["areaNm"]),
                    str(row["signguNm"]),
                    "center",
                )
            ] += 1
        if not row["related_match_id"]:
            counter[
                (
                    str(row["rlteTatsNm"]),
                    str(row["rlteRegnNm"]),
                    str(row["rlteSignguNm"]),
                    "related",
                )
            ] += 1

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["name", "region", "signgu", "role", "appear_count"],
        )
        writer.writeheader()
        for (name, region, signgu, role), count in counter.most_common(200):
            writer.writerow(
                {
                    "name": name,
                    "region": region,
                    "signgu": signgu,
                    "role": role,
                    "appear_count": count,
                }
            )


def print_summary(edges: list[dict], report_rows: list[dict], tar_raw_path: Path) -> None:
    status_counts = Counter(row["match_status"] for row in report_rows)
    total = len(report_rows)
    print("\n[요약]")
    print(f"- API edge 수: {len(edges)}")
    print(f"- match status: {dict(status_counts)}")
    if total:
        both = status_counts.get("both", 0)
        usable = total - status_counts.get("unmatched", 0)
        print(f"- both matched ratio: {both / total:.2%}")
        print(f"- center or related matched ratio: {usable / total:.2%}")
    print(f"- tar raw: {tar_raw_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TarRlteTar API를 places.db와 매칭해 tar_rlte_raw CSV 생성"
    )
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="places.db 경로")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="출력 디렉터리")
    parser.add_argument("--base-ym", default="202503", help="기준연월 YYYYMM")
    parser.add_argument(
        "--area-plan",
        default=None,
        help="areaCd,signguCd,label 컬럼을 가진 CSV. 생략하면 서울 구로구 pilot",
    )
    parser.add_argument(
        "--limit-edges",
        type=int,
        default=None,
        help="전체 API relation row 수 제한. pilot용",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    db_path = Path(args.db)
    out_dir = Path(args.out_dir)
    tag = datetime.now(timezone.utc).strftime("%Y%m%d")

    places = load_places(db_path)
    name_index = build_name_index(places)
    print(f"[DB] active places={len(places)}")

    areas = read_area_plan(Path(args.area_plan) if args.area_plan else None)
    edges = collect_edges(
        areas=areas,
        base_ym=args.base_ym,
        limit_edges=args.limit_edges,
    )

    inbound, outbound, report_rows = build_tar_counts(edges, name_index)

    tar_raw_path = out_dir / f"tar_rlte_raw_{tag}.csv"
    report_path = out_dir / f"tar_rlte_match_report_{tag}.csv"
    unmatched_top_path = out_dir / f"tar_rlte_unmatched_top_{tag}.csv"

    write_tar_raw(tar_raw_path, places, inbound, outbound)
    write_match_report(report_path, report_rows)
    write_unmatched_top(unmatched_top_path, report_rows)

    print(f"[match report] {report_path}")
    print(f"[unmatched top] {unmatched_top_path}")
    print_summary(edges, report_rows, tar_raw_path)


if __name__ == "__main__":
    main()