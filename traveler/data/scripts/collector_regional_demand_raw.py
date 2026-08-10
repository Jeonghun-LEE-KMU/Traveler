"""
지역별 관광 서비스 수요 raw signal 수집기.

역할:
  1. 한국관광공사_지역별 관광 자원 수요 API 호출
  2. 지역별 관광 서비스 수요 지표값(tarSvcDemIxVal)을 가져온다.
  3. places.db의 active 장소 중 해당 지역 주소를 가진 장소에 값을 매핑한다.
  4. iconic_score_v2.py 입력용 regional_demand_raw CSV를 저장한다.

주의:
  이 API는 장소별 데이터가 아니라 지역별 데이터다.
  따라서 content_id 직접 매칭이 아니라 areaNm/signguNm을 places.address에 매핑한다.

  지역별 관광 서비스 수요는 API에서 정상 응답이 확인된 최신 가용 기준월인 202509를 Sprint 3-4 고정 snapshot month로 사용했다. 검색 품질 평가의 재현성을 위해 동일 기준월의 raw signal을 고정하여 Iconic Score v2를 계산했다.
  
"""

from __future__ import annotations

import argparse
import csv
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv


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

API_KEY = (
    os.getenv("REGIONAL_DEMAND_API_KEY")
    or os.getenv("TOURISM_RESOURCE_DEMAND_API_KEY")
    or os.getenv("KORSERVICE_API_KEY")
)

BASE_URL = "https://apis.data.go.kr/B551011/AreaTarResDemService"
ENDPOINT = "areaTarSvcDemList"

DEFAULT_BASE_YM = "202509"
DEFAULT_TAR_SVC_DEM_IX_CD = "11"  # 관광서비스수요 전체


@dataclass(frozen=True)
class AreaDemandTarget:
    area_cd: str
    signgu_cd: str
    area_nm: str
    signgu_nm: str
    label: str


@dataclass(frozen=True)
class Place:
    content_id: str
    name: str
    address: str
    area_code: str
    sigungu_code: str
    content_type_id: str


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
                COALESCE(CAST(sigungu_code AS TEXT), '') AS sigungu_code,
                COALESCE(CAST(content_type_id AS TEXT), '') AS content_type_id
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
            content_type_id=str(row["content_type_id"]),
        )
        for row in rows
    ]


def default_targets() -> list[AreaDemandTarget]:
    # 응답 구조 확인에 성공한 서울 구로구 pilot.
    return [
        AreaDemandTarget(
            area_cd="11",
            signgu_cd="11530",
            area_nm="서울특별시",
            signgu_nm="구로구",
            label="서울 구로구",
        )
    ]


def read_area_plan(path: Path | None) -> list[AreaDemandTarget]:
    if path is None:
        return default_targets()
    if not path.exists():
        raise FileNotFoundError(f"area plan CSV를 찾을 수 없습니다: {path}")

    targets: list[AreaDemandTarget] = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"areaCd", "signguCd", "areaNm", "signguNm"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                "area plan CSV에는 areaCd, signguCd, areaNm, signguNm 컬럼이 필요합니다."
            )

        for row in reader:
            area_cd = str(row.get("areaCd") or "").strip()
            signgu_cd = str(row.get("signguCd") or "").strip()
            area_nm = str(row.get("areaNm") or "").strip()
            signgu_nm = str(row.get("signguNm") or "").strip()
            label = str(row.get("label") or f"{area_nm} {signgu_nm}").strip()
            if area_cd and signgu_cd and area_nm and signgu_nm:
                targets.append(
                    AreaDemandTarget(
                        area_cd=area_cd,
                        signgu_cd=signgu_cd,
                        area_nm=area_nm,
                        signgu_nm=signgu_nm,
                        label=label,
                    )
                )

    if not targets:
        raise ValueError("area plan CSV에 수집 대상이 없습니다.")
    return targets


def fetch_regional_demand(
    target: AreaDemandTarget,
    *,
    base_ym: str,
    tar_svc_dem_ix_cd: str,
) -> dict:
    if not API_KEY:
        raise RuntimeError("REGIONAL_DEMAND_API_KEY 환경변수가 필요합니다.")

    url = f"{BASE_URL}/{ENDPOINT}"
    params = {
        "serviceKey": API_KEY,
        "MobileOS": "ETC",
        "MobileApp": "TravelCurator",
        "_type": "json",
        "numOfRows": 5,
        "pageNo": 1,
        "baseYm": base_ym,
        "areaCd": target.area_cd,
        "signguCd": target.signgu_cd,
        "tarSvcDemIxCd": tar_svc_dem_ix_cd,
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    header = data.get("response", {}).get("header", {})
    if str(header.get("resultCode")) != "0000":
        raise RuntimeError(f"regional demand API 오류: {header}")

    body = data.get("response", {}).get("body", {})
    items = body.get("items") or {}
    if isinstance(items, str):
        return {}

    item = items.get("item") if isinstance(items, dict) else None
    if isinstance(item, list):
        return item[0] if item else {}
    if isinstance(item, dict):
        return item
    return {}

#⭐️여기는 나중에 cron 자동화때 수정 필요
def region_matches(place: Place, target: AreaDemandTarget) -> bool:
    address = place.address

    # 시군구 값이 있는 API 응답은 반드시 광역지역 + 시군구가 모두 맞아야 한다.
    # 예: 서울특별시 구로구 수요값은 address에 "서울특별시"와 "구로구"가 모두 있어야 매핑
    if target.signgu_nm:
        return bool(
            target.area_nm
            and target.area_nm in address
            and target.signgu_nm in address
        )

    # 시군구가 없는 광역 단위 데이터일 때만 areaNm 단독 매칭 허용
    return bool(target.area_nm and target.area_nm in address)


def map_demand_to_places(
    places: list[Place],
    targets: list[AreaDemandTarget],
    demand_by_target: dict[AreaDemandTarget, float],
) -> tuple[list[dict], list[dict]]:
    raw_rows: list[dict] = []
    report_rows: list[dict] = []
    seen_content_ids: set[str] = set()

    for target in targets:
        demand_value = demand_by_target.get(target)
        matched_places = [
            place
            for place in places
            if demand_value is not None and region_matches(place, target)
        ]

        report_rows.append(
            {
                "areaCd": target.area_cd,
                "areaNm": target.area_nm,
                "signguCd": target.signgu_cd,
                "signguNm": target.signgu_nm,
                "regional_demand_raw": demand_value,
                "matched_place_count": len(matched_places),
                "status": "ok" if demand_value is not None else "missing_api_data",
            }
        )

        if demand_value is None:
            continue

        for place in matched_places:
            if place.content_id in seen_content_ids:
                continue
            seen_content_ids.add(place.content_id)
            raw_rows.append(
                {
                    "content_id": place.content_id,
                    "name": place.name,
                    "areaNm": target.area_nm,
                    "signguNm": target.signgu_nm,
                    "content_type_id": place.content_type_id,
                    "regional_demand_raw": demand_value,
                }
            )

    return raw_rows, report_rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="지역별 관광 서비스 수요를 places content_id별 raw CSV로 변환"
    )
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="places.db 경로")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="출력 디렉터리")
    parser.add_argument("--base-ym", default=DEFAULT_BASE_YM, help="기준연월 YYYYMM")
    parser.add_argument(
        "--tar-svc-dem-ix-cd",
        default=DEFAULT_TAR_SVC_DEM_IX_CD,
        help="관광 서비스 수요 지표 코드. 기본값 11=전체 관광서비스수요",
    )
    parser.add_argument(
        "--area-plan",
        default=None,
        help="areaCd,signguCd,areaNm,signguNm,label 컬럼을 가진 CSV. 생략하면 서울 구로구 pilot",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_path = Path(args.db)
    out_dir = Path(args.out_dir)
    tag = datetime.now(timezone.utc).strftime("%Y%m%d")

    places = load_places(db_path)
    targets = read_area_plan(Path(args.area_plan) if args.area_plan else None)
    print(f"[DB] active places={len(places)}")
    print(f"[target] area count={len(targets)}")

    demand_by_target: dict[AreaDemandTarget, float] = {}
    for target in targets:
        item = fetch_regional_demand(
            target,
            base_ym=args.base_ym,
            tar_svc_dem_ix_cd=args.tar_svc_dem_ix_cd,
        )
        if not item:
            print(f"[API] {target.label}: 데이터 없음")
            continue

        raw_value = item.get("tarSvcDemIxVal")
        if raw_value is None or str(raw_value).strip() == "":
            print(f"[API] {target.label}: tarSvcDemIxVal 없음")
            continue

        demand_by_target[target] = float(raw_value)
        print(
            f"[API] {target.label}: "
            f"{item.get('tarSvcDemIxNm')}={demand_by_target[target]}"
        )

    raw_rows, report_rows = map_demand_to_places(
        places,
        targets,
        demand_by_target,
    )

    raw_path = out_dir / f"regional_demand_raw_{tag}.csv"
    report_path = out_dir / f"regional_demand_match_report_{tag}.csv"

    write_csv(
        raw_path,
        raw_rows,
        [
            "content_id",
            "name",
            "areaNm",
            "signguNm",
            "content_type_id",
            "regional_demand_raw",
        ],
    )
    write_csv(
        report_path,
        report_rows,
        [
            "areaCd",
            "areaNm",
            "signguCd",
            "signguNm",
            "regional_demand_raw",
            "matched_place_count",
            "status",
        ],
    )

    print("\n[완료]")
    print(f"- raw: {raw_path}")
    print(f"- report: {report_path}")
    print(f"- raw row 수: {len(raw_rows)}")
    print(f"- report row 수: {len(report_rows)}")


if __name__ == "__main__":
    main()