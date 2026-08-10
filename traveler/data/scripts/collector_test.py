"""
TourAPI 일반 장소 수집기.

기본 실행은 기존 전국 가중치 수집을 유지한다.
통합 테스트나 특정 지역 보강이 필요할 때만 CLI 옵션으로
areaCode, sigunguCode, 주소 지역명, 콘텐츠 유형을 제한한다.

예시:
  전국 기본 수집
    python3 traveler/data/scripts/collector_test.py

  경주시 문화시설·음식점 각각 30건 신규 수집
    python3 traveler/data/scripts/collector_test.py \
      --area-code 35 \
      --sigungu-code 2 \
      --region-name 경주시 \
      --content-types 14 39 \
      --limit-per-type 30

저장 정책:
  - content_id 기준 UPSERT
  - Iconic Score는 별도 ETL이 소유하므로 수정하지 않음
  - 임베딩에 사용되는 내용이 변경된 경우에만 embedded_at=NULL
  - 신규 장소는 schema.sql 기본값에 따라 is_active=1,
    iconic_score=0, embedded_at=NULL 상태로 저장
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import time
from collections.abc import Sequence

import requests
from dotenv import load_dotenv
from preprocessing_test import preprocess, preprocess_intro


load_dotenv()

API_KEY = os.getenv("KORSERVICE_API_KEY")

# ── 경로 설정 (scripts/ 기준) ──────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "..")

BASE_URL = "https://apis.data.go.kr/B551011/KorService2"
PET_BASE_URL = "http://apis.data.go.kr/B551011/KorService2"

DB_PATH = os.path.join(DATA_DIR, "places.db")
SCHEMA_PATH = os.path.join(DATA_DIR, "schema.sql")

# ── 기본 수집 설정 ──────────────────────────────────────────────
# CLI 옵션이 없을 때만 사용하는 전국 수집 목표다.
DEFAULT_TYPE_TARGETS = {
    "14": 150,
    #"39": 150,
    "12": 250,
}

RUN_PET_UPDATE = False
PAGE_SIZE = 100
REQUEST_INTERVAL_SECONDS = 1.0

TYPE_NAMES = {
    "12": "관광지",
    "14": "문화시설",
    #"15": "축제/행사",
    #"38": "쇼핑",
    #"39": "음식점+카페",
}

# 합계가 반드시 1.0이 되도록 유지한다.
# 마지막 전국 구간은 앞 지역에서 목표를 채우지 못한 경우의 보충 역할도 한다.
DEFAULT_AREA_WEIGHTS = [
    #("1", 0.20),   # 서울
    #("6", 0.15),   # 부산
    #("39", 0.15),  # 제주
    #("31", 0.15),  # 경기
    #("", 0.35),    # 기타 및 전국 보충
    ("8", 0.16),   # 세종
    ("3", 0.14),   # 대전
    ("33", 0.14),  # 충북
    ("37", 0.14),  # 전북
    ("5", 0.12),   # 광주
    ("34", 0.12),  # 충남
    ("7", 0.10),   # 울산
    ("36", 0.08),  # 경남
]

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TourAPI 장소를 수집해 SQLite places에 UPSERT",
    )
    parser.add_argument(
        "--area-code",
        help="TourAPI areaCode. 생략하면 기존 전국 가중치 수집",
    )
    parser.add_argument(
        "--sigungu-code",
        help="TourAPI sigunguCode. --area-code와 함께 사용",
    )
    parser.add_argument(
        "--region-name",
        help="저장 전 주소에 반드시 포함돼야 할 문자열. 예: 경주시",
    )
    parser.add_argument(
        "--content-types",
        nargs="+",
        choices=sorted(TYPE_NAMES),
        help="수집할 contentTypeId. 예: --content-types 14 39",
    )
    parser.add_argument(
        "--limit-per-type",
        type=int,
        help="선택한 각 콘텐츠 유형의 신규 수집 목표",
    )
    parser.add_argument(
        "--run-pet-update",
        action="store_true",
        help="일반 장소 수집 후 반려동물 정보를 갱신",
    )
    args = parser.parse_args()

    if args.sigungu_code and not args.area_code:
        parser.error("--sigungu-code는 --area-code와 함께 사용해야 합니다.")
    if args.region_name and not args.area_code:
        parser.error(
            "--region-name을 사용할 때는 API 호출량 보호를 위해 "
            "--area-code도 지정해야 합니다."
        )
    if args.limit_per_type is not None and args.limit_per_type <= 0:
        parser.error("--limit-per-type은 1 이상의 정수여야 합니다.")

    return args


def resolve_type_targets(args: argparse.Namespace) -> dict[str, int]:
    if not args.content_types:
        if args.limit_per_type is not None:
            raise ValueError(
                "--limit-per-type을 지정하려면 --content-types도 필요합니다."
            )
        return dict(DEFAULT_TYPE_TARGETS)

    targets: dict[str, int] = {}
    for content_type_id in args.content_types:
        if args.limit_per_type is not None:
            targets[content_type_id] = args.limit_per_type
        elif content_type_id in DEFAULT_TYPE_TARGETS:
            targets[content_type_id] = DEFAULT_TYPE_TARGETS[content_type_id]
        else:
            raise ValueError(
                f"type={content_type_id}의 기본 목표가 없습니다. "
                "--limit-per-type을 함께 지정하세요."
            )
    return targets


def validate_environment() -> None:
    if not API_KEY:
        raise RuntimeError(
            "KORSERVICE_API_KEY 환경변수가 설정되지 않았습니다."
        )
    if not os.path.exists(SCHEMA_PATH):
        raise FileNotFoundError(f"schema.sql을 찾을 수 없습니다: {SCHEMA_PATH}")


# ── 요청 재시도 ────────────────────────────────────────────────
def fetch_with_retry(
    url: str,
    params: dict,
    max_retry: int = 5,
) -> requests.Response:
    """
    429, 일시적 5xx, timeout/connection 오류를 재시도한다.

    4xx 중 인증키·파라미터 오류는 재시도해도 해결되지 않으므로 즉시 실패한다.
    """
    waits = [2, 5, 10, 20, 30]
    last_error: Exception | None = None

    for attempt in range(max_retry):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=(5, 30),
            )
            if response.status_code in RETRYABLE_STATUS_CODES:
                raise requests.HTTPError(
                    f"{response.status_code} temporary API error",
                    response=response,
                )
            response.raise_for_status()
            return response

        except requests.RequestException as error:
            last_error = error
            status_code = getattr(error.response, "status_code", None)
            retryable = (
                status_code is None
                or status_code in RETRYABLE_STATUS_CODES
            )
            if not retryable:
                raise

            if attempt == max_retry - 1:
                break

            wait = waits[min(attempt, len(waits) - 1)]
            print(
                "    [재시도] TourAPI 요청 실패: "
                f"{error} ({wait}초 후 {attempt + 2}/{max_retry})"
            )
            time.sleep(wait)

    raise RuntimeError(
        f"TourAPI 요청이 {max_retry}회 연속 실패했습니다: {last_error}"
    )


# ── DB 초기화 ─────────────────────────────────────────────────
def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    with open(SCHEMA_PATH, "r", encoding="utf-8") as schema_file:
        conn.executescript(schema_file.read())
    conn.commit()
    print(f"[DB] 초기화 완료: {DB_PATH}")
    return conn


def extract_items(data: dict) -> list[dict]:
    body = data.get("response", {}).get("body", {})
    raw_items = body.get("items") or {}
    items = raw_items.get("item", []) if isinstance(raw_items, dict) else []

    if isinstance(items, dict):
        return [items]
    if isinstance(items, list):
        return items
    return []


# ── TourAPI 호출 ──────────────────────────────────────────────
def fetch_area_based_list(
    content_type_id: str,
    *,
    num_of_rows: int = PAGE_SIZE,
    page_no: int = 1,
    area_code: str = "",
    sigungu_code: str = "",
) -> list[dict]:
    """areaBasedList2 목록을 조회한다."""
    url = f"{BASE_URL}/areaBasedList2"
    params = {
        "serviceKey": API_KEY,
        "numOfRows": num_of_rows,
        "pageNo": page_no,
        "MobileOS": "ETC",
        "MobileApp": "TravelCurator",
        "_type": "json",
        "arrange": "C",
        "contentTypeId": content_type_id,
    }
    if area_code:
        params["areaCode"] = area_code
    if sigungu_code:
        params["sigunguCode"] = sigungu_code

    response = fetch_with_retry(url, params)
    return extract_items(response.json())


def fetch_detail_common(content_id: str) -> dict:
    """detailCommon2 API에서 overview, homepage 등을 조회한다."""
    url = f"{BASE_URL}/detailCommon2"
    params = {
        "serviceKey": API_KEY,
        "MobileOS": "ETC",
        "MobileApp": "TravelCurator",
        "_type": "json",
        "contentId": content_id,
    }
    response = fetch_with_retry(url, params)
    items = extract_items(response.json())
    return items[0] if items else {}


def fetch_detail_intro(content_id: str, content_type_id: str) -> dict:
    """detailIntro2 API에서 콘텐츠 유형별 상세 정보를 조회한다."""
    url = f"{BASE_URL}/detailIntro2"
    params = {
        "serviceKey": API_KEY,
        "MobileOS": "ETC",
        "MobileApp": "TravelCurator",
        "_type": "json",
        "contentId": content_id,
        "contentTypeId": content_type_id,
    }
    response = fetch_with_retry(url, params)
    items = extract_items(response.json())
    return items[0] if items else {}


def fetch_pet_tour_list(
    num_of_rows: int = PAGE_SIZE,
    page_no: int = 1,
) -> list[dict]:
    """detailPetTour2 API에서 반려동물 동반 장소를 조회한다."""
    url = f"{PET_BASE_URL}/detailPetTour2"
    params = {
        "serviceKey": API_KEY,
        "numOfRows": num_of_rows,
        "pageNo": page_no,
        "MobileOS": "ETC",
        "MobileApp": "TravelCurator",
        "_type": "json",
    }
    response = fetch_with_retry(url, params)
    return extract_items(response.json())


# ── 파싱 & UPSERT ─────────────────────────────────────────────
def parse_and_insert(
    conn: sqlite3.Connection,
    clean: dict,
    intro_parsed: dict,
) -> None:
    """
    신규 장소는 INSERT하고 기존 장소는 수집기가 소유하는 컬럼만 UPDATE한다.

    Iconic Score는 별도 ETL이 소유하므로 수정하지 않는다.
    임베딩 텍스트에 사용되는 내용이 달라진 경우에만 embedded_at을 NULL로
    변경해 embed_pipeline.py가 해당 장소를 다시 임베딩하도록 한다.
    """
    row = {
        **clean,
        "parking": "",
        "rest_date": "",
        "use_time": "",
        "baby_carriage": "",
        "use_fee": "",
        "use_time_culture": "",
        "rest_date_culture": "",
        "parking_culture": "",
        "event_start_date": "",
        "event_end_date": "",
        "event_place": "",
        "play_time": "",
        "use_time_festival": "",
        "program": "",
        "parking_shopping": "",
        "rest_date_shopping": "",
        "open_time": "",
        "sale_item": "",
        "restroom": "",
        "parking_food": "",
        "rest_date_food": "",
        "open_time_food": "",
        "first_menu": "",
        "treat_menu": "",
        "kids_facility": "",
    }
    row.update(intro_parsed)

    insert_columns = [
        "content_id",
        "content_type_id",
        "name",
        "address",
        "area_code",
        "sigungu_code",
        "lat",
        "lng",
        "overview",
        "image_url",
        "homepage",
        "tel",
        "category",
        "modified_time",
        "parking",
        "rest_date",
        "use_time",
        "baby_carriage",
        "use_fee",
        "use_time_culture",
        "rest_date_culture",
        "parking_culture",
        "event_start_date",
        "event_end_date",
        "event_place",
        "play_time",
        "use_time_festival",
        "program",
        "parking_shopping",
        "rest_date_shopping",
        "open_time",
        "sale_item",
        "restroom",
        "parking_food",
        "rest_date_food",
        "open_time_food",
        "first_menu",
        "treat_menu",
        "kids_facility",
    ]
    update_columns = [
        column
        for column in insert_columns
        if column != "content_id"
    ]
    embedding_columns = [
        "name",
        "address",
        "overview",
        "category",
        "parking",
        "rest_date",
        "use_time",
        "baby_carriage",
        "use_fee",
        "use_time_culture",
        "rest_date_culture",
        "parking_culture",
        "event_start_date",
        "event_end_date",
        "event_place",
        "play_time",
        "use_time_festival",
        "program",
        "parking_shopping",
        "rest_date_shopping",
        "open_time",
        "sale_item",
        "restroom",
        "parking_food",
        "rest_date_food",
        "open_time_food",
        "first_menu",
        "treat_menu",
        "kids_facility",
    ]

    column_sql = ", ".join(insert_columns)
    value_sql = ", ".join(f":{column}" for column in insert_columns)
    update_sql = ", ".join(
        f"{column} = excluded.{column}"
        for column in update_columns
    )
    embedding_changed_sql = " OR ".join(
        f"places.{column} IS NOT excluded.{column}"
        for column in embedding_columns
    )

    sql = f"""
        INSERT INTO places ({column_sql})
        VALUES ({value_sql})
        ON CONFLICT(content_id) DO UPDATE SET
            {update_sql},
            embedded_at = CASE
                WHEN {embedding_changed_sql}
                THEN NULL
                ELSE places.embedded_at
            END
    """
    conn.execute(sql, row)


# ── 반려동물 UPDATE ───────────────────────────────────────────
def update_pet_allowed(conn: sqlite3.Connection) -> None:
    print("\n[반려동물] 동반 가능 장소 수집 시작...")

    existing_ids = {
        row[0]
        for row in conn.execute("SELECT content_id FROM places").fetchall()
    }
    print(f"  → 현재 DB 장소 수: {len(existing_ids)}건")

    pet_data: dict[str, dict] = {}
    page_no = 1
    while True:
        items = fetch_pet_tour_list(
            num_of_rows=PAGE_SIZE,
            page_no=page_no,
        )
        if not items:
            break

        for item in items:
            content_id = str(item.get("contentid", ""))
            etc_info = item.get("etcAcmpyInfo", "") or ""
            possible_capacity = item.get("acmpyPsblCpam", "") or ""
            pet_data[content_id] = {
                "pet_acmpy_type": item.get("acmpyTypeCd", "") or "",
                "pet_condition": item.get("acmpyNeedMtr", "") or "",
                "pet_info": f"{possible_capacity}\n{etc_info}".strip(),
            }

        if len(items) < PAGE_SIZE:
            break
        page_no += 1
        time.sleep(0.3)

    matched_ids = existing_ids & set(pet_data)
    print(
        f"  → 수집: {len(pet_data)}건 / DB 매칭: {len(matched_ids)}건"
    )

    for content_id in matched_ids:
        data = pet_data[content_id]
        conn.execute(
            """
            UPDATE places SET
                is_pet_allowed = 1,
                pet_acmpy_type = ?,
                pet_condition = ?,
                pet_info = ?
            WHERE content_id = ?
            """,
            [
                data["pet_acmpy_type"],
                data["pet_condition"],
                data["pet_info"],
                content_id,
            ],
        )

    conn.commit()
    print("  [반려동물] 업데이트 완료!")


def item_address(item: dict) -> str:
    return " ".join(
        [
            str(item.get("addr1") or ""),
            str(item.get("addr2") or ""),
        ]
    ).strip()


# ── 타입+지역 분배 수집 ───────────────────────────────────────
def collect_by_type_and_area(
    conn: sqlite3.Connection,
    type_id: str,
    target: int,
    *,
    area_weights: Sequence[tuple[str, float]],
    sigungu_code: str = "",
    region_name: str = "",
) -> int:
    type_name = TYPE_NAMES[type_id]
    print(
        f"\n[수집 시작] {type_name} "
        f"(typeId={type_id}) 신규 목표: {target}건"
    )

    collected = 0
    existing_ids = {
        str(row[0])
        for row in conn.execute(
            "SELECT content_id FROM places WHERE content_type_id = ?",
            (type_id,),
        ).fetchall()
    }
    seen_ids = set(existing_ids)
    print(
        f"  [{type_name}] 기존 적재 건수: "
        f"{len(existing_ids)}건 (스킵 예정)"
    )

    for area_code, weight in area_weights:
        if collected >= target:
            break

        # 마지막 구간에는 반올림 오차와 앞 구간의 부족분을 모두 배정한다.
        is_last_area = (area_code, weight) == area_weights[-1]
        if is_last_area:
            area_target = target - collected
        else:
            area_target = max(1, round(target * weight))
            area_target = min(area_target, target - collected)

        area_label = (
            region_name
            or (
                f"area={area_code}, sigungu={sigungu_code}"
                if sigungu_code
                else area_code or "전국보충"
            )
        )
        print(f"  [{type_name}] 지역={area_label} 목표={area_target}건")

        page_no = 1
        area_count = 0

        while area_count < area_target:
            try:
                items = fetch_area_based_list(
                    content_type_id=type_id,
                    num_of_rows=PAGE_SIZE,
                    page_no=page_no,
                    area_code=area_code,
                    sigungu_code=sigungu_code,
                )
            except Exception as error:
                print(
                    f"    [오류] 목록 API 실패 "
                    f"(page={page_no}): {error}"
                )
                break

            if not items:
                print(
                    f"    [종료] 더 이상 데이터 없음 (page={page_no})"
                )
                break

            for item in items:
                content_id = str(item.get("contentid", ""))
                if not content_id or content_id in seen_ids:
                    continue

                # API 지역코드가 잘못 지정되거나 예상 밖 응답이 섞여도
                # 타깃 지역 이외의 장소를 저장하지 않는다.
                if (
                    region_name
                    and region_name not in item_address(item)
                ):
                    continue

                try:
                    detail = fetch_detail_common(content_id)
                    intro = fetch_detail_intro(content_id, type_id)
                    clean = preprocess(item, detail, intro)
                    intro_parsed = preprocess_intro(intro, type_id)

                    cleaned_address = str(clean.get("address") or "")
                    if (
                        region_name
                        and region_name not in cleaned_address
                    ):
                        print(
                            "    [건너뜀] 주소 불일치: "
                            f"content_id={content_id}, "
                            f"address={cleaned_address}"
                        )
                        continue

                    parse_and_insert(conn, clean, intro_parsed)

                    # 성공적으로 UPSERT한 뒤에만 seen 처리한다.
                    seen_ids.add(content_id)
                    area_count += 1
                    collected += 1

                    if area_count % 10 == 0:
                        conn.commit()
                        print(
                            f"    [{area_label}] "
                            f"{area_count}/{area_target}건 저장 중..."
                        )

                    time.sleep(REQUEST_INTERVAL_SECONDS)

                except Exception as error:
                    print(
                        f"    [오류] content_id={content_id}: {error}"
                    )
                    continue

                if area_count >= area_target or collected >= target:
                    break

            if area_count >= area_target or collected >= target:
                break

            if len(items) < PAGE_SIZE:
                print(
                    f"    [종료] 마지막 페이지 (page={page_no})"
                )
                break

            page_no += 1

        conn.commit()
        print(
            f"  [{type_name}] 지역={area_label} 완료: "
            f"{area_count}건"
        )

    print(f"[{type_name}] 타입 완료 — 신규 수집: {collected}건")
    return collected


def main() -> None:
    args = parse_args()
    validate_environment()
    type_targets = resolve_type_targets(args)

    if args.area_code:
        area_weights = [(args.area_code, 1.0)]
        scope = (
            args.region_name
            or f"area={args.area_code}, sigungu={args.sigungu_code or '-'}"
        )
    else:
        area_weights = DEFAULT_AREA_WEIGHTS
        scope = "전국 가중치"

    print(
        "[설정] "
        f"범위={scope}, "
        f"타입별 목표={type_targets}, "
        f"반려동물 갱신={args.run_pet_update or RUN_PET_UPDATE}"
    )

    conn = init_db()
    try:
        total = 0
        for type_id, target in type_targets.items():
            total += collect_by_type_and_area(
                conn,
                type_id,
                target,
                area_weights=area_weights,
                sigungu_code=args.sigungu_code or "",
                region_name=args.region_name or "",
            )

        if args.run_pet_update or RUN_PET_UPDATE:
            update_pet_allowed(conn)

        print(
            f"\n[완료] 전체 수집 종료. "
            f"신규 총 {total}건 적재. DB: {DB_PATH}"
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
