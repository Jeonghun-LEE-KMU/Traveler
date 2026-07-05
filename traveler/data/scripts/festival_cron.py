"""
축제/행사 일일 수집 크론.

이 파일을 mobility_rag/traveler/data/scripts/festival_cron.py로 복사해서 사용한다.

역할:
0. TourAPI는 일일 한도가 1000이기 때문에 행사 목록은 모두 조회 후 상세 정보는 상위 100건만 불러오도록 설계(상위 n건은 변경 가능하다)
0-1. 우선 상위 100건을 순위 매기는 방법은, 정보가 잘 갖춰진 정도로 하며, 추후 iconic_score나 검색어 순위가 정리되면 해당 내용을 포함하도록 한다.
1. TourAPI에서 진행 중이거나 예정된 축제를 조회한다.
2. 신규 축제는 INSERT하고, 기존 축제는 UPSERT한다.
3. 변경되거나 다시 활성화된 축제만 embedded_at=NULL로 표시한다.
4. 종료일이 지난 축제는 삭제하지 않고 is_active=0으로 표시한다.

Pinecone upsert/delete는 embed_pipeline.py가 담당한다.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv
from preprocessing_test import preprocess, preprocess_intro


load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

API_KEY = os.getenv("KORSERVICE_API_KEY")
BASE_URL = "https://apis.data.go.kr/B551011/KorService2"

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = (BASE_DIR / ".." / "places.db").resolve()

KST = ZoneInfo("Asia/Seoul")
CONTENT_TYPE_FESTIVAL = "15"
PAGE_SIZE = 100
MAX_DETAIL_FETCHES = 100 #상위 100개만 저장
LOOKBACK_DAYS = 90 #과거에 시작했지만, 아직 끝나지 않은 축제를 찾기 위한 기간
LOOKAHEAD_DAYS = 180 #앞으로 시작 예정인 축제 기간
REQUEST_INTERVAL_SECONDS = 0.3


@dataclass
class CollectionSummary:
    collected: int = 0
    failed: int = 0
    expired: int = 0


def today_yyyymmdd() -> str:
    """서버 시간대와 관계없이 한국 날짜를 반환한다."""
    return datetime.now(KST).strftime("%Y%m%d")


def date_offset_yyyymmdd(days: int) -> str:
    return (datetime.now(KST) + timedelta(days=days)).strftime("%Y%m%d")


def validate_environment(db_path: Path) -> None:
    if not API_KEY:
        raise RuntimeError(
            "KORSERVICE_API_KEY 환경변수가 설정되지 않았습니다."
        )
    if not db_path.exists():
        raise FileNotFoundError(f"places.db를 찾을 수 없습니다: {db_path}")

    with sqlite3.connect(db_path) as conn:
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(places)").fetchall()
        }

    required = {"embedded_at", "is_active"}
    missing = required - columns
    if missing:
        raise RuntimeError(
            "필요한 마이그레이션이 적용되지 않았습니다. "
            f"누락 컬럼: {sorted(missing)}"
        )


def fetch_with_retry(
    session: requests.Session,
    url: str,
    params: dict,
    max_retries: int = 3,
) -> dict:
    """
    TourAPI를 호출하고 JSON 응답까지 검증한다.

    HTTP 429/5xx 및 네트워크 오류만 재시도한다.
    TourAPI는 HTTP 200 안에 오류 코드를 담기도 하므로 resultCode도 검사한다.
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(url, params=params, timeout=(5, 30))

            if response.status_code == 429 or response.status_code >= 500:
                response.raise_for_status()

            response.raise_for_status()
            data = response.json()

            header = data.get("response", {}).get("header", {})
            result_code = str(header.get("resultCode", ""))
            if result_code not in {"0000", "0"}:
                raise RuntimeError(
                    "TourAPI 오류 "
                    f"code={result_code}, "
                    f"message={header.get('resultMsg', '알 수 없음')}"
                )

            return data

        except (
            requests.Timeout,
            requests.ConnectionError,
            requests.HTTPError,
        ) as error:
            retryable_status = (
                getattr(error.response, "status_code", None)
                if isinstance(error, requests.HTTPError)
                else None
            )
            retryable = (
                not isinstance(error, requests.HTTPError)
                or retryable_status == 429
                or (retryable_status is not None and retryable_status >= 500)
            )

            if not retryable or attempt == max_retries:
                raise

            wait_seconds = 5 * attempt
            logger.warning(
                "TourAPI 호출 실패: %s. %d초 후 재시도 (%d/%d)",
                error,
                wait_seconds,
                attempt,
                max_retries,
            )
            time.sleep(wait_seconds)

    raise RuntimeError("TourAPI 최대 재시도 횟수를 초과했습니다.")


def extract_items(data: dict) -> tuple[list[dict], int]:
    body = data.get("response", {}).get("body", {})
    raw_items = body.get("items") or {}
    items = raw_items.get("item", []) if isinstance(raw_items, dict) else []

    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        items = []

    try:
        total_count = int(body.get("totalCount", len(items)))
    except (TypeError, ValueError):
        total_count = len(items)

    return items, total_count


def fetch_festival_page(
    session: requests.Session,
    event_start_date: str,
    event_end_date: str,
    page_no: int,
) -> tuple[list[dict], int]:
    data = fetch_with_retry(
        session,
        f"{BASE_URL}/searchFestival2",
        {
            "serviceKey": API_KEY,
            "numOfRows": PAGE_SIZE,
            "pageNo": page_no,
            "MobileOS": "ETC",
            "MobileApp": "TravelCurator",
            "_type": "json",
            # A: 제목순. O를 사용하면 대표 이미지 없는 축제가 빠질 수 있다.
            "arrange": "A",
            "eventStartDate": event_start_date,
            "eventEndDate": event_end_date,
        },
    )
    return extract_items(data)


def fetch_detail_common(
    session: requests.Session,
    content_id: str,
) -> dict:
    data = fetch_with_retry(
        session,
        f"{BASE_URL}/detailCommon2",
        {
            "serviceKey": API_KEY,
            "MobileOS": "ETC",
            "MobileApp": "TravelCurator",
            "_type": "json",
            "contentId": content_id,
        },
    )
    items, _ = extract_items(data)
    return items[0] if items else {}


def fetch_detail_intro(
    session: requests.Session,
    content_id: str,
) -> dict:
    data = fetch_with_retry(
        session,
        f"{BASE_URL}/detailIntro2",
        {
            "serviceKey": API_KEY,
            "MobileOS": "ETC",
            "MobileApp": "TravelCurator",
            "_type": "json",
            "contentId": content_id,
            "contentTypeId": CONTENT_TYPE_FESTIVAL,
        },
    )
    items, _ = extract_items(data)
    return items[0] if items else {}


def build_festival_row(
    item: dict,
    detail: dict,
    intro: dict,
) -> dict:
    """기존 preprocessing_test.py를 재사용해 DB 저장 행을 만든다."""
    clean = preprocess(item, detail, intro)
    clean["content_type_id"] = CONTENT_TYPE_FESTIVAL

    row = {
        **clean,
        "iconic_score": 0.0,
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
    row.update(preprocess_intro(intro, CONTENT_TYPE_FESTIVAL))
    return row


def candidate_score(item: dict) -> tuple[int, str]:
    """
    상세 API를 호출할 후보의 데이터 완성도를 계산한다.

    TourAPI 목록에는 실제 유명도 지표가 없으므로, v1에서는 대표 이미지,
    주소, 좌표, 전화번호가 잘 갖춰진 축제를 먼저 선택한다. 동점이면 최근
    수정된 데이터를 우선한다.
    """
    score = 0
    if item.get("firstimage"):
        score += 3
    if item.get("addr1"):
        score += 1
    if item.get("mapx") and item.get("mapy"):
        score += 1
    if item.get("tel"):
        score += 1

    modified_time = str(item.get("modifiedtime") or "")
    return score, modified_time


def upsert_festival(conn: sqlite3.Connection, row: dict) -> None:
    """
    신규 축제는 INSERT하고 기존 축제는 UPDATE한다.

    modified_time이 바뀌었거나 비활성 축제가 다시 나타나면 embedded_at을
    NULL로 바꾼다. 다음 embed_pipeline 실행에서 해당 축제만 다시 upsert된다.
    """
    conn.execute(
        """
        INSERT INTO places (
            content_id, content_type_id, name, address,
            area_code, sigungu_code, lat, lng,
            overview, image_url, homepage, tel,
            category, modified_time, iconic_score,
            parking, rest_date, use_time, baby_carriage,
            use_fee, use_time_culture, rest_date_culture, parking_culture,
            event_start_date, event_end_date, event_place,
            play_time, use_time_festival, program,
            parking_shopping, rest_date_shopping, open_time, sale_item, restroom,
            parking_food, rest_date_food, open_time_food,
            first_menu, treat_menu, kids_facility,
            is_active, embedded_at
        ) VALUES (
            :content_id, :content_type_id, :name, :address,
            :area_code, :sigungu_code, :lat, :lng,
            :overview, :image_url, :homepage, :tel,
            :category, :modified_time, :iconic_score,
            :parking, :rest_date, :use_time, :baby_carriage,
            :use_fee, :use_time_culture, :rest_date_culture, :parking_culture,
            :event_start_date, :event_end_date, :event_place,
            :play_time, :use_time_festival, :program,
            :parking_shopping, :rest_date_shopping, :open_time, :sale_item, :restroom,
            :parking_food, :rest_date_food, :open_time_food,
            :first_menu, :treat_menu, :kids_facility,
            1, NULL
        )
        ON CONFLICT(content_id) DO UPDATE SET
            content_type_id = excluded.content_type_id,
            name = excluded.name,
            address = excluded.address,
            area_code = excluded.area_code,
            sigungu_code = excluded.sigungu_code,
            lat = excluded.lat,
            lng = excluded.lng,
            overview = excluded.overview,
            image_url = excluded.image_url,
            homepage = excluded.homepage,
            tel = excluded.tel,
            category = excluded.category,
            event_start_date = excluded.event_start_date,
            event_end_date = excluded.event_end_date,
            event_place = excluded.event_place,
            play_time = excluded.play_time,
            use_time_festival = excluded.use_time_festival,
            program = excluded.program,
            embedded_at = CASE
                WHEN places.modified_time IS NOT excluded.modified_time
                  OR places.is_active = 0
                THEN NULL
                ELSE places.embedded_at
            END,
            modified_time = excluded.modified_time,
            is_active = 1
        """,
        row,
    )


def collect_festivals(
    conn: sqlite3.Connection,
    session: requests.Session,
) -> CollectionSummary:
    """
    최근 1년 사이에 시작했거나 앞으로 1년 이내 시작할 축제를 조회한다.

    조회 후 event_end_date가 오늘보다 이전인 항목은 저장하지 않는다.
    """
    summary = CollectionSummary()
    today = today_yyyymmdd()
    search_start = date_offset_yyyymmdd(-LOOKBACK_DAYS)
    search_end = date_offset_yyyymmdd(LOOKAHEAD_DAYS)

    logger.info(
        "축제 수집 시작: 조회 범위 %s~%s, 활성 기준일 %s",
        search_start,
        search_end,
        today,
    )

    seen_ids: set[str] = set()
    candidates: list[dict] = []
    page_no = 1

    # 1단계: 상세 API를 호출하지 않고 목록만 전체 조회한다.
    while True:
        items, total_count = fetch_festival_page(
            session,
            event_start_date=search_start,
            event_end_date=search_end,
            page_no=page_no,
        )
        if not items:
            break

        for item in items:
            content_id = str(item.get("contentid") or "").strip()
            if not content_id or content_id in seen_ids:
                continue
            seen_ids.add(content_id)

            # searchFestival2 목록에는 과거에 종료된 축제도 포함될 수 있다.
            # 상세 API를 호출하기 전에 목록의 종료일로 먼저 제외해 호출량을 아낀다.
            list_end_date = str(item.get("eventenddate") or "").replace("-", "")
            if (
                len(list_end_date) == 8
                and list_end_date.isdigit()
                and list_end_date < today
            ):
                continue

            candidates.append(item)

        logger.info(
            "축제 목록 수집 중: page=%d, 후보=%d",
            page_no,
            len(candidates),
        )

        if page_no * PAGE_SIZE >= total_count or len(items) < PAGE_SIZE:
            break
        page_no += 1

    # 2단계: 데이터가 잘 갖춰진 후보 중 상위 100개만 상세 조회한다.
    candidates.sort(key=candidate_score, reverse=True)
    selected_candidates = candidates[:MAX_DETAIL_FETCHES]
    logger.info(
        "상세 조회 대상 선택: 전체 후보=%d, 선택=%d",
        len(candidates),
        len(selected_candidates),
    )

    for item in selected_candidates:
        content_id = str(item.get("contentid") or "").strip()

        try:
            detail = fetch_detail_common(session, content_id)
            intro = fetch_detail_intro(session, content_id)
            row = build_festival_row(item, detail, intro)

            # 조회 범위에 과거 축제가 섞여 들어오면 저장 단계에서 제외한다.
            event_end_date = row.get("event_end_date") or ""
            if event_end_date and event_end_date < today:
                continue

            upsert_festival(conn, row)
            summary.collected += 1

        except Exception:
            summary.failed += 1
            logger.exception("축제 처리 실패: content_id=%s", content_id)

        time.sleep(REQUEST_INTERVAL_SECONDS)

    conn.commit()
    return summary


def deactivate_expired_festivals(
    conn: sqlite3.Connection,
    today: str,
) -> int:
    """
    종료된 축제를 비활성화한다.

    embedded_at은 그대로 둔다. '비활성 + embedded_at 존재' 상태를
    embed_pipeline.py가 Pinecone delete 대기 상태로 해석한다.
    """
    cursor = conn.execute(
        """
        UPDATE places
        SET is_active = 0
        WHERE content_type_id = ?
          AND is_active = 1
          AND length(event_end_date) = 8
          AND event_end_date NOT GLOB '*[^0-9]*'
          AND event_end_date < ?
        """,
        (CONTENT_TYPE_FESTIVAL, today),
    )
    conn.commit()
    return cursor.rowcount


def run(db_path: Path = DEFAULT_DB_PATH) -> CollectionSummary:
    validate_environment(db_path)

    with requests.Session() as session:
        with sqlite3.connect(db_path, timeout=30) as conn:
            summary = collect_festivals(conn, session)
            summary.expired = deactivate_expired_festivals(
                conn,
                today_yyyymmdd(),
            )

    logger.info(
        "축제 크론 완료: 저장=%d, 실패=%d, 만료=%d",
        summary.collected,
        summary.failed,
        summary.expired,
    )
    return summary


if __name__ == "__main__":
    run()
