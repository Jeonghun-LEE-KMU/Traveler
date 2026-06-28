# ============================================================
# TourAPI 데이터 수집 스크립트 (테스트 + 5,000건 적재용)
# 프로젝트: LLM-RAG 트래블 큐레이터
# 작성자: gayoung
# Sprint 2 | 2026-06-22
#
# 용도:
#   - 소량 테스트: NUM_OF_ROWS_TEST = 10 으로 실행
#   - 5,000건 적재: TYPE_TARGETS 주석 참고해서 전환 -> api 사용량 한도초과 때문에 현재 약 1400건정도 적재
#
# 변경사항 (Sprint 1 대비):
#   - collector_test.py 에서 api로 데이터 호출 -> preprocessing_test.py 불러서 전처리 -> collector_test.py 로 가서 전저리된 데이터 적재
#   - 5000건 적재 완료되고, 데이터 적재 자동화 하면 collector.py와 preprocessing.py로 다시 깔끔하게 짤 예정
#   - DB_PATH / SCHEMA_PATH: scripts/ 기준 경로로 수정
#   - iconic_score: readcount → 한국관광 100선 CSV 기반 rank_score로 교체
#   - 5,000건 분배 기준 추가 (타입별 목표 건수 + 지역별 가중치)
#   - 중복 방지: seen_ids로 같은 content_id 중복 수집 방지
#   - 페이지네이션: 목표 건수 달성까지 자동 페이지 넘김
#   - 429 에러 자동 재시도 (5초, 10초, 15초 대기)
# ============================================================

import csv
import os
import sqlite3
import time

import requests
from dotenv import load_dotenv
from preprocessing_test import preprocess, preprocess_intro

load_dotenv()
API_KEY = os.getenv("KORSERVICE_API_KEY")

# ── 경로 설정 (scripts/ 기준) ──────────────────────────────────
BASE_DIR     = os.path.dirname(__file__)          # traveler/data/scripts/
DATA_DIR     = os.path.join(BASE_DIR, "..")       # traveler/data/

BASE_URL     = "https://apis.data.go.kr/B551011/KorService2"
PET_BASE_URL = "http://apis.data.go.kr/B551011/KorService2"  # 반려동물 API는 http

DB_PATH          = os.path.join(DATA_DIR, "places.db")
SCHEMA_PATH      = os.path.join(DATA_DIR, "schema.sql")
TOURISM_100_CSV  = os.path.join(DATA_DIR, "korea_tourism_100_2025_2026.csv")

# ── 수집 건수 설정 ─────────────────────────────────────────────
# [테스트용] 타입당 10건
# TYPE_TARGETS = {
#     "12": 10,   # 관광지
#     "14": 10,   # 문화시설
#     "15": 10,   # 축제/행사
#     "38": 10,   # 쇼핑
#     "39": 10,   # 음식점+카페
# }

# [5,000건 적재용]
TYPE_TARGETS = {
    "12": 1500,   # 관광지
    "14": 1000,   # 문화시설
    "15": 500,    # 축제/행사
    "38": 500,    # 쇼핑
    "39": 1500,   # 음식점+카페
}

TYPE_NAMES = {
    "12": "관광지",
    "14": "문화시설",
    "15": "축제/행사",
    "38": "쇼핑",
    "39": "음식점+카페",
}

# TourAPI areaCode 기반 지역별 가중치
AREA_WEIGHTS = [
    ("1",  0.20),   # 서울
    ("6",  0.15),   # 부산
    ("39", 0.15),   # 제주
    ("31", 0.15),   # 경기
    ("",   0.35),   # 기타 전국 (area_code 미지정)
]


# ── 429 재시도 래퍼 ───────────────────────────────────────────
def fetch_with_retry(url: str, params: dict, max_retry: int = 3) -> requests.Response:
    """
    429 Too Many Requests 에러 시 자동 재시도.
    1차: 5초 대기, 2차: 10초 대기, 3차: 15초 대기
    """
    for attempt in range(max_retry):
        try:
            res = requests.get(url, params=params, timeout=10)
            res.raise_for_status()
            return res
        except Exception as e:
            if "429" in str(e):
                wait = 5 * (attempt + 1)
                print(f"    [429] 요청 한도 초과 → {wait}초 대기 후 재시도 ({attempt + 1}/{max_retry})")
                time.sleep(wait)
            else:
                raise e
    raise Exception("최대 재시도 횟수 초과")


# ── 한국관광 100선 로드 ───────────────────────────────────────
def load_tourism_100_ranks(csv_path: str) -> dict:
    """한국관광 100선 CSV → {place_name: rank} 딕셔너리 반환"""
    ranks = {}
    if not os.path.exists(csv_path):
        print(f"[경고] 한국관광 100선 CSV 없음: {csv_path} → rank_score=0 처리")
        return ranks
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ranks[row["place_name"].strip()] = int(row["rank"])
    print(f"[100선] {len(ranks)}건 로드 완료")
    return ranks


def calc_rank_score(name: str, ranks: dict, rank_max: int = 100) -> float:
    """장소명 기반 rank_score 계산."""
    rank = ranks.get(name.strip())
    if rank is None:
        return 0.0
    return round((rank_max + 1 - rank) / rank_max, 4)


# ── DB 초기화 ─────────────────────────────────────────────────
def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        conn.executescript(f.read())
    conn.commit()
    print(f"[DB] 초기화 완료: {DB_PATH}")
    return conn


# ── TourAPI 호출 ──────────────────────────────────────────────
def fetch_area_based_list(
    content_type_id: str,
    num_of_rows: int = 100,
    page_no: int = 1,
    area_code: str = "",
) -> list:
    """areaBasedList2 API 호출 — 관광지 목록"""
    url = f"{BASE_URL}/areaBasedList2"
    params = {
        "serviceKey":    API_KEY,
        "numOfRows":     num_of_rows,
        "pageNo":        page_no,
        "MobileOS":      "ETC",
        "MobileApp":     "TravelCurator",
        "_type":         "json",
        "arrange":       "O",
        "contentTypeId": content_type_id,
    }
    if area_code:
        params["areaCode"] = area_code

    res = fetch_with_retry(url, params)
    data = res.json()
    items = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])
    return items if isinstance(items, list) else [items]


def fetch_detail_common(content_id: str) -> dict:
    """detailCommon2 API 호출 — overview, homepage 등"""
    url = f"{BASE_URL}/detailCommon2"
    params = {
        "serviceKey": API_KEY,
        "MobileOS":   "ETC",
        "MobileApp":  "TravelCurator",
        "_type":      "json",
        "contentId":  content_id,
    }
    res = fetch_with_retry(url, params)
    data = res.json()
    items = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])
    return items[0] if items else {}


def fetch_detail_intro(content_id: str, content_type_id: str) -> dict:
    """detailIntro2 API 호출 — 타입별 상세 정보"""
    url = f"{BASE_URL}/detailIntro2"
    params = {
        "serviceKey":    API_KEY,
        "MobileOS":      "ETC",
        "MobileApp":     "TravelCurator",
        "_type":         "json",
        "contentId":     content_id,
        "contentTypeId": content_type_id,
    }
    res = fetch_with_retry(url, params)
    data = res.json()
    items = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])
    return items[0] if items else {}


def fetch_pet_tour_list(num_of_rows: int = 100, page_no: int = 1) -> list:
    """detailPetTour2 API 호출 — 반려동물 동반 가능 장소 (http만 지원)"""
    url = f"{PET_BASE_URL}/detailPetTour2"
    params = {
        "serviceKey": API_KEY,
        "numOfRows":  num_of_rows,
        "pageNo":     page_no,
        "MobileOS":   "ETC",
        "MobileApp":  "TravelCurator",
        "_type":      "json",
    }
    res = requests.get(url, params=params, timeout=10)
    res.raise_for_status()
    data = res.json()
    items = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])
    return items if isinstance(items, list) else [items]


# ── 파싱 & INSERT ─────────────────────────────────────────────
def parse_and_insert(
    conn: sqlite3.Connection,
    clean: dict,
    intro_parsed: dict,
    iconic_score: float,
):
    """전처리된 데이터를 places 테이블에 INSERT OR REPLACE"""
    row = {
        **clean,
        "iconic_score": iconic_score,
        "parking": "", "rest_date": "", "use_time": "", "baby_carriage": "",
        "use_fee": "", "use_time_culture": "", "rest_date_culture": "", "parking_culture": "",
        "event_start_date": "", "event_end_date": "", "event_place": "",
        "play_time": "", "use_time_festival": "", "program": "",
        "parking_shopping": "", "rest_date_shopping": "", "open_time": "",
        "sale_item": "", "restroom": "",
        "parking_food": "", "rest_date_food": "", "open_time_food": "",
        "first_menu": "", "treat_menu": "", "kids_facility": "",
    }
    row.update(intro_parsed)

    conn.execute("""
        INSERT OR REPLACE INTO places (
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
            first_menu, treat_menu, kids_facility
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
            :first_menu, :treat_menu, :kids_facility
        )
    """, row)


# ── 반려동물 UPDATE ───────────────────────────────────────────
def update_pet_allowed(conn: sqlite3.Connection):
    """detailPetTour2로 반려동물 가능 장소 수집 → places UPDATE"""
    print("\n[반려동물] 동반 가능 장소 수집 시작...")

    existing_ids = set(
        row[0] for row in conn.execute("SELECT content_id FROM places").fetchall()
    )
    print(f"  → 현재 DB 장소 수: {len(existing_ids)}건")

    pet_data = {}
    page_no = 1
    while True:
        items = fetch_pet_tour_list(num_of_rows=100, page_no=page_no)
        if not items:
            break
        for item in items:
            cid       = str(item.get("contentid", ""))
            etc_info  = item.get("etcAcmpyInfo", "") or ""
            psbl_cpam = item.get("acmpyPsblCpam", "") or ""
            pet_data[cid] = {
                "pet_acmpy_type": item.get("acmpyTypeCd", "") or "",
                "pet_condition":  item.get("acmpyNeedMtr", "") or "",
                "pet_info":       f"{psbl_cpam}\n{etc_info}".strip(),
            }
        if len(items) < 100:
            break
        page_no += 1
        time.sleep(0.3)

    matched = existing_ids & set(pet_data.keys())
    print(f"  → 수집: {len(pet_data)}건 / DB 매칭: {len(matched)}건")

    for cid in matched:
        d = pet_data[cid]
        conn.execute("""
            UPDATE places SET
                is_pet_allowed = 1,
                pet_acmpy_type = ?,
                pet_condition  = ?,
                pet_info       = ?
            WHERE content_id = ?
        """, [d["pet_acmpy_type"], d["pet_condition"], d["pet_info"], cid])

    conn.commit()
    print("  [반려동물] 업데이트 완료!")


# ── 타입+지역 분배 수집 ───────────────────────────────────────
def collect_by_type_and_area(
    conn: sqlite3.Connection,
    type_id: str,
    target: int,
    ranks: dict,
):
    type_name = TYPE_NAMES[type_id]
    print(f"\n[수집 시작] {type_name} (typeId={type_id}) 목표: {target}건")

    #변경 — DB에 이미 있는 content_id를 seen_ids에 미리 로드
    collected = 0
    cur = conn.execute(
        "SELECT content_id FROM places WHERE content_type_id = ?", (type_id,)
    )
    seen_ids = set(row[0] for row in cur.fetchall())
    print(f"  [{type_name}] 기존 적재 건수: {len(seen_ids)}건 (스킵 예정)")

    for area_code, weight in AREA_WEIGHTS:
        area_target = max(1, round(target * weight))
        area_label  = area_code if area_code else "기타전국"

        if collected >= target:
            break

        area_target = min(area_target, target - collected)
        print(f"  [{type_name}] 지역={area_label} 목표={area_target}건")

        page_no    = 1
        area_count = 0

        while area_count < area_target:
            need        = area_target - area_count
            num_of_rows = min(need, 100)

            try:
                items = fetch_area_based_list(
                    content_type_id=type_id,
                    num_of_rows=num_of_rows,
                    page_no=page_no,
                    area_code=area_code,
                )
            except Exception as e:
                print(f"    [오류] 목록 API 실패 (page={page_no}): {e}")
                break

            if not items:
                print(f"    [종료] 더 이상 데이터 없음 (page={page_no})")
                break

            for item in items:
                cid = str(item.get("contentid", ""))
                if not cid or cid in seen_ids:
                    continue
                seen_ids.add(cid)

                try:
                    detail       = fetch_detail_common(cid)
                    intro        = fetch_detail_intro(cid, type_id)
                    clean        = preprocess(item, detail, intro)
                    intro_parsed = preprocess_intro(intro, type_id)
                    score        = calc_rank_score(clean["name"], ranks)

                    parse_and_insert(conn, clean, intro_parsed, score)
                    area_count += 1
                    collected  += 1

                    if area_count % 10 == 0:
                        conn.commit()
                        print(f"    [{area_label}] {area_count}/{area_target}건 저장 중...")

                    time.sleep(1.0)

                except Exception as e:
                    print(f"    [오류] content_id={cid}: {e}")
                    continue

                if area_count >= area_target:
                    break

            page_no += 1
            if len(items) < num_of_rows:
                print(f"    [종료] 마지막 페이지 (page={page_no - 1})")
                break

        conn.commit()
        print(f"  [{type_name}] 지역={area_label} 완료: {area_count}건")

    print(f"[{type_name}] 타입 완료 — 수집: {collected}건")
    return collected


# ── 메인 ──────────────────────────────────────────────────────
def main():
    conn  = init_db()
    ranks = load_tourism_100_ranks(TOURISM_100_CSV)

    total = 0
    for type_id, target in TYPE_TARGETS.items():
        n     = collect_by_type_and_area(conn, type_id, target, ranks)
        total += n

    update_pet_allowed(conn)

    conn.close()
    print(f"\n[완료] 전체 수집 종료. 총 {total}건 적재. DB: {DB_PATH}")


if __name__ == "__main__":
    main()