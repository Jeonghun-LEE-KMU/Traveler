# ============================================================
# 전처리 모듈 (collector_test.py에서 import해서 사용)
# 프로젝트: LLM-RAG 트래블 큐레이터
# 작성자: gayoung
# Sprint 2 | 2026-06-22
#
# 역할:
#   collector_test.py가 API에서 받아온 raw 데이터를
#   DB 적재 전에 정제해서 반환
#
# 전처리 항목:
#   1. HTML 태그 제거 (overview, address 등)
#   2. 줄바꿈(\n, \r) → 공백으로 정리
#   3. None → 빈 문자열 통일
#   4. 앞뒤 공백 제거 (strip)
#   5. overview 길이 제한 (Pinecone 토큰 초과 방지) -> 중요한 정보를 충분히 담지 못한다면, 제한 제거할 예정
#   6. 좌표(lat/lng) 0이면 None으로 처리
#   7. 주소 addr1 + addr2 합칠 때 중복 공백 제거
#
# 나중에 자동화 단계에서 preprocessing.py로 분리/확장 예정
# ============================================================

import re


# ── 설정 ──────────────────────────────────────────────────────
# OVERVIEW_MAX_LEN: Pinecone 적재 후 토큰 초과 시 추가 예정


# ── 텍스트 정제 함수들 ────────────────────────────────────────
def remove_html_tags(text: str) -> str:
    """HTML 태그 제거 (<br>, <p>, <b> 등)"""
    if not text:
        return ""
    return re.sub(r"<[^>]+>", " ", text)


def clean_text(text: str, max_len: int = None) -> str:
    """
    텍스트 정제 통합 함수.
    1. None → 빈 문자열
    2. HTML 태그 제거
    3. 줄바꿈 → 공백
    4. 중복 공백 제거
    5. 앞뒤 공백 제거
    6. 길이 제한 (max_len 지정 시)
    """
    if not text:
        return ""
    text = remove_html_tags(text)
    text = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
    text = re.sub(r" +", " ", text)
    text = text.strip()
    if max_len:
        text = text[:max_len]
    return text


def clean_address(addr1: str, addr2: str) -> str:
    """
    addr1 + addr2 합쳐서 주소 정제.
    중복 공백, 앞뒤 공백 제거.
    """
    addr1 = (addr1 or "").strip()
    addr2 = (addr2 or "").strip()
    combined = f"{addr1} {addr2}".strip()
    return re.sub(r" +", " ", combined)


def clean_coord(value) -> float:
    """
    위경도 정제.
    None, 0, 빈 문자열 → 0.0
    """
    try:
        v = float(value or 0)
        return v if v != 0 else 0.0
    except (ValueError, TypeError):
        return 0.0


# ── 메인 전처리 함수 ──────────────────────────────────────────
def preprocess(item: dict, detail: dict, intro: dict) -> dict:
    """
    API 응답(item, detail, intro)을 받아서 정제된 dict 반환.
    collector_test.py의 parse_and_insert()에서 호출.

    Parameters
    ----------
    item   : areaBasedList2 응답의 개별 장소 dict
    detail : detailCommon2 응답 dict (overview, homepage 등)
    intro  : detailIntro2 응답 dict (타입별 상세 정보)

    Returns
    -------
    정제된 dict (DB 적재용)
    """
    return {
        # 기본 정보
        "content_id":      str(item.get("contentid", "") or ""),
        "content_type_id": str(item.get("contenttypeid", "") or ""),
        "name":            clean_text(item.get("title")),
        "address":         clean_address(item.get("addr1"), item.get("addr2")),
        "area_code":       str(item.get("areacode", "") or ""),
        "sigungu_code":    str(item.get("sigungucode", "") or ""),
        "lat":             clean_coord(item.get("mapy")),
        "lng":             clean_coord(item.get("mapx")),

        # 상세 정보 (detailCommon2)
        "overview":        clean_text(detail.get("overview")),
        "image_url":       (detail.get("firstimage") or item.get("firstimage") or "").strip(),
        "homepage":        clean_text(detail.get("homepage")),
        "tel":             (item.get("tel") or "").strip(),
        "category":        (item.get("cat1") or "").strip(),
        "modified_time":   (item.get("modifiedtime") or "").strip(),
    }


def preprocess_intro(intro: dict, content_type_id: str) -> dict:
    """
    detailIntro2 응답을 타입별로 파싱 + 정제해서 반환.
    collector_test.py의 parse_intro_by_type() 역할을 대체.
    """
    def c(key):
        return clean_text(intro.get(key))

    if content_type_id == "12":
        return {
            "parking":       c("parking"),
            "rest_date":     c("restdate"),
            "use_time":      c("usetime"),
            "baby_carriage": c("chkbabycarriage"),
        }
    elif content_type_id == "14":
        return {
            "parking_culture":   c("parkingculture"),
            "rest_date_culture": c("restdateculture"),
            "use_time_culture":  c("usetimeculture"),
            "use_fee":           c("usefee"),
        }
    elif content_type_id == "15":
        return {
            "event_start_date":  c("eventstartdate"),
            "event_end_date":    c("eventenddate"),
            "event_place":       c("eventplace"),
            "play_time":         c("playtime"),
            "use_time_festival": c("usetimefestival"),
            "program":           c("program"),
        }
    elif content_type_id == "38":
        return {
            "parking_shopping":   c("parkingshopping"),
            "rest_date_shopping": c("restdateshopping"),
            "open_time":          c("opentime"),
            "sale_item":          c("saleitem"),
            "restroom":           c("restroom"),
        }
    elif content_type_id == "39":
        return {
            "parking_food":   c("parkingfood"),
            "rest_date_food": c("restdatefood"),
            "open_time_food": c("opentimefood"),
            "first_menu":     c("firstmenu"),
            "treat_menu":     c("treatmenu"),
            "kids_facility":  c("kidsfacility"),
        }
    return {}