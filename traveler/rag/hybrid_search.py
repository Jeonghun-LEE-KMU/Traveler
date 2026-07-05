"""
hybrid_search.py — Pinecone + SQLite 기반 하이브리드 장소 검색

[전체 연결 흐름]
사용자 입력
  → main.py
  → preference.py / intent.py
      - 사용자 선호도, 지역, 콘텐츠 유형 추출
  → retriever.py
  → search_places()  ★ 이 파일
      1. Pinecone: 의미적으로 비슷한 장소 검색
      2. SQLite: 장소 상세정보 및 활성 상태 확인
      3. 지역·콘텐츠 유형·방문 이력 필터링
      4. semantic, keyword, preference, iconic 점수로 재정렬
  → retriever.py가 검색 결과를 SpotCandidate로 변환
  → ContextData 생성
  → course_generator.py
  → LLM 여행 코스 생성

[이 파일의 책임]
- Pinecone과 SQLite를 함께 사용해 추천 후보 장소를 검색한다.
- 지역, 콘텐츠 유형, 사용자 선호도, 방문 장소 제외 조건을 반영한다.
- 검색 후보별 최종 점수를 계산해 정렬된 결과를 반환한다.
- ContextData나 최종 여행 코스를 직접 생성하지 않는다.

[다른 파일과의 인터페이스]
- 호출하는 파일: traveler/rag/retriever.py
- 입력:
    user_query   : 사용자의 원문 질문
    region       : intent/backend에서 추출한 지역
    content_types: 검색할 TourAPI 콘텐츠 유형
    preferences  : preference.py에서 추출하거나 Redis에 누적된 선호도
    visited_ids  : Redis에서 조회한 이미 방문한 장소 ID
    top_k        : 반환할 장소 수
- 반환:
    list[HybridSearchResult]

[통합 시 Backend 담당 작업]
retriever.py에서 search_places()를 호출하고, 반환 결과를
SpotCandidate로 변환하여 ContextData.candidate_spots에 넣어야 한다.

[주의]
- 테스트용 지역이나 선호도를 이 파일에 하드코딩하지 않는다.
- 지역과 선호도는 main.py/retriever.py에서 전달받는다.
- hybrid_search_test.py는 단독 품질 검증용이며 실제 서비스에서 호출하지 않는다.
"""

from __future__ import annotations

import math
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

from traveler.rag.embed import INDEX_NAME, embed_texts


load_dotenv()

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "places.db"
DENSE_CANDIDATE_K = 200

REGION_AREA_CODES = {
    "서울": "1",
    "인천": "2",
    "대전": "3",
    "대구": "4",
    "광주": "5",
    "부산": "6",
    "울산": "7",
    "세종": "8",
    "경기": "31",
    "강원": "32",
    "충북": "33",
    "충남": "34",
    "경북": "35",
    "경남": "36",
    "전북": "37",
    "전남": "38",
    "제주": "39",
}

REGION_ALIASES = {
    "서울": ("서울특별시", "서울"),
    "인천": ("인천광역시", "인천"),
    "대전": ("대전광역시", "대전"),
    "대구": ("대구광역시", "대구"),
    "광주": ("광주광역시", "광주"),
    "부산": ("부산광역시", "부산"),
    "울산": ("울산광역시", "울산"),
    "세종": ("세종특별자치시", "세종"),
    "경기": ("경기도", "경기"),
    "강원": ("강원특별자치도", "강원도", "강원"),
    "충북": ("충청북도", "충북"),
    "충남": ("충청남도", "충남"),
    "경북": ("경상북도", "경북"),
    "경남": ("경상남도", "경남"),
    "전북": ("전북특별자치도", "전라북도", "전북"),
    "전남": ("전라남도", "전남"),
    "제주": ("제주특별자치도", "제주도", "제주"),
}

CATEGORY_NAMES = {
    "12": "관광",
    "14": "문화",
    "15": "축제",
    "38": "쇼핑",
    "39": "음식",
}

STOPWORDS = {
    "추천",
    "추천해줘",
    "좋은",
    "좋아",
    "곳은",
    "어디야",
    "가고",
    "싶어",
    "먹고",
    "에서",
    "으로",
    "여행",
    "장소",
}

QUERY_CONTROL_TERMS = {
    # 콘텐츠 유형은 detect_content_types()가 별도로 처리한다.
    "관광지",
    "관광명소",
    "음식점",
    "식당",
    "맛집",
    "카페",
    "축제",
    "행사",
    # 선호 조건은 calculate_preference_score()가 별도로 처리한다.
    "데이트하기",
    "데이트",
    "분위기좋은",
    "분위기 있는",
    "분위기",
    "아이와",
    "아이",
    "어린이",
    "유아",
    "가족",
    "반려동물",
    "강아지",
    "애견",
    "비 오는 날",
    "비오는 날",
    "실내",
}

KOREAN_PARTICLE_SUFFIXES = (
    "에서",
    "으로",
    "하고",
    "이랑",
    "처럼",
    "까지",
    "부터",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "의",
)


@dataclass(frozen=True)
class HybridSearchResult:
    content_id: str
    name: str
    content_type_id: str
    category: str
    address: str
    lat: float
    lng: float
    indoor: bool
    semantic_score: float
    keyword_score: float
    preference_score: float
    iconic_score: float
    final_score: float


def detect_region(text: str) -> str | None:
    for region, aliases in REGION_ALIASES.items():
        if any(alias in text for alias in aliases):
            return region
    return None


def detect_content_types(text: str) -> set[str]:
    if any(word in text for word in ("음식", "식당", "맛집", "카페", "먹")):
        return {"39"}
    if any(word in text for word in ("축제", "행사", "페스티벌")):
        return {"15"}
    if any(word in text for word in ("박물관", "미술관", "도서관", "문화시설")):
        return {"14"}
    if any(word in text for word in ("관광지", "명소", "산책", "공원", "체험")):
        return {"12", "14"}

    # 축제 임베딩 텍스트가 확정되기 전 MVP 기본 검색에서는 제외한다.
    return {"12", "14", "39"}


def query_tokens(text: str) -> list[str]:
    cleaned = text.lower()

    # 지역은 SQLite 필터가 처리하므로 키워드 점수에 중복 반영하지 않는다.
    removable_terms = set(QUERY_CONTROL_TERMS)
    for aliases in REGION_ALIASES.values():
        removable_terms.update(alias.lower() for alias in aliases)

    for term in sorted(removable_terms, key=len, reverse=True):
        cleaned = cleaned.replace(term, " ")

    raw_tokens = re.findall(r"[0-9A-Za-z가-힣]{2,}", cleaned)
    tokens: list[str] = []
    for token in raw_tokens:
        for suffix in KOREAN_PARTICLE_SUFFIXES:
            if token.endswith(suffix) and len(token) > len(suffix) + 1:
                token = token[: -len(suffix)]
                break
        if len(token) >= 2:
            tokens.append(token)

    return [
        token
        for token in dict.fromkeys(tokens)
        if token not in STOPWORDS
    ]


def normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    minimum = min(scores.values())
    maximum = max(scores.values())
    if minimum == maximum:
        # 모든 장소가 0점이면 키워드 신호가 없다는 뜻이다.
        # 이때 전부 1점으로 만들면 검색어와 무관한 장소가 상위로 올라간다.
        normalized_value = 0.0 if maximum == 0.0 else 1.0
        return {key: normalized_value for key in scores}
    return {
        key: (value - minimum) / (maximum - minimum)
        for key, value in scores.items()
    }


def row_text(row: dict) -> str:
    fields = [
        "name",
        "address",
        "overview",
        "first_menu",
        "treat_menu",
        "kids_facility",
        "event_place",
        "program",
    ]
    return " ".join(
        str(row.get(field) or "")
        for field in fields
    ).lower()


def is_indoor(row: dict) -> bool:
    content_type_id = str(row.get("content_type_id") or "")
    if content_type_id in {"14", "39"}:
        return True

    name = str(row.get("name") or "")
    outdoor_name_words = (
        "공원",
        "해수욕장",
        "계곡",
        "둘레길",
        "산책로",
        "광장",
        "수목원",
        "자연휴양림",
    )
    if any(word in name for word in outdoor_name_words):
        return False

    text = row_text(row)
    return any(
        word in text
        for word in (
            "실내",
            "박물관",
            "미술관",
            "전시관",
            "체험관",
            "스파",
            "아쿠아리움",
        )
    )


def calculate_keyword_score(
    row: dict,
    tokens: list[str],
) -> float:
    if not tokens:
        return 0.0

    name = str(row.get("name") or "").lower()
    text = row_text(row)
    score = 0.0
    for token in tokens:
        if token in name:
            score += 3.0
        elif token in text:
            score += 1.0
    return score


def calculate_preference_score(row: dict, query: str) -> float:
    text = row_text(row)
    scores: list[float] = []

    if any(word in query for word in ("아이", "어린이", "유아", "가족")):
        if int(row.get("is_nokidszone") or 0) == 1:
            return 0.0
        kids_facility = str(row.get("kids_facility") or "").strip()
        kids_words = ("아이", "어린이", "유아", "키즈", "가족")
        scores.append(
            1.0
            if kids_facility or any(word in text for word in kids_words)
            else 0.0
        )

    if any(word in query for word in ("데이트", "분위기", "감성")):
        mood_words = (
            "데이트",
            "분위기",
            "감성",
            "인테리어",
            "야경",
            "전망",
            "뷰",
            "테라스",
            "브런치",
            "통창",
        )
        hit_count = sum(word in text for word in mood_words)
        scores.append(min(hit_count / 2.0, 1.0))

    if any(word in query for word in ("비", "비오는", "비 오는")):
        scores.append(1.0 if is_indoor(row) else 0.0)

    if any(word in query for word in ("반려동물", "강아지", "애견")):
        scores.append(
            1.0 if int(row.get("is_pet_allowed") or 0) == 1 else 0.0
        )

    return sum(scores) / len(scores) if scores else 0.5


def calculate_accumulated_preference_score(
    row: dict,
    preferences: dict | None,
) -> float:
    """preference.py가 만든 누적 선호 dict를 장소 점수로 변환한다."""
    if not preferences:
        return 0.5

    scores: list[float] = []
    companion = str(preferences.get("companion") or "")
    content_type_id = str(row.get("content_type_id") or "")
    text = row_text(row)

    if companion.startswith("가족"):
        scores.append(calculate_preference_score(row, "아이 가족"))
    elif companion == "커플":
        scores.append(calculate_preference_score(row, "데이트 분위기"))

    moods = [
        mood.strip()
        for mood in str(preferences.get("mood") or "").split(",")
        if mood.strip()
    ]
    if moods:
        food_moods = {"맛집", "카페", "디저트", "브런치"}
        culture_moods = {"역사", "전통", "문화", "예술"}
        nature_moods = {"자연", "바다", "산", "숲", "강"}
        mood_hits = 0

        for mood in moods:
            if mood in food_moods:
                matched = content_type_id == "39" or mood in text
            elif mood in culture_moods:
                matched = content_type_id == "14" or mood in text
            elif mood in nature_moods:
                matched = any(
                    word in text
                    for word in ("자연", "바다", "해변", "산", "숲", "강")
                )
            else:
                matched = mood in text
            mood_hits += int(matched)

        scores.append(mood_hits / len(moods))

    # travel_mode는 현재 위치/Tmap 거리가 연결된 뒤 거리 점수로 사용한다.
    return sum(scores) / len(scores) if scores else 0.5


def matches_region(row: dict, region: str | None) -> bool:
    if not region:
        return True

    # 서울·부산·경기처럼 TourAPI areaCode가 있는 광역 지역은
    # 구조화된 코드로 먼저 확인한다.
    expected_area_code = REGION_AREA_CODES.get(region)
    if (
        expected_area_code
        and str(row.get("area_code") or "") == expected_area_code
    ):
        return True

    # 경주처럼 광역 areaCode 표에 없는 시·군도 받을 수 있도록
    # 주소 문자열을 안전한 fallback으로 사용한다.
    address = str(row.get("address") or "")
    aliases = REGION_ALIASES.get(region)
    if aliases is None:
        aliases = (
            (region,)
            if region.endswith(("시", "군", "구"))
            else (region, f"{region}시", f"{region}군", f"{region}구")
        )
    return any(alias in address for alias in aliases)


def load_active_places(
    conn: sqlite3.Connection,
    *,
    region: str | None,
    content_types: set[str],
) -> list[dict]:
    placeholders = ", ".join("?" for _ in content_types)
    rows = conn.execute(
        f"""
        SELECT *
        FROM places
        WHERE is_active = 1
          AND content_type_id IN ({placeholders})
        """,
        sorted(content_types),
    ).fetchall()
    places = [dict(row) for row in rows]
    return [
        row
        for row in places
        if matches_region(row, region)
    ]


def search_places(
    user_query: str,
    *,
    preferences: dict | None = None,
    visited_ids: set[str] | None = None,
    region: str | None = None,
    content_types: list[str] | None = None,
    top_k: int = 10,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> list[HybridSearchResult]:
    """
    Pinecone 의미 점수와 SQLite 키워드·선호·Iconic 점수를 결합한다.

    preferences:
      preference.py/Redis가 만든 companion, mood, travel_mode dict.
    visited_ids:
      Redis visited SET. 해당 content_id는 검색 결과에서 완전히 제외한다.

    현재 가중치:
      semantic 65% + keyword 20% + preference 10% + iconic 5%
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if not openai_key or not pinecone_key:
        raise RuntimeError(
            ".env에 OPENAI_API_KEY와 PINECONE_API_KEY가 필요합니다."
        )

    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"places.db를 찾을 수 없습니다: {db_path}")

    resolved_region = region or detect_region(user_query)
    resolved_types = (
        set(content_types)
        if content_types
        else detect_content_types(user_query)
    )

    openai_client = OpenAI(api_key=openai_key)
    index = Pinecone(api_key=pinecone_key).Index(INDEX_NAME)
    query_vector = embed_texts(openai_client, [user_query])[0]
    dense_response = index.query(
        vector=query_vector,
        top_k=DENSE_CANDIDATE_K,
        include_metadata=False,
    )
    raw_dense_scores = {
        str(match.id): float(match.score)
        for match in (dense_response.matches or [])
    }
    semantic_scores = normalize_scores(raw_dense_scores)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = load_active_places(
            conn,
            region=resolved_region,
            content_types=resolved_types,
        )

    tokens = query_tokens(user_query)
    if resolved_region:
        # 지역은 SQLite hard filter가 담당한다. 지역명을 키워드 점수에도
        # 다시 반영하면 모든 지역 내 장소가 동일하게 1점이 되는 문제가 생긴다.
        normalized_region = resolved_region.lower()
        region_tokens = {
            normalized_region,
            normalized_region.removesuffix("시"),
            normalized_region.removesuffix("군"),
            normalized_region.removesuffix("구"),
        }
        region_tokens.discard("")
        tokens = [
            token
            for token in tokens
            if token not in region_tokens
        ]
    raw_keyword_scores = {
        str(row["content_id"]): calculate_keyword_score(row, tokens)
        for row in rows
    }
    keyword_scores = normalize_scores(raw_keyword_scores)

    results: list[HybridSearchResult] = []
    excluded_ids = {
        str(content_id)
        for content_id in (visited_ids or set())
    }
    requires_indoor = "실내" in user_query
    companion = str((preferences or {}).get("companion") or "")
    requires_kid_friendly = any(
        word in user_query
        for word in ("아이", "어린이", "유아", "가족")
    ) or companion.startswith("가족")

    for row in rows:
        content_id = str(row["content_id"])
        if content_id in excluded_ids:
            continue

        indoor = is_indoor(row)

        # 사용자가 명시적으로 '실내'를 요청했다면 소프트 점수가 아닌
        # 반드시 만족해야 하는 필터로 적용한다.
        if requires_indoor and not indoor:
            continue

        # 아이 동반 요청과 노키즈존은 동시에 만족할 수 없는 필수 조건이다.
        if (
            requires_kid_friendly
            and int(row.get("is_nokidszone") or 0) == 1
        ):
            continue

        semantic_score = semantic_scores.get(content_id, 0.0)
        keyword_score = keyword_scores.get(content_id, 0.0)
        query_preference_score = calculate_preference_score(row, user_query)
        if preferences:
            accumulated_preference_score = (
                calculate_accumulated_preference_score(row, preferences)
            )
            # 이번 요청을 누적 선호보다 우선한다.
            preference_score = (
                0.7 * query_preference_score
                + 0.3 * accumulated_preference_score
            )
        else:
            preference_score = query_preference_score
        iconic_score = float(row.get("iconic_score") or 0.0)

        # 의미·키워드 신호가 모두 없는 행은 후보에서 제외한다.
        if semantic_score == 0.0 and keyword_score == 0.0:
            continue

        final_score = (
            0.65 * semantic_score
            + 0.20 * keyword_score
            + 0.10 * preference_score
            + 0.05 * iconic_score
        )
        results.append(
            HybridSearchResult(
                content_id=content_id,
                name=str(row.get("name") or ""),
                content_type_id=str(row.get("content_type_id") or ""),
                category=CATEGORY_NAMES.get(
                    str(row.get("content_type_id") or ""),
                    "기타",
                ),
                address=str(row.get("address") or ""),
                lat=float(row.get("lat") or 0.0),
                lng=float(row.get("lng") or 0.0),
                indoor=indoor,
                semantic_score=semantic_score,
                keyword_score=keyword_score,
                preference_score=preference_score,
                iconic_score=iconic_score,
                final_score=final_score,
            )
        )

    results.sort(key=lambda result: result.final_score, reverse=True)
    return results[:top_k]


def estimate_travel_minutes(
    current_lat: float,
    current_lng: float,
    place_lat: float,
    place_lng: float,
    *,
    average_speed_kmh: float = 30.0,
) -> float:
    """
    Tmap 연동 전 MVP용 직선거리 기반 예상 이동시간.

    실제 서비스에서는 retriever/Tmap 결과로 교체해야 한다.
    """
    if not all([current_lat, current_lng, place_lat, place_lng]):
        return 0.0

    earth_radius_km = 6371.0
    lat1 = math.radians(current_lat)
    lat2 = math.radians(place_lat)
    delta_lat = math.radians(place_lat - current_lat)
    delta_lng = math.radians(place_lng - current_lng)
    value = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(delta_lng / 2) ** 2
    )
    distance_km = earth_radius_km * 2 * math.atan2(
        math.sqrt(value),
        math.sqrt(1 - value),
    )
    return round((distance_km / average_speed_kmh) * 60, 1)
