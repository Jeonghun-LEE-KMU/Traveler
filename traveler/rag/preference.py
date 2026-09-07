"""
사용자 선호도 추출 & 프롬프트 주입 모듈

LLM 호출 없이 키워드 매칭으로 선호도 추출 (비용/속도 최적화)
추출한 선호도는 session.py를 통해 PostgreSQL에 누적
"""

# 광역 + 주요 시/군 단위 지역 매핑
# hybrid_search.detect_region은 광역 17개만 지원 → 여기서 시/군 단위 보완
_REGION_KEYWORDS: list[tuple[str, str]] = [
    # 경상북도
    ("경주", "경주"),
    ("안동", "안동"),
    ("포항", "포항"),
    ("구미", "구미"),
    ("영주", "영주"),
    ("경산", "경산"),
    # 경상남도
    ("통영", "통영"),
    ("거제", "거제"),
    ("여수", "전남"),      # 전남으로 fallback
    ("밀양", "밀양"),
    ("창원", "창원"),
    ("김해", "김해"),
    # 강원
    ("강릉", "강릉"),
    ("속초", "속초"),
    ("춘천", "춘천"),
    ("평창", "평창"),
    ("정선", "정선"),
    ("인제", "인제"),
    # 전라북도
    ("전주", "전주"),
    ("남원", "남원"),
    ("군산", "군산"),
    # 전라남도
    ("여수", "여수"),
    ("순천", "순천"),
    ("담양", "담양"),
    ("보성", "보성"),
    ("구례", "구례"),
    ("하동", "하동"),
    # 충청
    ("청주", "청주"),
    ("공주", "공주"),
    ("부여", "부여"),
    # 제주
    ("서귀포", "서귀포"),
    # 광역시 (hybrid_search에도 있지만 중복 안전)
    ("수원", "경기"),
    ("용인", "경기"),
]


def extract_region(message: str) -> str | None:
    """
    사용자 메시지에서 지역명 추출.
    hybrid_search.detect_region이 광역 17개만 지원하므로 시/군 단위 보완.
    """
    for keyword, region in _REGION_KEYWORDS:
        if keyword in message:
            return region
    return None

_COMPANION_KEYWORDS: list[tuple[str, str]] = [
    ("유아", "가족(유아)"),
    ("애기", "가족(유아)"),
    ("아기", "가족(유아)"),
    ("아이", "가족(아이)"),
    ("어린이", "가족(아이)"),
    ("초등", "가족(아이)"),
    ("가족", "가족"),
    ("남자친구", "커플"),
    ("여자친구", "커플"),
    ("남친", "커플"),
    ("여친", "커플"),
    ("커플", "커플"),
    ("연인", "커플"),
    ("친구", "친구"),
    ("혼자", "혼자"),
    ("솔로", "혼자"),
    ("1인", "혼자"),
]

_MOOD_KEYWORDS: list[str] = [
    "힐링", "조용", "여유", "감성", "낭만", "로맨틱",
    "활동적", "신나", "스릴", "모험",
    "역사", "전통", "문화", "예술",
    "맛집", "카페", "디저트", "브런치",
    "자연", "바다", "산", "숲", "강",
    "트렌디", "핫플", "이색", "특별",
    "드라이브", "야경", "뷰",
]

_TRAVEL_MODE_KEYWORDS: list[tuple[str, str]] = [
    ("당일치기", "근교"),
    ("근교", "근교"),
    ("드라이브", "근교"),
    ("동네", "근교"),
    ("고속도로", "장거리"),
    ("장거리", "장거리"),
    ("고속", "장거리"),
    ("타지", "타지"),
    ("여행", "타지"),
    ("관광", "타지"),
    ("투어", "타지"),
]


def extract_prefs_from_message(message: str) -> dict:
    prefs: dict = {}

    for keyword, value in _COMPANION_KEYWORDS:
        if keyword in message:
            prefs["companion"] = value
            break

    found_moods = [kw for kw in _MOOD_KEYWORDS if kw in message]
    if found_moods:
        prefs["mood"] = ", ".join(found_moods)

    for keyword, value in _TRAVEL_MODE_KEYWORDS:
        if keyword in message:
            prefs["travel_mode"] = value
            break

    return prefs


def build_prefs_context(prefs: dict) -> str:
    if not prefs:
        return ""

    lines = ["[이 사용자의 누적 여행 선호도]"]

    if prefs.get("companion"):
        lines.append(f"  - 동행자: {prefs['companion']}")
    if prefs.get("mood"):
        lines.append(f"  - 선호 분위기: {prefs['mood']}")
    if prefs.get("travel_mode"):
        lines.append(f"  - 여행 유형: {prefs['travel_mode']}")

    lines.append("위 선호도를 참고하되, 이번 요청 내용이 최우선입니다.")
    return "\n".join(lines)


def build_visited_context(visited: set[str]) -> str:
    if not visited:
        return ""

    lines = ["[이미 방문한 장소 - 추천에서 제외]"]
    for pid in sorted(visited, key=lambda x: int(x) if x.isdigit() else 0):
        lines.append(f"  - place_id: {pid}")
    return "\n".join(lines)
