"""
사용자 선호도 추출 & 프롬프트 주입 모듈

LLM 호출 없이 키워드 매칭으로 선호도 추출 (비용/속도 최적화)
추출한 선호도는 session.py를 통해 Redis에 누적
"""

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
