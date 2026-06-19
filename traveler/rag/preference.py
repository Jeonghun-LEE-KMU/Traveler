"""
사용자 선호도 추출 & 프롬프트 주입 모듈 — Sprint 2

역할:
  1. 사용자 메시지에서 여행 선호도 키워드를 추출 → session.py를 통해 Redis에 누적
  2. 누적된 선호도를 GPT 프롬프트에 주입할 문자열로 변환
  3. visited SET을 GPT가 이해할 수 있는 컨텍스트 문자열로 변환

[설계 결정 — LLM 호출 없이 키워드 매칭]
  WHY: 선호도 추출에 LLM을 쓰면 매 요청마다 추가 API 비용 + 지연 발생.
       여행 선호도는 "커플", "힐링" 같은 명확한 단어 → 키워드 매칭으로 충분.
  트레이드오프: "두 명이서 가요" 같은 표현은 못 잡음 → Sprint 4에서 NER 기반으로 보강 예정.

[누적 방식]
  Redis HASH에 HSET으로 병합 → 이전 값 유지, 새 값만 업데이트.
  ex) 1회차: companion="혼자" → 2회차: mood="감성" → 최종: 둘 다 저장
"""

# ── 키워드 매핑 테이블 ────────────────────────────────────────────────────────

# 동행자 유형 — 가장 구체적인 키워드를 앞에 배치 (early-exit)
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

# 분위기 키워드 — 여러 개 매칭 가능 (쉼표로 누적)
_MOOD_KEYWORDS: list[str] = [
    "힐링", "조용", "여유", "감성", "낭만", "로맨틱",
    "활동적", "신나", "스릴", "모험",
    "역사", "전통", "문화", "예술",
    "맛집", "카페", "디저트", "브런치",
    "자연", "바다", "산", "숲", "강",
    "트렌디", "핫플", "이색", "특별",
    "드라이브", "야경", "뷰",
]

# 여행 모드 — Triple Mode (근교/타지/장거리)
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


# ── 선호도 추출 ───────────────────────────────────────────────────────────────

def extract_prefs_from_message(message: str) -> dict:
    """
    사용자 메시지에서 여행 선호도 키워드를 추출한다.

    INPUT: 사용자 자연어 메시지 (ex: "커플이랑 감성 카페 드라이브 하고 싶어")
    OUTPUT: 추출된 선호도 dict
            ex: {"companion": "커플", "mood": "감성, 카페, 드라이브", "travel_mode": "근교"}
            매칭 없으면 해당 키 없음 → session.update_prefs에서 스킵

    [early-exit 패턴]
    companion, travel_mode는 첫 매칭에서 바로 break.
    mood는 전체 스캔 (여러 개 누적 가능).
    """
    prefs: dict = {}

    # ── 동행자 (첫 매칭에서 종료) ──────────────────────────────
    for keyword, value in _COMPANION_KEYWORDS:
        if keyword in message:
            prefs["companion"] = value
            break

    # ── 분위기 (전체 스캔, 여러 개 가능) ──────────────────────
    found_moods = [kw for kw in _MOOD_KEYWORDS if kw in message]
    if found_moods:
        prefs["mood"] = ", ".join(found_moods)

    # ── 여행 모드 (첫 매칭에서 종료) ───────────────────────────
    for keyword, value in _TRAVEL_MODE_KEYWORDS:
        if keyword in message:
            prefs["travel_mode"] = value
            break

    return prefs


# ── 프롬프트 컨텍스트 조립 ────────────────────────────────────────────────────

def build_prefs_context(prefs: dict) -> str:
    """
    누적 선호도 dict를 GPT 프롬프트 주입용 문자열로 변환한다.

    INPUT: Redis에서 가져온 누적 선호도 dict
    OUTPUT: 프롬프트에 삽입할 문자열 (비어있으면 빈 문자열 → 주입 스킵)

    ex 출력:
        [이 사용자의 누적 여행 선호도 — 이전 대화에서 파악됨]
          - 주요 동행자: 커플
          - 선호 분위기: 감성, 힐링
          - 선호 여행 유형: 근교
        위 선호도를 참고하되, 이번 요청 내용이 최우선입니다.
    """
    if not prefs:
        return ""

    lines = ["[이 사용자의 누적 여행 선호도 — 이전 대화에서 파악됨]"]

    if prefs.get("companion"):
        lines.append(f"  - 주요 동행자: {prefs['companion']}")
    if prefs.get("mood"):
        lines.append(f"  - 선호 분위기: {prefs['mood']}")
    if prefs.get("travel_mode"):
        lines.append(f"  - 선호 여행 유형: {prefs['travel_mode']}")

    lines.append("위 선호도를 참고하되, 이번 요청 내용이 최우선입니다.")
    return "\n".join(lines)


def build_visited_context(visited: set[str]) -> str:
    """
    방문한 place_id SET을 GPT 프롬프트 주입용 문자열로 변환한다.

    INPUT: Redis에서 가져온 visited SET (ex: {"1", "3", "7"})
    OUTPUT: 프롬프트에 삽입할 문자열 (비어있으면 빈 문자열)

    ex 출력:
        [이미 방문한 장소 — 추천에서 반드시 제외]
          - place_id: 1
          - place_id: 3
    """
    if not visited:
        return ""

    lines = ["[이미 방문한 장소 — 추천에서 반드시 제외]"]
    for pid in sorted(visited, key=lambda x: int(x) if x.isdigit() else 0):
        lines.append(f"  - place_id: {pid}")
    return "\n".join(lines)
