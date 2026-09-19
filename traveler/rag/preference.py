"""
사용자 선호도 추출 & 프롬프트 주입 모듈

LLM 호출 없이 키워드 매칭으로 선호도 추출 (비용/속도 최적화)
추출한 선호도는 session.py를 통해 PostgreSQL에 누적
"""

import re
from decimal import Decimal, InvalidOperation

# 광역 + 주요 시/군 단위 지역 매핑
# hybrid_search.detect_region은 광역 17개만 지원 → 여기서 시/군 단위 보완
_REGION_KEYWORDS: list[tuple[str, str]] = [
    # 경상북도
    ("경주", "경주"),
    # 경주 권역·랜드마크 → 전부 "경주"로 매핑 (2026-09-15, 지역 특화)
    # region.py REGION_COORDS 에 같은 키를 넣었다. 거기서는 좌표를 잡고,
    # 여기서는 hybrid_search 지역 필터용 region 값을 잡는다. 둘 다 있어야
    # "감포 바다" 같은 권역 질의가 전국 검색으로 새지 않는다.
    ("보문단지", "경주"),
    ("감포", "경주"),
    ("문무대왕릉", "경주"),
    ("불국사", "경주"),
    ("석굴암", "경주"),
    ("황리단길", "경주"),
    ("대릉원", "경주"),
    ("첨성대", "경주"),
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
    # 경기도/수도권 근교 (develop 지역 커버리지 유지)
    ("가평", "가평"),
    ("양평", "양평"),
    ("남양주", "남양주"),
    ("파주", "파주"),
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
# WHY: 일정 계산은 속도 모델을 달리 적용하므로, LLM 설명이 아니라 사용자가
# 요청 본문에 명시한 이동수단만 정규화해 ContextData로 전달한다.
_TRANSPORT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"대중교통"), "대중교통"),
    (re.compile(r"지하철(?:\s*(?:로|으로|을|를)?\s*(?:타고|이용(?:해서)?|만)?)?"), "대중교통"),
    (re.compile(r"(?:기차|열차|KTX|SRT)(?:\s*(?:로|으로|을|를)?\s*(?:타고|이용(?:해서)?|만)?)?", re.IGNORECASE), "대중교통"),
    # "버스킹"은 공연 장르이지 교통수단이 아니다. 조사·동사를 요구해 오인을 막는다.
    (re.compile(r"버스(?!킹)(?:\s*(?:로|를?\s*타고|타고|이용(?:해서)?|만))"), "대중교통"),
    # 속도 모델을 제공하지 않는 명시 수단은 자차로 추정하지 않고 ``불명``으로 남긴다.
    (re.compile(r"(?:자전거|킥보드)(?:\s*(?:로|으로|를?\s*타고|타고|이용(?:해서)?|만))"), "불명"),
    (re.compile(r"걸어서|도보(?:\s*(?:로|로만|여행)?)?"), "도보"),
    # ``기차로`` 안의 "차로"처럼 다른 명사에 포함된 음절은 자차가 아니다.
    (re.compile(r"자동차|자차|(?<![가-힣])차(?:로|를?\s*타고|타고|\s*(?:없이|말고)|나|와)"), "자차"),
]
_TRANSPORT_NEGATION_RE = re.compile(r"(?:없이|말고|아니(?:라)?|안\s*(?:타|갈)|못\s*(?:타|갈))")


# 숫자 없이 길이를 말하는 표현 → 시간. 긴 표현이 먼저 와야 "하루 종일"이
# "하루"에 먼저 걸리지 않는다(리스트 순서가 곧 우선순위).
# develop의 당일치기 8시간 정책과 오전/오후/저녁 표현도 같은 파서에서 처리한다.
_DURATION_KEYWORDS: list[tuple[str, int]] = [
    ("하루 종일", 8),
    ("하루종일", 8),
    ("당일치기", 8),
    ("종일", 8),
    ("반나절", 5),      # 4~6시간의 중앙값 — duration_hours가 정수 하나라 범위를 못 담는다
    ("하루", 8),
    ("1박", 8),         # 다일 일정은 코스 모델이 하루짜리라 "첫날 종일"로 근사한다
    ("2박", 8),
    ("오전", 3),
    ("오후", 3),
    ("저녁", 2),
]

# "5시간 코스" → 5. 뒤에 거리·소요를 뜻하는 말이 오면 여행 길이가 아니므로 건너뛴다
# ("3시간 거리에 있는", "3시간 걸리는 곳", "3시간 소요되는 곳").
# 숫자 경계를 강제해 "123시간"을 뒤의 "23시간"으로 부분 매칭하지 않는다.
_DECIMAL_NUMBER = r"-?(?:\d+(?:[.,]\d+)?|[.,]\d+)"
_DURATION_RANGE_RE = re.compile(
    rf"(?<![\d.,]){_DECIMAL_NUMBER}\s*(?:시간|분)?\s*(?:~|[-–])\s*{_DECIMAL_NUMBER}\s*(?:시간|분)"
)
_HOURS_AND_MINUTES_RE = re.compile(
    rf"(?<![\d.,])(?P<hours>{_DECIMAL_NUMBER})\s*시간"
    rf"(?:\s*(?P<half>반)|\s*(?P<minutes>{_DECIMAL_NUMBER})\s*분)?"
)
_MINUTES_RE = re.compile(rf"(?<![\d.,])(?P<minutes>{_DECIMAL_NUMBER})\s*분")
_DURATION_CONTEXT_RE = re.compile(
    r"^\s*(?:이내\s*거리|거리|걸(?:리|려|리는)|소요|(?:전|후)(?:에)?(?=\s|$)|예약)")
_DURATION_PREFIX_RE = re.compile(r"(?:이동\s*(?:소요)?시간|체류시간)\s*$")
_MIN_DURATION_MINUTES = 1
_MAX_DURATION_MINUTES = 12 * 60
_DURATION_ERROR = "가용 시간은 1분 이상 12시간 이하의 정수 분으로 입력해 주세요."


# WHY: 프론트의 보조 문구가 사용자 요청으로 섞이면 "구성: 4시간 코스" 같은
# UI 값이 선호도·일정 파싱을 오염시킨다. 알려진 라벨만 경계로 삼아 자유 입력은 보존한다.
_REQUEST_LABEL_RE = re.compile(r"^\s*요청\s*:\s*(.*)$")
_UI_LABEL_RE = re.compile(r"^\s*(?:여행\s*유형|동행자|구성|교통\s*수단|이동\s*수단|요청)\s*:")


def extract_user_request(message: str) -> str:
    """
    프론트 UI 라벨을 제외한 사용자의 직접 요청 본문을 반환한다.

    # INPUT: UI 선택값과 "요청:" 본문이 함께 담긴 메시지
    # OUTPUT: 멀티라인 자유 입력 문자열(→ preference/schedule 파서)
    # WHY: 인식 가능한 UI 라벨에서만 멈춰 사용자가 여러 줄로 쓴 요청은 잃지 않는다.
    """
    lines = message.splitlines()

    for index, line in enumerate(lines):
        request_match = _REQUEST_LABEL_RE.match(line)
        if not request_match:
            continue

        request_lines = [request_match.group(1)]
        for following_line in lines[index + 1:]:
            if _UI_LABEL_RE.match(following_line):
                break
            request_lines.append(following_line)
        return "\n".join(request_lines).strip()

    # "요청:"이 없어도 알려진 UI 라벨 줄만 빼고 나머지 자유 입력은 보존한다.
    # 따라서 UI 구성값은 세션 선호도 누적의 사용자 제약이 될 수 없다.
    return "\n".join(line for line in lines if not _UI_LABEL_RE.match(line)).strip()


def _is_non_availability_time(text: str, start: int, end: int) -> bool:
    """이동·과거 시점을 설명하는 시간은 여행 가능 시간으로 쓰지 않는다."""
    # 시간 뒤에 바로 붙은 여행/이동 문맥만 검사한다. 앞선 "30분 걸리는 곳"이
    # 뒤의 "4시간 여행"까지 삼키거나 "1시간 이내로 코스"를 막으면 안 된다.
    prefix = text[max(0, start - 12):start]
    return bool(_DURATION_PREFIX_RE.search(prefix) or _DURATION_CONTEXT_RE.search(text[end:]))


def _normalise_duration_minutes(minutes: Decimal) -> int:
    if minutes != minutes.to_integral_value():
        raise ValueError(_DURATION_ERROR)
    normalized = int(minutes)
    if not _MIN_DURATION_MINUTES <= normalized <= _MAX_DURATION_MINUTES:
        raise ValueError(_DURATION_ERROR)
    return normalized


def extract_duration_minutes(text: str) -> int | None:
    """사용자가 명시한 여행 가능 시간을 정수 분으로 반환한다.

    이동 소요·거리·예약 시점은 ``None``으로 무시한다. 반면 범위, 0분, 음수,
    12시간 초과, 분 단위로 정규화할 수 없는 명시 입력은 기본 예산으로 숨기지 않고
    한국어 안내가 담긴 ``ValueError``로 호출자에게 전달한다.
    """
    if _DURATION_RANGE_RE.search(text):
        raise ValueError(_DURATION_ERROR)

    for match in _HOURS_AND_MINUTES_RE.finditer(text):
        if _is_non_availability_time(text, match.start(), match.end()):
            continue
        try:
            minutes = Decimal(match.group("hours").replace(",", ".")) * Decimal(60)
            if match.group("half"):
                minutes += Decimal(30)
            elif match.group("minutes") is not None:
                minutes += Decimal(match.group("minutes").replace(",", "."))
        except InvalidOperation as error:  # 방어적 처리: 정규식과 Decimal 계약을 함께 유지한다.
            raise ValueError(_DURATION_ERROR) from error
        return _normalise_duration_minutes(minutes)

    for match in _MINUTES_RE.finditer(text):
        if _is_non_availability_time(text, match.start(), match.end()):
            continue
        try:
            return _normalise_duration_minutes(Decimal(match.group("minutes").replace(",", ".")))
        except InvalidOperation as error:
            raise ValueError(_DURATION_ERROR) from error

    # "하루 전에"는 과거 예약 시점이며 "하루 코스"와 다르다.
    if re.search(r"하루\s*전", text):
        return None
    for keyword, hours in _DURATION_KEYWORDS:
        if keyword in text:
            return hours * 60
    return None


def extract_duration_hours(text: str) -> int | float | None:
    """
    요청 문장에서 여행 가능 시간을 뽑는다. 못 찾으면 None.

    # WHY: 프론트는 여행 시간을 보내지 않아 ContextData.duration_hours 기본값이
    #   그대로 프롬프트의 "가용 시간"에 들어간다. 그래서 "하루 종일"이라고 말해도
    #   3시간짜리 코스가 나왔다(2026-09-16 실측: 6회 모두 체류 합계 180분).
    # INPUT: "요청:" 이후 사용자 자유 입력
    # OUTPUT: 기존 정수 시간 또는 분 단위 계약을 잃지 않는 소수 시간. None이면 기본값 유지.
    """
    minutes = extract_duration_minutes(text)
    if minutes is None:
        return None
    hours = minutes / 60
    return int(hours) if hours.is_integer() else hours


def _extract_transport(text: str) -> tuple[str | None, bool]:
    """교통수단과 명시 여부를 반환한다.

    혼합 방식과 부정만 한 방식은 기본 자차로 덮지 않는다. 일정 계산에서는
    ``혼합``/``불명``을 속도 모델이 없는 값으로 보아 unverified로 처리한다.
    """
    found: set[str] = set()
    mentioned = False
    for pattern, transport in _TRANSPORT_PATTERNS:
        for match in pattern.finditer(text):
            mentioned = True
            nearby = text[max(0, match.start() - 2): min(len(text), match.end() + 8)]
            if not _TRANSPORT_NEGATION_RE.search(nearby):
                found.add(transport)

    if len(found) == 1:
        return next(iter(found)), True
    if len(found) > 1:
        return "혼합", True
    return ("불명", True) if mentioned else (None, False)


def extract_duration(message: str) -> int | float | None:
    """develop의 호출자도 PR #47과 같은 가용시간 파서를 사용한다."""
    return extract_duration_hours(message)


def extract_prefs_from_message(message: str) -> dict:
    prefs: dict = {}

    # 프론트가 "동행자: 가족\n...\n요청: 친구랑 ..." 형식으로 보내는 경우,
    # UI 선택값(동행자 줄)보다 직접 입력한 요청 줄이 우선한다.
    # 알려진 뒤 UI 라벨은 제외하되, 여러 줄 사용자 요청은 유지한다.
    companion_source = extract_user_request(message)

    for keyword, value in _COMPANION_KEYWORDS:
        if keyword in companion_source:
            prefs["companion"] = value
            break

    transport, transport_explicit = _extract_transport(companion_source)
    if transport_explicit:
        prefs["transport"] = transport
        # 세션에 이전 기본 자차가 있어도 이번 요청은 명시 교통 조건을 남긴다.
        prefs["transport_assumed"] = False

    # 시간도 companion과 같은 구간에서만 읽는다 — 프론트가 뒤에 붙이는
    # "구성: ... 4~5곳" 줄이나 UI 선택 줄을 여행 길이로 오인하지 않게 한다.
    duration_minutes = extract_duration_minutes(companion_source)
    if duration_minutes is not None:
        prefs["duration_minutes"] = duration_minutes
        duration_hours = duration_minutes / 60
        prefs["duration_hours"] = int(duration_hours) if duration_hours.is_integer() else duration_hours

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
