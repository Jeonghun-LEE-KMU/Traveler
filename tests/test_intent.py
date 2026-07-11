"""
intent.py 단위 테스트 — parse_companion().

WHY 이 테스트가 필요한가:
  parse_companion()은 사용자 자연어(user_query)에서 동승자 라벨을 부분 문자열
  키워드 매칭으로 뽑아내는 순수 함수다. wall-clock/IO 의존이 없어 고정 입력만으로
  전 동작을 검증할 수 있다(season.py와 같은 철학).

  핵심 정책 두 가지를 회귀로 고정한다:
    1. 우선순위(표 위→아래, 첫 매칭에서 멈춤) — 다중 매칭 시 어느 라벨이 이기는지.
    2. "혼자" 명시("혼자") vs 동승자 미언급("")의 구별 — 6.4 vs 6.5가 핵심 대비.
"""

import pytest

from traveler.core.intent import parse_companion


# ── 6.1 카테고리별 매칭 (각 1건) ──────────────────────────────────────────────
#
# WHY 파라미터화(parametrize)를 쓰는가:
#   8개 카테고리가 "입력 → 기대 라벨" 형태로 구조가 동일하다. 같은 assert를
#   8번 복붙하는 대신 (입력, 기대) 쌍을 표로 두면 어떤 케이스가 깨졌는지
#   pytest가 케이스 단위로 보고해준다(DRY + 가독성).

@pytest.mark.parametrize(
    "user_query, expected",
    [
        ("여자친구랑 부산 데이트 코스 추천", "연인"),
        ("와이프랑 제주 신혼여행", "배우자"),
        ("아이들 데리고 갈 만한 곳", "아이"),
        ("부모님 모시고 효도여행 가고 싶어요", "부모님 모시고"),
        ("온가족 나들이 코스", "가족"),
        ("강아지랑 갈 수 있는 곳", "반려동물"),
        ("회사 워크샵 장소", "단체/모임"),
        ("친구들이랑 놀러 갈 곳", "친구"),
    ],
)
def test_each_category_matches_representative_input(user_query, expected):
    """8개 동승자 카테고리가 각 대표 입력에서 올바른 라벨로 매칭되는지 확인."""
    assert parse_companion(user_query) == expected


# ── 6.2 다중 매칭 우선순위 (결정 7 검증) ──────────────────────────────────────
#
# WHY 따로 두는가:
#   6.1은 "단일 카테고리가 잡히는가"를, 6.2는 "두 카테고리가 동시에 잡힐 때
#   표 순서대로 위쪽이 이기는가"를 검증한다. 표를 누가 재정렬하면 여기서 깨진다.

@pytest.mark.parametrize(
    "user_query, expected",
    [
        # "가족"과 "아이"가 둘 다 매칭되지만 "아이"가 표에서 위 → "아이" 승.
        ("가족이랑 아이 데리고 갈 곳", "아이"),
        # "배우자"(부부)가 "부모님 모시고"보다 위 → "배우자" 승.
        ("부부 동반 부모님 모시고 여행", "배우자"),
        # "반려동물"(강아지)이 "친구"보다 위 → "반려동물" 승.
        ("친구랑 강아지 데려가도 되나요", "반려동물"),
    ],
)
def test_multi_match_resolves_by_table_order(user_query, expected):
    """여러 카테고리가 동시에 매칭되면 표 위쪽(우선순위 높음) 라벨이 이기는지 확인."""
    assert parse_companion(user_query) == expected


# ── 6.3 부분 문자열 오탐 — 알려진 한계를 테스트로 pin ─────────────────────────

def test_substring_false_positive_is_pinned():
    """
    "남친"이 "남친구" 안에 부분 문자열로 들어 있어 "연인"으로 오탐된다.

    WHY: `in` 부분 문자열 매칭의 의도된 트레이드오프(결정 8)다. 버그가 아니라
    "알려진 한계"를 회귀 테스트로 고정해, 이 동작이 우연히 바뀌면 테스트가 깨져
    "한계가 변했다"를 알리게 한다.
    """
    assert parse_companion("남친구랑 여행") == "연인"


# ── 6.4 "혼자" 명시 → 라벨 "혼자" 반환 (결정 3 개정) ──────────────────────────

@pytest.mark.parametrize(
    "user_query",
    [
        "혼자 훌쩍 떠나고 싶어요",
        "나홀로 여행 코스",
    ],
)
def test_explicit_solo_returns_honja(user_query):
    """
    명시적으로 "혼자"/"나홀로"라고 말하면 라벨 "혼자"를 반환하는지 확인.

    WHY: 사용자가 실제로 "혼자 간다"고 말한 것은 정보이므로 GPT에 전달한다.
    동승자 미언급("")과 구별된다 — 6.5의 대비 케이스 참조.
    """
    assert parse_companion(user_query) == "혼자"


# ── 6.5 동승자 미언급 → 빈 문자열 "" (6.4와 대비) ────────────────────────────

@pytest.mark.parametrize(
    "user_query",
    [
        "맛집 추천해줘",
        "경주에서 반나절 자차로 여행하고 싶어요",
    ],
)
def test_no_companion_mentioned_returns_empty(user_query):
    """
    동승자 키워드가 하나도 없으면 빈 문자열 ""을 반환하는지 확인.

    WHY: "미언급"은 "혼자라고 단정"하면 안 되는 정보 없음 상태다. 6.4의
    "명시한 혼자"("혼자")와 나란히 둬서 정책의 의도를 회귀로 고정한다.
    """
    assert parse_companion(user_query) == ""


# ── 6.5b 맨몸 키워드 부분열 오탐 제거 회귀 (fix/jeonghun/intent-fp) ───────────
#
# WHY 이 섹션이 필요한가:
#   "아이"/"자기"/"팀"처럼 짧고 흔한 맨몸 키워드는 "아이스", "자기 전", "팀장님"
#   같은 무관한 단어 안에 부분 문자열로 우연히 포함돼 오탐을 낸다. 조사 결합형
#   ("아이랑","팀이랑" 등)으로 교체한 뒤 실제로 오탐이 사라졌는지 회귀로 고정한다.

def test_ice_americano_does_not_match_child():
    """
    "아이스"의 "아이"가 맨몸 키워드였을 때 "아이" 카테고리로 오탐되던 케이스.

    WHY: "혼자"도 함께 언급되므로, 오탐이 사라지면 "혼자"가 정상적으로 이겨야
    한다(우선순위상 "아이"가 위에 있어 오탐 시 "아이"가 먼저 매칭됐었다).
    """
    assert parse_companion("아이스 아메리카노 마시며 혼자 드라이브") == "혼자"


def test_jagi_jeon_does_not_match_lover():
    """
    "자기 전"의 "자기"가 맨몸 키워드였을 때 "연인" 카테고리로 오탐되던 케이스.

    WHY: 동승자 언급이 전혀 없는 문장이므로, 오탐이 사라지면 미언급("")이 맞다.
    """
    assert parse_companion("자기 전에 갈만한 야경") == ""


def test_team_leader_does_not_match_group():
    """
    "팀장님"의 "팀"이 맨몸 키워드였을 때 "단체/모임" 카테고리로 오탐되던 케이스.

    WHY: "팀장님이랑 둘이"는 실제로는 동승자 카테고리 중 무엇에도 명확히
    속하지 않는 애매한 표현이다. 오분류(특히 "단체/모임"으로 잘못 단정)보다
    미언급("")으로 안전하게 떨어지는 편이 낫다는 정책을 회귀로 고정한다.
    """
    assert parse_companion("팀장님이랑 둘이") != "단체/모임"


# ── 6.6 None / 빈 / 공백 입력 (4절 Graceful Degradation) ─────────────────────

@pytest.mark.parametrize(
    "user_query",
    [
        None,    # "keyword" in None은 TypeError → 진입부 가드가 흡수
        "",      # 어떤 키워드도 in ""로 매칭 불가
        "   ",   # 공백뿐 — 가드는 통과하지만 매칭 없어 끝에서 "" 반환
    ],
)
def test_none_empty_whitespace_return_empty(user_query):
    """
    None/빈 문자열/공백뿐 입력이 예외 없이 빈 문자열 ""을 반환하는지 확인.

    WHY: 파이프라인 중간 단계라 비정상 입력에 예외를 던지면 안 된다
    (Graceful Degradation). 특히 None은 가드가 없으면 TypeError로 터진다.
    """
    assert parse_companion(user_query) == ""
