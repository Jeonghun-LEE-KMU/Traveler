"""
season.py 단위 테스트.

WHY 이 테스트가 필요한가:
  get_season()은 today를 주입받는 순수 함수다. wall-clock(date.today())을
  직접 의존하지 않고 고정 날짜로 월 경계를 검증해야 실행 시점과 무관하게
  항상 같은 결과가 나온다.
"""

from datetime import date

from traveler.core.season import get_season


def test_each_month_maps_to_correct_season():
    """1~12월 전체가 의도한 계절로 매핑되는지 확인."""
    expected = {
        1: "겨울", 2: "겨울", 3: "봄", 4: "봄", 5: "봄",
        6: "여름", 7: "여름", 8: "여름", 9: "가을", 10: "가을",
        11: "가을", 12: "겨울",
    }
    for month, season in expected.items():
        assert get_season(date(2026, month, 15)) == season


def test_year_boundary_december_to_january_both_winter():
    """12월과 1월이 연도가 달라도 똑같이 겨울로 매핑되는지 확인 (절대 경계 X)."""
    assert get_season(date(2025, 12, 31)) == "겨울"
    assert get_season(date(2026, 1, 1)) == "겨울"


def test_season_boundary_feb_to_mar():
    """2월 말 → 3월 초 경계에서 겨울 → 봄으로 바뀌는지 확인."""
    assert get_season(date(2026, 2, 28)) == "겨울"
    assert get_season(date(2026, 3, 1)) == "봄"


def test_no_argument_uses_real_today():
    """today를 안 넘기면 실제 오늘 날짜 기준으로 계산되는지 확인 (예외 없이 동작)."""
    result = get_season()
    assert result in {"봄", "여름", "가을", "겨울"}
