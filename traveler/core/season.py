"""
계절 컨텍스트 모듈 (L2).

[설계 근거]
  weather.py(L1)는 기상청 API를 호출하는 외부 의존이 있어 api/에 위치하지만,
  계절은 오늘 날짜만으로 계산되는 순수 함수라 외부 의존이 없다.
  따라서 RAG 인터페이스 계약(ContextData)을 건드리지 않고
  prompt.py가 직접 호출해 쓰는 core/ 모듈로 둔다.
"""

from datetime import date

_SEASON_BY_MONTH = {
    3: "봄", 4: "봄", 5: "봄",
    6: "여름", 7: "여름", 8: "여름",
    9: "가을", 10: "가을", 11: "가을",
    12: "겨울", 1: "겨울", 2: "겨울",
}


def get_season(today: date | None = None) -> str:
    """
    날짜를 봄/여름/가을/겨울 중 하나로 매핑한다.

    # WHY today를 옵션으로 받는가?
    #   today=None이면 date.today()로 실제 오늘 날짜를 써서 실 서비스에서는
    #   자동으로 현재 계절이 계산된다. 테스트에서는 today에 고정 날짜를 넣어
    #   월 경계(예: 12월→1월)를 wall-clock과 무관하게 검증할 수 있다.
    """
    today = today or date.today()
    return _SEASON_BY_MONTH[today.month]
