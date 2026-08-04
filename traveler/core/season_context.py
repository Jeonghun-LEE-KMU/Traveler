"""
계절 컨텍스트 조립 모듈 (L2 계절 + L3 축제).

[설계 근거 — Sprint 5 Task 5, docs/superpowers/specs/2026-08-03-sprint5-eval-pipeline-design.md §6]
  season.py(L2)는 날짜만으로 계산되는 순수 함수라 외부 의존이 없다.
  이번 스프린트에서 추가하는 축제(L3)도 같은 철학을 따른다 — 실제 축제 데이터를
  SQLite에서 조회하는 부분(ContextData.festivals 계약 확장)은 다음 스프린트로
  미루고(D4), 이 모듈은 "이미 가진 축제 리스트를 필터링·조립하는 순수 함수"만
  제공한다. 외부 의존(DB·API)이 없으므로 season.py와 나란히 core/에 둔다.

  season.py 자체는 무수정(외과적 변경 원칙) — get_season()만 import해서 재사용한다.
"""

from dataclasses import dataclass
from datetime import date, datetime

from traveler.core.season import get_season


@dataclass(frozen=True)
class FestivalInfo:
    """
    진행 중인 축제·행사 1건.

    # WHY frozen dataclass?
    #   season_context 모듈의 다른 값 객체와 마찬가지로 불변으로 두면
    #   필터링·정렬 과정에서 원본이 실수로 변경될 걱정이 없다.
    event_start_date / event_end_date는 "YYYYMMDD" 문자열이다(예: "20260805").
    잘못된 형식이 섞여도 filter_active_festivals()가 예외 없이 스킵한다.
    """
    name: str
    event_start_date: str
    event_end_date: str
    event_place: str = ""


def _parse_yyyymmdd(value: str) -> date:
    """
    "YYYYMMDD" 문자열을 date로 파싱한다. 실패하면 ValueError를 그대로 던진다.

    # WHY 별도 함수로 분리?
    #   filter_active_festivals()의 try/except 블록을 짧게 유지해
    #   "무엇을 파싱하다 실패했는가"를 한눈에 보이게 하기 위함.
    """
    return datetime.strptime(value, "%Y%m%d").date()


def filter_active_festivals(
    festivals: list[FestivalInfo], today: date, *, limit: int = 3
) -> list[FestivalInfo]:
    """
    today가 event_start_date~event_end_date 사이(경계 포함)인 축제만 골라
    상위 limit개까지 반환한다.

    # WHY 날짜 문자열이 잘못된 항목을 예외 없이 스킵하는가(Graceful Degradation)?
    #   축제 데이터는 외부(팀원)가 채우는 값이라 형식 오류가 섞일 수 있다.
    #   축제 1건의 오타 때문에 코스 생성 전체가 죽으면 안 되므로, 그 항목만
    #   조용히 제외하고 나머지는 정상 처리한다.
    # WHY event_start_date 오름차순 정렬?
    #   상한(limit)에 걸려 일부만 노출해야 할 때, 사용자에게 더 임박했거나
    #   이미 진행 중인(시작일이 가까운) 축제를 우선 보여주는 편이 실용적이다.
    #   "언제 끝나는지"보다 "언제 시작했는지"가 지금 이 순간의 관련도를 더 잘 반영한다.
    """
    active: list[tuple[date, FestivalInfo]] = []
    for festival in festivals:
        try:
            start = _parse_yyyymmdd(festival.event_start_date)
            end = _parse_yyyymmdd(festival.event_end_date)
        except (ValueError, TypeError):
            continue  # 잘못된 날짜 문자열 → 해당 항목만 스킵
        if start <= today <= end:
            active.append((start, festival))

    active.sort(key=lambda pair: pair[0])
    return [festival for _, festival in active[:limit]]


def build_season_context(
    today: date | None = None, festivals: list[FestivalInfo] | None = None
) -> str:
    """
    L2(계절) + L3(축제) 컨텍스트 블록을 하나의 문자열로 조립한다.

    # WHY festivals가 None/빈 리스트(또는 필터 후 빈 리스트)면 계절 한 줄만 반환하는가?
    #   기존 build_context_message()가 만들던 "현재 계절: {get_season(today)}" 한 줄과
    #   완전히 동일해야 회귀 0이 보장된다(설계 spec §6.3). 축제가 실제로 없을 때도
    #   빈 헤더만 붙는 것보다, 아무것도 덧붙이지 않는 편이 더 정직한 출력이다.
    """
    resolved_today = today or date.today()
    lines = [f"현재 계절: {get_season(today)}"]

    active = filter_active_festivals(festivals, resolved_today) if festivals else []
    if active:
        lines.append("[진행 중인 축제·행사]")
        for festival in active:
            place_part = f", {festival.event_place}" if festival.event_place else ""
            lines.append(
                f"  - {festival.name} ({festival.event_start_date}~{festival.event_end_date}{place_part})"
            )

    return "\n".join(lines)
