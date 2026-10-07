"""Deterministic, coordinate-based schedule policy for generated courses.

This module deliberately estimates only the outbound route: origin -> first stop
and each consecutive stop.  It excludes the return trip and does not call a live
directions API, so callers must present the result as an estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
import re
from typing import Literal

from traveler.rag.preference import extract_user_request
from traveler.rag.types import ContextData, SpotCandidate
from traveler.validation.schema import RecommendedPlaceModel, StreamingCourseModel


DEFAULT_MIN_PLACES = 3
DEFAULT_MAX_PLACES = 5
_CATEGORY_DEFAULT_STAY_MIN = {
    "관광": 60,
    "문화": 60,
    "음식": 60,
    "쇼핑": 40,
    "축제": 90,
}
_EXPLICIT_COUNT_RE = re.compile(
    r"(?<![\d.])(10|[1-9])\s*(?:곳|개|장소)"
    r"(?=$|[\s.,!?]|만|으로|로|을|를|짜리|추천|코스)"
)
_TOTAL_COUNT_PREFIX_RE = re.compile(r"(?:총|전체|합계|모두|합쳐서)\s*$")
_PARTIAL_COUNT_RE = re.compile(r"추가|넣|포함|빼|제외|교체|바꾸|바꿔|대신")
_TRANSPORT_SPEED_KMH = {
    "자차": 30.0,
    "자동차": 30.0,
    "car": 30.0,
    "도보": 4.0,
    "walk": 4.0,
    "walking": 4.0,
}


@dataclass
class ScheduleResult:
    """The policy output; the caller owns how its note/metadata is surfaced."""

    places: list[RecommendedPlaceModel]
    budget_min: int
    travel_min: float | None
    stay_min: int
    total_min: float | None
    status: Literal["within_budget", "unverified", "no_feasible"]
    adjusted: bool
    note: str


def _explicit_place_count(user_query: str) -> int | None:
    """Read whole-course totals, never a partial edit or the suffix of a range."""
    # WHY: 수량을 추가 의도와 분리하지 않으면 '카페 1곳 추가'가 전체 1곳으로 잘린다.
    query = extract_user_request(user_query or "")
    matches = [match for match in _EXPLICIT_COUNT_RE.finditer(query)
               if not re.search(r"[\d~～–-]\s*$", query[:match.start()])
               and not re.match(r"\s*[~～–-]", query[match.end():])]
    explicit_totals = []
    for index, match in enumerate(matches):
        next_start = matches[index + 1].start() if index + 1 < len(matches) else len(query)
        # '총 2곳 추가'의 총은 추가량이다. 해당 숫자 뒤의 편집 동작부터 구분한다.
        suffix = query[match.end():next_start]
        if (_TOTAL_COUNT_PREFIX_RE.search(query[:match.start()])
                and not _PARTIAL_COUNT_RE.search(suffix)):
            explicit_totals.append(int(match.group(1)))
    if explicit_totals:
        return explicit_totals[-1]
    if _PARTIAL_COUNT_RE.search(query) or len(matches) > 1:
        return None
    for match in matches:
        suffix = query[match.end():].lstrip()
        if (not suffix or suffix.startswith(('만', '코스', '추천', '짜리'))
                or re.match(r'(?:으로|로)\s*(?:줄|늘|해|하|만들|짜)', suffix)):
            return int(match.group(1))
    return None


def default_stay_minutes(category: str) -> int:
    """Use one category fallback for generation and add/replace edits."""
    return _CATEGORY_DEFAULT_STAY_MIN.get(category, 60)


def _default_stay_min(candidate: SpotCandidate | None) -> int:
    return default_stay_minutes(candidate.category) if candidate else 60


def _valid_coordinate(lat: object, lng: object) -> bool:
    """Reject sentinel zeroes as well as absent, non-finite, and out-of-range values."""
    if isinstance(lat, bool) or isinstance(lng, bool):
        return False
    if not isinstance(lat, (int, float)) or not isinstance(lng, (int, float)):
        return False
    if not isfinite(lat) or not isfinite(lng):
        return False
    return lat != 0 and lng != 0 and -90 <= lat <= 90 and -180 <= lng <= 180


def _transport_speed_kmh(transport: object) -> float | None:
    if not isinstance(transport, str):
        return None
    return _TRANSPORT_SPEED_KMH.get(transport.strip().lower())


def _budget_min(ctx: ContextData) -> int | None:
    minutes = getattr(ctx, 'duration_minutes', None)
    if minutes is not None:
        return minutes if type(minutes) is int and 1 <= minutes <= 720 else None
    hours = ctx.duration_hours
    if isinstance(hours, bool) or not isinstance(hours, (int, float)):
        return None
    if not isfinite(hours) or hours <= 0:
        return None
    return int(hours * 60)


_FAR_ORIGIN_THRESHOLD_KM = 100.0


def _origin_coordinates(ctx: ContextData) -> tuple[object, object]:
    """Use the actual incoming GPS when the integration layer supplies it.

    ``current_lat/current_lng`` remains the compatibility fallback because older
    direct callers only have the retrieval/search-center coordinates.
    """
    origin_lat = getattr(ctx, "schedule_origin_lat", None)
    origin_lng = getattr(ctx, "schedule_origin_lng", None)
    if origin_lat is not None or origin_lng is not None:
        return origin_lat, origin_lng
    return ctx.current_lat, ctx.current_lng


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    from math import radians, cos, sin, atan2, sqrt
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def _estimate_leg_minutes(
    start_lat: float,
    start_lng: float,
    end_lat: float,
    end_lng: float,
    speed_kmh: float,
) -> float:
    """Use the existing MVP coordinate estimator lazily to avoid a heavy import at startup."""
    from traveler.rag.hybrid_search import estimate_travel_minutes

    return estimate_travel_minutes(
        start_lat,
        start_lng,
        end_lat,
        end_lng,
        average_speed_kmh=speed_kmh,
    )


def _with_stay_defaults(
    places: list[RecommendedPlaceModel], candidates_by_id: dict[str, SpotCandidate]
) -> tuple[list[RecommendedPlaceModel], bool]:
    updated: list[RecommendedPlaceModel] = []
    adjusted = False
    for place in places:
        if place.duration_min is None:
            place = place.model_copy(
                update={"duration_min": _default_stay_min(candidates_by_id.get(place.place_id))}
            )
            adjusted = True
        updated.append(place)
    return updated, adjusted


def _resequence(places: list[RecommendedPlaceModel]) -> list[RecommendedPlaceModel]:
    return [place.model_copy(update={"visit_sequence": index}) for index, place in enumerate(places, 1)]


def _selected_unique(
    course: StreamingCourseModel, max_places: int
) -> tuple[list[RecommendedPlaceModel], bool]:
    unique: list[RecommendedPlaceModel] = []
    seen_ids: set[str] = set()
    adjusted = False
    for place in course.recommended_places:
        if place.place_id in seen_ids or len(unique) >= max_places:
            adjusted = True
            continue
        seen_ids.add(place.place_id)
        unique.append(place)
    resequenced = _resequence(unique)
    if [place.visit_sequence for place in unique] != [place.visit_sequence for place in resequenced]:
        adjusted = True
    return resequenced, adjusted


def _route_totals(
    places: list[RecommendedPlaceModel],
    ctx: ContextData,
    candidates_by_id: dict[str, SpotCandidate],
    speed_kmh: float,
) -> tuple[float, int, float] | None:
    """Return travel, stay, total or ``None`` when any selected leg is unverifiable."""
    origin_lat, origin_lng = _origin_coordinates(ctx)
    if not _valid_coordinate(origin_lat, origin_lng):
        return None

    previous_lat, previous_lng = origin_lat, origin_lng
    travel_min = 0.0
    stay_min = 0
    for place in places:
        candidate = candidates_by_id.get(place.place_id)
        if candidate is None or not _valid_coordinate(candidate.lat, candidate.lng):
            return None
        try:
            travel_min += _estimate_leg_minutes(
                previous_lat, previous_lng, candidate.lat, candidate.lng, speed_kmh
            )
        except (ImportError, TypeError, ValueError, ZeroDivisionError):
            return None
        stay_min += place.duration_min or 0
        previous_lat, previous_lng = candidate.lat, candidate.lng
    return round(travel_min, 1), stay_min, round(travel_min + stay_min, 1)


def _note_within_budget(
    travel_min: float,
    stay_min: int,
    total_min: float,
    budget_min: int,
    transport: object,
    speed_kmh: float,
    extra: str = "",
) -> str:
    transport_label = transport.strip() if isinstance(transport, str) else "알 수 없는 이동 수단"
    note = (
        f"예상 총 {total_min:.0f}분(이동 {travel_min:.0f}분 + 체류 {stay_min}분) / "
        f"가용 {budget_min}분 — 출발지→첫 장소와 장소 간 이동 포함, 복귀 이동 제외. "
        f"좌표 직선거리·{transport_label} 평균{speed_kmh:g}km/h 추정(도로/교통/주차 미반영)"
    )
    return f"{note}. {extra}" if extra else note


def _unverified_result(
    places: list[RecommendedPlaceModel], budget_min: int, stay_min: int, adjusted: bool, reason: str
) -> ScheduleResult:
    return ScheduleResult(
        places=places,
        budget_min=budget_min,
        travel_min=None,
        stay_min=stay_min,
        total_min=None,
        status="unverified",
        adjusted=adjusted,
        note=(
            f"예상 이동시간을 검증할 수 없어 예산 충족 여부를 확인하지 않았습니다: {reason}. "
            "출발지→첫 장소와 장소 간 이동이 대상이며, 복귀 이동은 제외됩니다."
        ),
    )


def _no_feasible_result(budget_min: int, adjusted: bool, note: str) -> ScheduleResult:
    return ScheduleResult(
        places=[],
        budget_min=budget_min,
        travel_min=None,
        stay_min=0,
        total_min=None,
        status="no_feasible",
        adjusted=adjusted,
        note=note,
    )


def apply_schedule(
    course: StreamingCourseModel, ctx: ContextData, *, fill_minimum: bool = True
) -> ScheduleResult:
    """Apply the conservative 3--5 place and estimated-time policy.

    This is intentionally greedy: it preserves the LLM-selected order, trims only
    from the end when the time budget is exceeded, then appends ranked candidates
    that fit.  It does not claim to find an optimal route.
    """
    explicit_count = _explicit_place_count(ctx.user_query)
    maximum = explicit_count if explicit_count is not None else (DEFAULT_MAX_PLACES if fill_minimum else 10)
    # De-duplicate before grounding, but apply the requested/default maximum after
    # dropping unknown IDs so five hallucinations cannot hide a later real choice.
    places, adjusted = _selected_unique(course, 10)
    had_selected_places = bool(places)
    # A legacy retriever path can still emit an int content id; the output schema
    # is string-based, so normalize only the lookup key at this boundary.
    candidates_by_id = {str(spot.place_id): spot for spot in ctx.candidate_spots}
    budget_min = _budget_min(ctx)
    # `_build_fallback()` historically used place_id="1" when retrieval returned
    # no candidates.  That placeholder is not a real recommendation and must not
    # escape as a scheduled stop.
    if not candidates_by_id:
        return _no_feasible_result(budget_min or 0, True, "추천 후보가 없어 가짜 장소를 만들지 않았습니다.")

    grounded_places = [place for place in places if place.place_id in candidates_by_id]
    if len(grounded_places) != len(places):
        adjusted = True
    places = _resequence(grounded_places[:maximum])
    if len(grounded_places) > maximum:
        adjusted = True

    places, defaults_added = _with_stay_defaults(places, candidates_by_id)
    adjusted = adjusted or defaults_added
    original_stays = {place.place_id: place.duration_min for place in places}
    _grounded_full = grounded_places  # kept for far-origin retry

    # A server-built no_feasible result has no selection by design.  Only replace
    # a removed *unknown* LLM selection during an initial generation, never an
    # intentionally empty result.
    if not places and not had_selected_places:
        return _no_feasible_result(
            budget_min or 0, adjusted, "선택된 장소가 없어 후보를 임의로 추가하지 않았습니다."
        )

    if budget_min is None:
        if not places:
            return _no_feasible_result(0, adjusted, "유효한 선택 장소와 가용 시간이 없어 코스를 만들지 않았습니다.")
        return _unverified_result(
            places,
            0,
            sum(place.duration_min or 0 for place in places),
            adjusted,
            "가용 시간이 없거나 유효하지 않음",
        )

    # Even when travel cannot be estimated, known stay time is a hard lower bound.
    trimmed_for_budget = False
    while places and sum(place.duration_min or 0 for place in places) > budget_min:
        places = places[:-1]
        adjusted = True
        trimmed_for_budget = True

    if not places and not fill_minimum:
        return _no_feasible_result(
            budget_min, adjusted, "예산 안에 들어오는 선택 장소가 없어 코스를 만들지 않았습니다."
        )

    speed_kmh = _transport_speed_kmh(ctx.transport)
    if speed_kmh is None:
        if not places:
            return _no_feasible_result(budget_min, adjusted, "이동시간을 검증할 수 없어 새 후보를 추가하지 않았습니다.")
        return _unverified_result(
            places,
            budget_min,
            sum(place.duration_min or 0 for place in places),
            adjusted,
            f"이동 수단 {ctx.transport!r}의 신뢰할 수 있는 속도 모델이 없음",
        )

    origin_lat, origin_lng = _origin_coordinates(ctx)
    if not _valid_coordinate(origin_lat, origin_lng):
        if not places:
            return _no_feasible_result(budget_min, adjusted, "출발지 좌표가 없어 새 후보를 추가하지 않았습니다.")
        return _unverified_result(
            places,
            budget_min,
            sum(place.duration_min or 0 for place in places),
            adjusted,
            "출발지 또는 선택 장소의 좌표가 없거나 유효하지 않음",
        )

    totals = (0.0, 0, 0.0)
    if places:
        totals = _route_totals(places, ctx, candidates_by_id, speed_kmh)
    if totals is None:
        return _unverified_result(
            places,
            budget_min,
            sum(place.duration_min or 0 for place in places),
            adjusted,
            "출발지 또는 선택 장소의 좌표가 없거나 유효하지 않음",
        )

    # Time is a hard limit.  Preserve the selected sequence and only drop its tail.
    while places and totals[2] > budget_min:
        places = places[:-1]
        adjusted = True
        trimmed_for_budget = True
        if places:
            totals = _route_totals(places, ctx, candidates_by_id, speed_kmh)
            if totals is None:
                return _unverified_result(
                    places,
                    budget_min,
                    sum(place.duration_min or 0 for place in places),
                    adjusted,
                    "출발지 또는 선택 장소의 좌표가 없거나 유효하지 않음",
                )

    target_count = explicit_count if explicit_count is not None else DEFAULT_MIN_PLACES
    skipped_for_budget = False
    skipped_for_coordinates = False
    if fill_minimum and len(places) < target_count:
        selected_ids = {place.place_id for place in places}
        for candidate in ctx.candidate_spots:
            if len(places) >= target_count:
                break
            candidate_id = str(candidate.place_id)
            if candidate_id in selected_ids:
                continue
            if not _valid_coordinate(candidate.lat, candidate.lng):
                skipped_for_coordinates = True
                continue
            candidate_place = RecommendedPlaceModel(
                place_id=candidate_id,
                place_name=candidate.place_name or None,
                visit_sequence=1,
                duration_min=original_stays.get(candidate_id, _default_stay_min(candidate)),
            )
            # 방금 제외한 후보도 달라진 경로에서는 가능할 수 있다. 원래 체류를 유지하고
            # 전체 경로를 재계산하므로 시간을 줄이거나 검증을 우회해서 재삽입하지 않는다.
            proposed = places + [candidate_place]
            proposed_totals = _route_totals(proposed, ctx, candidates_by_id, speed_kmh)
            if proposed_totals is None:
                skipped_for_coordinates = True
                continue
            if proposed_totals[2] > budget_min:
                skipped_for_budget = True
                continue
            places = proposed
            totals = proposed_totals
            selected_ids.add(candidate_id)
            adjusted = True

    if not places:
        # WHY: 사용자 GPS가 검색 지역과 매우 멀 때(예: 서울 사용자가 "경주 코스" 요청)
        # origin→첫장소 이동이 예산을 전부 소진한다. 이 경우 검색 중심을 임시 출발지로
        # 재시도하여 로컬 코스를 생성한다. ctx.schedule_origin_lat 값은 변경하지 않는다.
        _retry_with_search_center = False
        _o_lat, _o_lng = _origin_coordinates(ctx)
        _s_lat, _s_lng = ctx.current_lat, ctx.current_lng
        if (fill_minimum
                and _valid_coordinate(_o_lat, _o_lng)
                and _valid_coordinate(_s_lat, _s_lng)
                and _haversine_km(_o_lat, _o_lng, _s_lat, _s_lng) > _FAR_ORIGIN_THRESHOLD_KM):
            _retry_with_search_center = True

        if _retry_with_search_center:
            _saved_origin_lat = getattr(ctx, "schedule_origin_lat", None)
            _saved_origin_lng = getattr(ctx, "schedule_origin_lng", None)
            ctx.schedule_origin_lat = _s_lat
            ctx.schedule_origin_lng = _s_lng
            try:
                places = _resequence(_grounded_full[:maximum])
                places, _d2 = _with_stay_defaults(places, candidates_by_id)
                while places and sum(p.duration_min or 0 for p in places) > budget_min:
                    places = places[:-1]
                    adjusted = True
                    trimmed_for_budget = True
                if places:
                    totals = _route_totals(places, ctx, candidates_by_id, speed_kmh)
                    if totals is None:
                        places = []
                    else:
                        while places and totals[2] > budget_min:
                            places = places[:-1]
                            adjusted = True
                            trimmed_for_budget = True
                            if places:
                                totals = _route_totals(places, ctx, candidates_by_id, speed_kmh)
                                if totals is None:
                                    places = []
                                    break
            finally:
                ctx.schedule_origin_lat = _saved_origin_lat
                ctx.schedule_origin_lng = _saved_origin_lng

    if not places:
        return _no_feasible_result(
            budget_min,
            adjusted,
            "현재 후보 순서와 좌표 추정 기준에서 예산 안에 들어오는 실제 후보를 찾지 못했습니다.",
        )

    places = _resequence(places)
    travel_min, stay_min, total_min = totals
    exception_note = "시간 예산을 초과한 장소를 줄였습니다" if trimmed_for_budget else ""
    if not exception_note and len(places) < target_count:
        if not fill_minimum:
            exception_note = "수정 요청의 장소 수를 유지했습니다"
        elif skipped_for_budget:
            exception_note = "시간 예산 때문에 기본 최소 장소 수보다 적게 유지했습니다"
        elif skipped_for_coordinates:
            exception_note = "유효 좌표가 있는 후보가 부족해 기본 최소 장소 수보다 적게 유지했습니다"
        else:
            exception_note = "후보가 부족해 기본 최소 장소 수보다 적게 유지했습니다"
    if getattr(ctx, 'transport_assumed', False):
        exception_note = ". ".join(filter(None, [exception_note, "이동수단 미지정으로 자차를 가정했습니다"]))

    return ScheduleResult(
        places=places,
        budget_min=budget_min,
        travel_min=travel_min,
        stay_min=stay_min,
        total_min=total_min,
        status="within_budget",
        adjusted=adjusted,
        note=_note_within_budget(
            travel_min,
            stay_min,
            total_min,
            budget_min,
            ctx.transport,
            speed_kmh,
            exception_note,
        ),
    )
