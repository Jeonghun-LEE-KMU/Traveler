"""생성·수정·로컬 QA가 공유하는 최종 응답 및 SSE 오류 직렬화."""

import json

from traveler.core.course_schedule import _valid_coordinate
from traveler.rag.types import ContextData
from traveler.validation.schema import StreamingCourseModel


def course_payload(course: StreamingCourseModel, ctx: ContextData, *, bot_message: str | None = None) -> dict:
    """서버가 검증한 코스와 같은 후보 ID에서만 지도 좌표를 가져온다."""
    # WHY: 생성/수정의 좌표 보강 규칙이 달라지면 시간 계산과 지도가 서로 다른 장소를 쓴다.
    candidates = {str(spot.place_id): spot for spot in ctx.candidate_spots}
    places = []
    for place in course.recommended_places:
        spot = candidates.get(place.place_id)
        valid = spot is not None and _valid_coordinate(spot.lat, spot.lng)
        places.append({
            'place_id': place.place_id, 'visit_sequence': place.visit_sequence,
            'place_name': place.place_name or (spot.place_name if spot else None),
            'duration_min': place.duration_min,
            'lat': spot.lat if valid else None, 'lng': spot.lng if valid else None,
            'recommend_reason': place.recommend_reason or None,
            'image_url': spot.image_url if spot else None,
            'event_start_date': spot.event_start_date if spot else None,
            'event_end_date': spot.event_end_date if spot else None,
            'use_time': spot.use_time if spot else None,
            'rest_date': spot.rest_date if spot else None,
        })
    return {
        'bot_message': course.bot_message if bot_message is None else bot_message,
        'recommended_places': places,
        'time_budget': course.time_budget.model_dump() if course.time_budget else None,
    }


def application_error_event(code: str, message: str, *, retryable: bool) -> str:
    """브라우저의 연결 오류와 구분되는 명시적 애플리케이션 오류 이벤트."""
    data = {'error': {'code': code, 'message': message, 'retryable': retryable}}
    return 'event: application_error\ndata: ' + json.dumps(data, ensure_ascii=False) + '\n\n'
