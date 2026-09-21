"""PR47 정훈 담당 TC07/API 응답 검증용 서버. 운영 main.app에 포함하지 않는다.

실행: python -m uvicorn scripts.pm_fixture_server:app --host 127.0.0.1 --port 8765
RAG/LLM/DB를 호출하지 않는다. 합성 후보를 실제 일정 정책과 직렬화에 통과시킨다.
자연어 장소 편집 분류기는 테스트하지 않으며 시간·종료 계약 확인에만 사용한다.
"""

from dataclasses import replace
import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from traveler.core.course_generator import parse_course, validate_course_schedule
from traveler.core.course_response import application_error_event, course_payload
from traveler.rag.preference import extract_prefs_from_message
from traveler.rag.types import ContextData, SpotCandidate
from traveler.validation.schema import StreamingCourseModel


app = FastAPI(title='PR47 LOCAL FIXTURE — NOT PRODUCTION')
sessions: dict[str, tuple[StreamingCourseModel, ContextData]] = {}


def fixture_context(query: str, lat: float, lon: float) -> ContextData:
    prefs = extract_prefs_from_message(query)
    return ContextData(
        user_query=query, current_lat=35.856, current_lng=129.224,
        schedule_origin_lat=lat, schedule_origin_lng=lon, weather='맑음',
        duration_minutes=prefs.get('duration_minutes'),
        duration_hours=prefs.get('duration_hours', 5),
        duration_hours_min=None if 'duration_minutes' in prefs else 3,
        transport=prefs.get('transport', '자차'), transport_assumed='transport' not in prefs,
        candidate_spots=[SpotCandidate(
            f'fixture-{i}', f'[테스트] 장소 {i}', '관광', 0, False,
            lat=35.856 + i * .001, lng=129.224,
        ) for i in range(1, 5)],
    )


def fixture_raw(query: str) -> str:
    # TC07: 모든 후보가 가용시간을 초과하도록 체류를 고정한다 (실측값 아님).
    stay = 480 if '테스트: 추천불가' in query else 30
    return json.dumps({'bot_message': '[로컬 QA 합성 응답] 실제 장소 추천이 아닙니다.',
        'recommended_places': [
            {'place_id': f'fixture-{i}', 'visit_sequence': i, 'duration_min': stay}
            for i in range(1, 5)
        ]}, ensure_ascii=False)


def remember(session_id: str, course: StreamingCourseModel, ctx: ContextData) -> dict:
    if course.recommended_places:
        sessions[session_id] = course, ctx
    else:
        sessions.pop(session_id, None)
    return course_payload(course, ctx)


def stream_event(chunk: str) -> StreamingResponse:
    return StreamingResponse(iter([chunk]), media_type='text/event-stream',
                             headers={'Cache-Control': 'no-cache'})


@app.get('/api/chat')
def chat(user_message: str, session_id: str, lat: float = 0, lon: float = 0):
    try:
        ctx = fixture_context(user_message, lat, lon)
    except ValueError as error:
        return stream_event(application_error_event('INVALID_REQUEST', str(error), retryable=False))
    if '테스트: 오류' in user_message:
        return stream_event(application_error_event('GENERATION_FAILED', '로컬 QA 생성 실패 예시입니다.', retryable=True))
    course, _ = parse_course(fixture_raw(user_message), ctx)
    payload = remember(session_id, course, ctx)
    draft = 'data: ' + json.dumps({'bot_message': '확인 중입니다.', 'recommended_places': []}, ensure_ascii=False) + '\n\n'
    return stream_event(draft + 'data: ' + json.dumps(payload, ensure_ascii=False) + '\n\n')


class Revision(BaseModel):
    session_id: str
    user_message: str


@app.post('/api/chat/revise')
def revise(request: Revision):
    try:
        prefs = extract_prefs_from_message(request.user_message)
    except ValueError as error:
        raise HTTPException(422, detail=str(error)) from error
    if request.session_id not in sessions:
        raise HTTPException(404, detail='저장된 코스가 없습니다. 새 코스를 요청해 주세요.')
    course, ctx = sessions[request.session_id]
    updates = {key: prefs[key] for key in ('duration_minutes', 'duration_hours', 'transport', 'transport_assumed') if key in prefs}
    if 'duration_minutes' in updates:
        updates['duration_hours_min'] = None
    ctx = replace(ctx, user_query=request.user_message, **updates)
    if '테스트: 추천불가' in request.user_message:
        course = StreamingCourseModel.model_validate_json(fixture_raw(request.user_message))
    course = validate_course_schedule(course, ctx, fill_minimum=False)
    return remember(request.session_id, course, ctx)


@app.get('/health')
def health():
    return {'status': 'ok', 'note': 'LOCAL FIXTURE: no LLM, DB or live directions'}


@app.post('/api/session/{session_id}/clear')
def clear(session_id: str):
    sessions.pop(session_id, None)
    return {'session_id': session_id, 'ok': True}
