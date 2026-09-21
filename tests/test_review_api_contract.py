"""실제 API와 큐에 기록되는 최종/오류 SSE의 회귀 테스트 (외부 I/O 없음)."""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

import main
from tests.test_main_chat_persist import _patch_llm_layer
from tests.test_schedule_integration import _context, _raw
from traveler.validation.schema import StreamingCourseModel


@pytest.fixture(autouse=True)
def isolate_services(monkeypatch):
    async def weather(*args): return '맑음'
    async def visited(*args): return set()
    async def prefs(*args): return {}
    async def save(*args): return None
    monkeypatch.setattr(main, 'get_weather', weather)
    monkeypatch.setattr(main, 'get_visited', visited)
    monkeypatch.setattr(main, 'get_prefs', prefs)
    monkeypatch.setattr(main, 'update_prefs', save)
    monkeypatch.setattr(main, 'rag_get_context', lambda *a, **kw: (_context(step_km=1), None))
    yield
    for sid in list(main._session_courses):
        if sid.startswith('review-'):
            main._session_courses.pop(sid, None)
            main.memory_store.pop(sid, None)


def events(text):
    result = []
    for block in text.strip().split('\n\n'):
        if not block: continue
        lines = block.splitlines()
        event = next((line[7:] for line in lines if line.startswith('event: ')), 'message')
        data = '\n'.join(line[6:] for line in lines if line.startswith('data: '))
        result.append((event, json.loads(data)))
    return result


@pytest.mark.parametrize('query,minutes', [('30분 코스', 30), ('1.5시간 코스', 90), ('4시간 30분 코스',270)])
@pytest.mark.parametrize('endpoint', ['/api/chat','/api/chat/revise'])
def test_minutes_apply_at_both_api_boundaries(monkeypatch, query, minutes, endpoint):
    sid='review-minute'
    original=StreamingCourseModel.model_validate_json(_raw(count=3))
    main._session_courses[sid]=(original,_context(step_km=1),'타지')
    _patch_llm_layer(monkeypatch, [_raw(count=3)])
    async def revise(*a,**kw): return original
    monkeypatch.setattr(main,'revise_course',revise)
    with TestClient(main.app) as client:
        payload={'user_message':query,'session_id':sid}
        response=client.get(endpoint, params={**payload,'lat':35.0,'lon':129.0}) if endpoint.endswith('chat') else client.post(endpoint,json=payload)
    assert response.status_code==200
    body=events(response.text)[-1][1] if endpoint.endswith('chat') else response.json()
    assert body['time_budget']['budget_min']==minutes
    assert body['time_budget']['total_min'] is None or body['time_budget']['total_min']<=minutes


def test_invalid_chat_is_named_error_without_service_calls(monkeypatch):
    async def forbidden(*a): raise AssertionError('Invalid input must not call external services')
    monkeypatch.setattr(main,'get_weather',forbidden)
    with TestClient(main.app) as client:
        response=client.get('/api/chat',params={'user_message':'0분 코스','session_id':'review-invalid'})
    assert response.status_code==200
    [(kind, body)]=events(response.text)
    assert kind=='application_error'
    assert body['error']['code']=='INVALID_REQUEST'
    assert body['error']['retryable'] is False
    assert body['error']['message']
    assert 'time_budget' not in body


def test_invalid_revise_is_422_and_keeps_saved_course():
    sid='review-invalid-revise'
    original=StreamingCourseModel.model_validate_json(_raw(count=1))
    main._session_courses[sid]=(original,_context(),'타지')
    with TestClient(main.app) as client:
        response=client.post('/api/chat/revise',json={'session_id':sid,'user_message':'0분만 가능해'})
    assert response.status_code==422
    assert response.json()['detail']
    assert main._session_courses[sid][0] is original


@pytest.mark.asyncio
async def test_producer_exception_has_explicit_error_then_sentinel(monkeypatch):
    def broken(*a,**kw): raise RuntimeError('secret-internal-error')
    monkeypatch.setattr(main,'ChatOpenAI',broken)
    queue=asyncio.Queue()
    await main._stream_and_persist(queue,'review-error','타지','문맥',_context())
    chunks=[queue.get_nowait() for _ in range(queue.qsize())]
    assert chunks[-1] is None
    [(kind,body)]=events(''.join(chunks[:-1]))
    assert kind=='application_error'
    assert body['error']['code']=='GENERATION_FAILED'
    assert 'secret-internal-error' not in body['error']['message']


@pytest.mark.asyncio
async def test_no_feasible_final_sse_clears_previous_course(monkeypatch):
    sid='review-empty'
    main._session_courses[sid]=(StreamingCourseModel.model_validate_json(_raw(count=1)),_context(),'타지')
    _patch_llm_layer(monkeypatch,[_raw(count=1,stay=120)])
    queue=asyncio.Queue()
    await main._stream_and_persist(queue,sid,'타지','문맥',_context(hours=1,count=1))
    chunks=[queue.get_nowait() for _ in range(queue.qsize())]
    kind,body=events(''.join(chunks[:-1]))[-1]
    assert kind=='message'
    assert body['recommended_places']==[]
    assert body['time_budget']['status']=='no_feasible'
    assert sid not in main._session_courses


@pytest.mark.asyncio
async def test_invalid_candidate_coordinates_are_null_on_wire(monkeypatch):
    ctx=_context(count=1)
    ctx.candidate_spots[0].lat=0.0
    _patch_llm_layer(monkeypatch,[_raw(count=1)])
    queue=asyncio.Queue()
    await main._stream_and_persist(queue,'review-coordinate','타지','문맥',ctx)
    chunks=[queue.get_nowait() for _ in range(queue.qsize())]
    _,body=events(''.join(chunks[:-1]))[-1]
    assert body['time_budget']['status']=='unverified'
    assert body['recommended_places'][0]['lat'] is None
    assert body['recommended_places'][0]['lng'] is None


def test_setup_exception_is_named_error_without_internal_details(monkeypatch):
    async def broken(*args): raise RuntimeError('private-service-address')
    monkeypatch.setattr(main, 'get_weather', broken)
    with TestClient(main.app) as client:
        response = client.get('/api/chat', params={'user_message': '경주 4시간', 'session_id': 'review-setup'})
    [(kind, body)] = events(response.text)
    assert kind == 'application_error'
    assert body['error']['code'] == 'GENERATION_FAILED'
    assert 'private-service-address' not in response.text


def test_missing_gps_is_unverified_without_erasing_place_coordinates(monkeypatch):
    _patch_llm_layer(monkeypatch, [_raw(count=3)])
    with TestClient(main.app) as client:
        response = client.get('/api/chat', params={'user_message': '경주 4시간', 'session_id': 'review-gps'})
    _, body = events(response.text)[-1]
    assert body['time_budget']['status'] == 'unverified'
    assert body['time_budget']['total_min'] is None
    assert body['recommended_places'][0]['lat'] is not None


@pytest.mark.parametrize('query,count', [('카페 1곳 추가해줘', 4), ('총 2곳으로 줄여줘', 2)])
def test_revision_count_policy_at_api_boundary(monkeypatch, query, count):
    sid = 'review-count'
    original = StreamingCourseModel.model_validate_json(_raw(count=3))
    main._session_courses[sid] = (original, _context(step_km=1), '타지')
    # 편집 계층의 결과를 실제 API 후처리에 통과시킨다. LLM 분류 품질 검증은 별도다.
    async def edited(*args, **kwargs):
        return StreamingCourseModel.model_validate_json(_raw(count=4))
    monkeypatch.setattr(main, 'revise_course', edited)
    with TestClient(main.app) as client:
        response = client.post('/api/chat/revise', json={'user_message': query, 'session_id': sid})
    assert len(response.json()['recommended_places']) == count


def test_revision_retains_minute_budget_and_replaces_assumed_transport(monkeypatch):
    sid = 'review-retain'
    original = StreamingCourseModel.model_validate_json(_raw(count=3))
    ctx = _context(step_km=1)
    ctx.transport_assumed = True
    main._session_courses[sid] = (original, ctx, '타지')
    async def unchanged(message, course, *args, **kwargs): return course
    monkeypatch.setattr(main, 'revise_course', unchanged)
    with TestClient(main.app) as client:
        client.post('/api/chat/revise', json={'user_message': '1.5시간 코스', 'session_id': sid})
        response = client.post('/api/chat/revise', json={'user_message': '기차로 이동할래', 'session_id': sid})
    assert response.json()['time_budget']['budget_min'] == 90
    assert response.json()['time_budget']['status'] == 'unverified'
    assert main._session_courses[sid][1].transport_assumed is False
