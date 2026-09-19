"""통제된 로컬 QA도 실제 시간 정책과 응답 직렬화를 통과해야 한다."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scripts.pm_fixture_server import app, sessions


@pytest.fixture
def client():
    sessions.clear()
    with TestClient(app) as client:
        yield client
    sessions.clear()


def final_chat(client, query, *, gps=True):
    params = {'user_message': query, 'session_id': 'fixture-test'}
    if gps:
        params.update(lat=35.856, lon=129.224)
    response = client.get('/api/chat', params=params)
    block = response.text.strip().split('\n\n')[-1]
    return json.loads(next(line[6:] for line in block.splitlines() if line.startswith('data: ')))


def test_controlled_normal_course(client):
    result = final_chat(client, '경주 4시간 코스')
    assert len(result['recommended_places']) == 4
    assert result['time_budget']['status'] == 'within_budget'
    assert result['time_budget']['total_min'] <= 240


@pytest.mark.parametrize('revision', [False, True])
def test_controlled_empty_clears_saved_course(client, revision):
    final_chat(client, '경주 4시간 코스')
    query = '테스트: 추천불가 1시간 코스'
    result = (client.post('/api/chat/revise', json={'user_message': query, 'session_id': 'fixture-test'}).json()
              if revision else final_chat(client, query))
    assert result['recommended_places'] == []
    assert result['time_budget']['status'] == 'no_feasible'
    assert 'fixture-test' not in sessions
    response = client.post('/api/chat/revise', json={'user_message': '다시', 'session_id': 'fixture-test'})
    assert response.status_code == 404


def test_missing_origin_keeps_valid_place_coordinates(client):
    result = final_chat(client, '경주 4시간 코스', gps=False)
    assert result['time_budget']['status'] == 'unverified'
    assert result['time_budget']['total_min'] is None
    assert result['recommended_places'][0]['lat']


def test_transit_and_decimal_budget(client):
    result = final_chat(client, '기차로 경주 1.5시간 코스')
    assert result['time_budget']['budget_min'] == 90
    assert result['time_budget']['status'] == 'unverified'


def test_error_contract_and_reset(client):
    result = final_chat(client, '0분 코스')
    assert result['error']['code'] == 'INVALID_REQUEST'
    result = final_chat(client, '테스트: 오류')
    assert result['error']['code'] == 'GENERATION_FAILED'
    final_chat(client, '4시간 코스')
    response = client.post('/api/chat/revise', json={'user_message': '0분 코스', 'session_id': 'fixture-test'})
    assert response.status_code == 422
    assert response.json()['detail']
    client.post('/api/session/fixture-test/clear')
    assert not sessions


def test_documented_samples_match_real_fixture_responses(client):
    samples = json.loads((Path(__file__).parents[1] / 'docs/fixtures/final_response_samples.json').read_text())
    for sample in samples['responses'].values():
        result = final_chat(client, sample['query'], gps=sample['gps_provided'])
        assert result == sample['data']
