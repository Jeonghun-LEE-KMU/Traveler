"""
BotMessageExtractor 단위 테스트 — #13 대응.

WHY 이 테스트가 필요한가:
  main.py가 LLM 원시 토큰을 그대로 SSE로 흘려보내면 프론트에 JSON 문법 조각
  (`{"bot_message": "안녕...`)이 그대로 노출된다. 이 테스트는 청크 경계가
  어디서 끊기든(1글자·3글자 단위) 최종적으로 원본 bot_message 텍스트만
  복원되고, 중간에 JSON 문법이 절대 새어나가지 않는지 확인한다.
"""

import json

from traveler.core.stream_text import BotMessageExtractor

_PAYLOAD = {
    "bot_message": '경주 "야경" 코스\n좋아요 가',
    "recommended_places": [{"place_id": "1", "visit_sequence": 1}],
}
_FULL_JSON = json.dumps(_PAYLOAD, ensure_ascii=False)


def _feed_all(chunks: list[str]) -> str:
    extractor = BotMessageExtractor()
    visible_parts = []
    for chunk in chunks:
        visible = extractor.feed(chunk)
        # ★ 핵심: 어떤 청크 경계에서도 JSON 문법 조각이 새어나가면 안 된다.
        assert "{" not in visible
        assert "recommended_places" not in visible
        visible_parts.append(visible)
    return "".join(visible_parts)


def test_one_char_chunks_reconstruct_bot_message_exactly():
    chunks = list(_FULL_JSON)  # 1글자씩
    assert _feed_all(chunks) == _PAYLOAD["bot_message"]


def test_three_char_chunks_reconstruct_bot_message_exactly():
    chunks = [_FULL_JSON[i:i + 3] for i in range(0, len(_FULL_JSON), 3)]
    assert _feed_all(chunks) == _PAYLOAD["bot_message"]


def test_recommended_places_before_bot_message_still_works():
    """필드 순서가 뒤바뀐 JSON(레거시/엣지 케이스)도 bot_message만 정확히 뽑아내는지 확인."""
    payload = {
        "recommended_places": [{"place_id": "1", "visit_sequence": 1}],
        "bot_message": "안녕하세요 반가워요",
    }
    full_json = json.dumps(payload, ensure_ascii=False)
    chunks = [full_json[i:i + 4] for i in range(0, len(full_json), 4)]
    assert _feed_all(chunks) == payload["bot_message"]


def test_no_bot_message_yields_empty_string():
    """bot_message 필드가 아예 없는 스트림은 항상 빈 문자열만 반환하는지 확인."""
    extractor = BotMessageExtractor()
    result = extractor.feed('{"foo": "bar"}')
    assert result == ""
