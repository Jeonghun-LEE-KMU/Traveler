"""LLM이 스트리밍하는 원시 JSON 조각에서 "bot_message" 필드의 디코딩된 텍스트만 점진적으로 뽑아낸다."""

import json


class BotMessageExtractor:
    """
    스트리밍 중인 (아직 완성되지 않은) JSON 문자열 버퍼를 매 청크마다 통째로
    재스캔해 "bot_message" 필드 값 중 새로 드러난 부분만 돌려준다.

    # WHY (#13): main.py가 GPT 원시 토큰을 그대로 SSE로 흘려보내면 프론트에
    # `{"bot_message": "안녕...` 같은 JSON 문법 조각이 그대로 노출된다.
    # 프론트(src/features/travel/model/streamText.ts)가 같은 문제를 이미
    # 같은 방식(버퍼 재스캔)으로 풀어놨다 — 백엔드도 동일 알고리즘을 따라
    # 두 쪽의 파싱 결과가 어긋나지 않게 한다.
    """

    def __init__(self) -> None:
        self._buf = ""     # 지금까지 받은 원시 텍스트 누적본 (미완성 JSON일 수 있음)
        self._emitted = 0  # 지금까지 내보낸 디코딩된 문자 수

    def feed(self, chunk: str) -> str:
        """새 청크를 누적하고, 새로 보이게 된 디코딩된 텍스트만 반환한다(없으면 "")."""
        self._buf += chunk
        visible = self._extract_visible()
        if visible is None:
            return ""
        new_text = visible[self._emitted:]
        self._emitted = len(visible)
        return new_text

    # ⚠️ 주의: 매 feed마다 누적 버퍼 전체를 처음부터 재스캔한다 — O(n²)지만
    # 코스 bot_message는 수백 자 안팎이라 무해하다. 메시지가 훨씬 길어지면
    # 증분 상태 머신으로 바꿀 것.
    def _extract_visible(self) -> str | None:
        key = '"bot_message"'
        key_idx = self._buf.find(key)
        if key_idx == -1:
            return None
        colon_idx = self._buf.find(":", key_idx + len(key))
        if colon_idx == -1:
            return None

        n = len(self._buf)
        i = colon_idx + 1
        while i < n and self._buf[i] in " \t\r\n":
            i += 1
        if i >= n or self._buf[i] != '"':
            return None
        i += 1  # 여는 따옴표를 지나감

        literal_chars: list[str] = []
        escape = False
        closed = False
        while i < n:
            ch = self._buf[i]
            if escape:
                literal_chars.append(ch)
                escape = False
            elif ch == "\\":
                literal_chars.append(ch)
                escape = True
            elif ch == '"':
                closed = True
                i += 1
                break
            else:
                literal_chars.append(ch)
            i += 1

        literal = "".join(literal_chars)
        if not closed:
            literal = self._trim_incomplete_escape(literal, escape)

        try:
            return json.loads('"' + literal + '"')
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _trim_incomplete_escape(literal: str, trailing_backslash_pending: bool) -> str:
        """
        문자열이 아직 안 닫혔을 때(스트리밍 중) 끝에 남은 미완성 이스케이프를 잘라낸다.
        안 잘라내면 json.loads가 다음 청크를 기다리지 못하고 바로 예외를 던진다.
        """
        if trailing_backslash_pending:
            literal = literal[:-1]  # 다음 글자를 기다리는 중인 백슬래시 하나 — 제거
        idx = literal.rfind("\\u")
        if idx != -1 and len(literal) - (idx + 2) < 4:
            literal = literal[:idx]  # \uXXXX가 4자리를 채우기 전 — 그 \u부터 제거
        return literal
