"""
골든셋 로더 — Sprint 5 Task 1.

[설계 근거] docs/superpowers/specs/2026-08-03-sprint5-eval-pipeline-design.md §2
모드별(근교/장거리/타지) 고정 입력 케이스 30개를 eval/cases.jsonl에 동결해두고,
이 모듈은 그걸 읽어 GoldenCase로 복원하는 역할만 한다. LLM·네트워크 의존 0 —
API 키 없이도 항상 성공해야 한다(pytest에서 회귀 없이 돌아가야 하므로).

`ctx` 딕셔너리 키는 ContextData 필드명과 정확히 1:1이어야 한다.
`ContextData(**row["ctx"])`로 바로 복원하기 때문에, 계약이 어긋나면(오타 등)
여기서 즉시 TypeError로 실패해야 드리프트를 조기에 잡는다.
"""

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from traveler.rag.types import ContextData, SpotCandidate


@dataclass(frozen=True)
class GoldenCase:
    """골든셋 한 줄을 나타내는 불변 케이스.

    # WHY frozen=True인가:
    #   골든셋은 "동결"된 회귀 테스트 입력이다. 로딩 후 어딘가에서 실수로
    #   값을 바꾸면 같은 case_id인데 다른 입력으로 채점되는 조용한 버그가
    #   생긴다. frozen=True로 두면 그런 변경 시도 자체가 예외로 막힌다.
    """
    case_id: str
    mode: str
    today: date
    ctx: ContextData
    expectations: dict
    reference_example: str | None
    tags: list[str]


def load_cases(path: Path, *, mode: str | None = None) -> list[GoldenCase]:
    """JSONL 골든셋을 읽어 GoldenCase 리스트로 복원한다.

    Args:
        path: `eval/cases.jsonl` 같은 JSONL 파일 경로. 한 줄 = 한 케이스.
        mode: 지정하면 해당 모드(예: "근교")의 케이스만 필터링해 반환한다.
              None이면 전체 케이스를 반환한다.
    """
    cases: list[GoldenCase] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            # ctx 딕셔너리 안의 candidate_spots는 SpotCandidate 딕셔너리 목록이라
            # ContextData(**ctx_dict)에 그대로 넘기기 전에 dataclass 인스턴스로
            # 한 겹 먼저 복원해야 한다.
            ctx_dict = dict(row["ctx"])
            ctx_dict["candidate_spots"] = [
                SpotCandidate(**spot) for spot in ctx_dict["candidate_spots"]
            ]
            ctx = ContextData(**ctx_dict)

            cases.append(
                GoldenCase(
                    case_id=row["case_id"],
                    mode=row["mode"],
                    today=date.fromisoformat(row["today"]),
                    ctx=ctx,
                    expectations=row["expectations"],
                    reference_example=row.get("reference_example"),
                    tags=row["tags"],
                )
            )

    if mode is not None:
        cases = [c for c in cases if c.mode == mode]
    return cases
