"""
eval/run_judge.py 단위 테스트 — Sprint 5 Task 4.

[설계 근거] docs/superpowers/specs/2026-08-03-sprint5-eval-pipeline-design.md §5.5
실 API 호출 없이 집계·회귀판정·exit code 로직만 검증한다. generate_course와
judge_course(둘 다 eval.run_judge가 import한 심볼)를 목업하고, load_cases/measure는
실제 함수 그대로 사용한다 — 이 로더·메트릭 자체는 이미 test_eval_dataset.py /
test_eval_metrics.py가 검증하므로, 여기서는 run_judge.py가 그 결과를 올바르게
조립·판정하는지만 본다.
"""

import json
from pathlib import Path
from unittest.mock import patch

from eval.judge import AxisScore, JudgeResult
from eval.run_judge import main
from traveler.validation.schema import RecommendedPlaceModel, StreamingCourseModel


# ── 공통 픽스처 ────────────────────────────────────────────────────────────────

def _case_dict(case_id: str, mode: str = "타지", tags: list[str] | None = None) -> dict:
    """
    eval/cases.jsonl 한 줄과 동일한 스키마의 골든 케이스 딕셔너리.

    WHY user_query=case_id인가: generate_course 목업이 ctx만 받고 case_id는
    받지 못하므로(실제 시그니처와 동일하게 유지), ctx.user_query를 case_id로
    심어 목업 side_effect가 "어느 케이스인지"를 구분할 수 있게 한다.
    """
    return {
        "case_id": case_id,
        "mode": mode,
        "today": "2026-08-04",
        "ctx": {
            "user_query": case_id,
            "current_lat": 0.0,
            "current_lng": 0.0,
            "weather": "맑음",
            "duration_hours": 3,
            "transport": "자차",
            "companion": "",
            "mood": "",
            "candidate_spots": [
                {
                    "place_id": "1",
                    "place_name": "장소1",
                    "category": "관광",
                    "distance_min": 10.0,
                    "indoor": True,
                }
            ],
        },
        "expectations": {"expect_places_range": [1, 3]},
        "reference_example": None,
        "tags": tags or [],
    }


def _write_dataset(path: Path, cases: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")


def _write_baseline(path: Path, mode: str, score: float) -> None:
    """
    baseline.json 픽스처. 5축 전부를 동일 점수로 채워, 이번 실행 judge 점수가
    같은 score면 delta=0이 되어 회귀로 잡히지 않게 한다.
    """
    axes = ("condition_fit", "mood_match", "route_sanity", "diversity", "completeness")
    path.write_text(
        json.dumps({mode: {axis: score for axis in axes}}), encoding="utf-8"
    )


def _passing_course() -> StreamingCourseModel:
    """구조 메트릭을 전부 통과하는 코스 — place_id="1"은 후보 목록 안에 있음."""
    return StreamingCourseModel(
        bot_message="좋은 코스를 추천해요",
        recommended_places=[RecommendedPlaceModel(place_id="1", visit_sequence=1, duration_min=60)],
    )


def _failing_course() -> StreamingCourseModel:
    """구조 메트릭 FAIL — place_id="999"는 후보 목록(place_id="1") 밖이라 grounded=False."""
    return StreamingCourseModel(
        bot_message="아무 코스나 추천해요",
        recommended_places=[RecommendedPlaceModel(place_id="999", visit_sequence=1, duration_min=60)],
    )


def _judge_result(score: int = 5) -> JudgeResult:
    axis = AxisScore(reason="근거", score=score)
    return JudgeResult(
        condition_fit=axis,
        mood_match=axis,
        route_sanity=axis,
        diversity=axis,
        completeness=axis,
    )


# ── Test (a): 전부 구조 PASS → verdict PASS, exit code 0 ─────────────────────

@patch("eval.run_judge.judge_course")
@patch("eval.run_judge.generate_course")
def test_all_structural_pass_verdict_pass(mock_generate, mock_judge, tmp_path):
    dataset = tmp_path / "cases.jsonl"
    _write_dataset(dataset, [_case_dict("case-1"), _case_dict("case-2")])
    out = tmp_path / "report.json"
    baseline = tmp_path / "baseline.json"
    _write_baseline(baseline, mode="타지", score=5.0)  # delta=0 → 회귀 아님

    mock_generate.return_value = _passing_course()
    mock_judge.return_value = _judge_result(5)

    exit_code = main(
        ["--dataset", str(dataset), "--out", str(out), "--baseline", str(baseline)]
    )

    assert exit_code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["regression"]["verdict"] == "PASS"
    assert len(report["cases"]) == 2
    assert all(c["structural"]["passed"] for c in report["cases"])


# ── Test (b): 구조 FAIL 케이스 포함 → verdict FAIL, exit code 1 ──────────────

@patch("eval.run_judge.judge_course")
@patch("eval.run_judge.generate_course")
def test_one_structural_fail_verdict_fail(mock_generate, mock_judge, tmp_path):
    dataset = tmp_path / "cases.jsonl"
    _write_dataset(dataset, [_case_dict("case-ok"), _case_dict("case-bad")])
    out = tmp_path / "report.json"

    async def fake_generate(ctx, travel_mode=None, temperature=None):
        return _passing_course() if ctx.user_query == "case-ok" else _failing_course()

    mock_generate.side_effect = fake_generate
    mock_judge.return_value = _judge_result(5)

    exit_code = main(["--dataset", str(dataset), "--out", str(out)])

    assert exit_code == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["regression"]["verdict"] == "FAIL"
    by_case = {c["case_id"]: c for c in report["cases"]}
    assert by_case["case-ok"]["structural"]["passed"] is True
    assert by_case["case-bad"]["structural"]["passed"] is False
    assert by_case["case-bad"]["skipped_reason"] == "structural_fail"
    assert by_case["case-bad"]["judge"] is None
    assert any("structural FAIL" in f for f in report["regression"]["failures"])


# ── Test (c): baseline 대비 judge 평균 -0.3 이상 하락 → verdict FAIL ─────────

@patch("eval.run_judge.judge_course")
@patch("eval.run_judge.generate_course")
def test_baseline_regression_detected(mock_generate, mock_judge, tmp_path):
    dataset = tmp_path / "cases.jsonl"
    _write_dataset(dataset, [_case_dict("case-1", mode="타지")])
    out = tmp_path / "report.json"
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "타지": {
                    "condition_fit": 4.5,
                    "mood_match": 4.5,
                    "route_sanity": 4.5,
                    "diversity": 4.5,
                    "completeness": 4.5,
                }
            }
        ),
        encoding="utf-8",
    )

    mock_generate.return_value = _passing_course()
    # 이번 실행 점수가 baseline(4.5)보다 크게 낮음 → delta <= -0.3
    mock_judge.return_value = _judge_result(1)

    exit_code = main(
        ["--dataset", str(dataset), "--out", str(out), "--baseline", str(baseline)]
    )

    assert exit_code == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["regression"]["verdict"] == "FAIL"
    assert "타지" in report["regression"]["deltas"]
    assert report["regression"]["deltas"]["타지"]["condition_fit"] <= -0.3


# ── Test (d): baseline 파일이 없는 최초 실행 → 예외 없이 동작 + INCONCLUSIVE ──

@patch("eval.run_judge.judge_course")
@patch("eval.run_judge.generate_course")
def test_missing_baseline_first_run_is_inconclusive(mock_generate, mock_judge, tmp_path):
    """
    [PM 리뷰 반영 — 2026-08-10] baseline 없음은 더 이상 조용히 PASS로 묻히지 않는다.
    구조 메트릭은 전부 통과했지만 회귀 판정 자체를 할 수 없으므로 verdict는
    INCONCLUSIVE, exit code는 2다.
    """
    dataset = tmp_path / "cases.jsonl"
    _write_dataset(dataset, [_case_dict("case-1")])
    out = tmp_path / "report.json"
    baseline = tmp_path / "does_not_exist_baseline.json"  # 의도적으로 존재하지 않음

    mock_generate.return_value = _passing_course()
    mock_judge.return_value = _judge_result(5)

    exit_code = main(
        ["--dataset", str(dataset), "--out", str(out), "--baseline", str(baseline)]
    )

    assert exit_code == 2
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["regression"]["verdict"] == "INCONCLUSIVE"
    assert report["regression"]["failures"] == []
    assert any("baseline 파일 없음" in n for n in report["regression"]["notes"])
    assert report["regression"]["deltas"] == {}


# ── Test (e): 한 케이스의 generate_course 예외 → 전체는 죽지 않고 나머지 처리 ──

@patch("eval.run_judge.judge_course")
@patch("eval.run_judge.generate_course")
def test_generate_course_exception_does_not_crash_run(mock_generate, mock_judge, tmp_path):
    dataset = tmp_path / "cases.jsonl"
    _write_dataset(dataset, [_case_dict("case-good"), _case_dict("case-explodes")])
    out = tmp_path / "report.json"

    async def fake_generate(ctx, travel_mode=None, temperature=None):
        if ctx.user_query == "case-explodes":
            raise RuntimeError("simulated network failure")
        return _passing_course()

    mock_generate.side_effect = fake_generate
    mock_judge.return_value = _judge_result(5)

    # 예외를 전파하지 않고 정상적으로 report.json을 만들어야 한다.
    exit_code = main(["--dataset", str(dataset), "--out", str(out)])

    report = json.loads(out.read_text(encoding="utf-8"))
    assert len(report["cases"]) == 2
    by_case = {c["case_id"]: c for c in report["cases"]}

    assert by_case["case-good"]["structural"]["passed"] is True
    assert by_case["case-explodes"]["structural"]["passed"] is False
    assert by_case["case-explodes"]["skipped_reason"] == "error"

    # 구조 FAIL이 하나라도 있으므로 verdict는 FAIL — 하지만 죽지 않고 끝까지 실행됨이 핵심.
    assert exit_code == 1
    assert report["regression"]["verdict"] == "FAIL"


# ── Test (f): --no-judge → judge_course를 아예 호출하지 않음 ─────────────────

@patch("eval.run_judge.judge_course")
@patch("eval.run_judge.generate_course")
def test_no_judge_flag_skips_judge_call(mock_generate, mock_judge, tmp_path):
    dataset = tmp_path / "cases.jsonl"
    _write_dataset(dataset, [_case_dict("case-1")])
    out = tmp_path / "report.json"
    # baseline을 빈 dict로 미리 만들어 "baseline 없음" 노트가 섞이지 않게 한다 —
    # 이 테스트가 보려는 건 --no-judge가 judge 호출 자체를 건너뛰는지다.
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{}", encoding="utf-8")

    mock_generate.return_value = _passing_course()

    exit_code = main(
        ["--dataset", str(dataset), "--out", str(out), "--no-judge", "--baseline", str(baseline)]
    )

    assert exit_code == 0
    mock_judge.assert_not_called()
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["cases"][0]["skipped_reason"] == "no_judge_flag"
    assert report["cases"][0]["judge"] is None


# ── Test (g): --save-baseline → baseline 파일에 모드별 axis 평균 저장 ─────────

@patch("eval.run_judge.judge_course")
@patch("eval.run_judge.generate_course")
def test_save_baseline_writes_axis_means(mock_generate, mock_judge, tmp_path):
    """
    baseline.json이 아직 없는 최초 실행이라 verdict는 INCONCLUSIVE(exit code 2)지만,
    FAIL이 아니므로 --save-baseline은 정상적으로 저장된다 — 이게 baseline을 처음
    만드는 표준 시나리오다(PM 리뷰 반영: FAIL일 때만 저장을 막는다).
    """
    dataset = tmp_path / "cases.jsonl"
    _write_dataset(dataset, [_case_dict("case-1", mode="근교")])
    out = tmp_path / "report.json"
    baseline = tmp_path / "baseline.json"

    mock_generate.return_value = _passing_course()
    mock_judge.return_value = _judge_result(4)

    exit_code = main(
        [
            "--dataset", str(dataset),
            "--out", str(out),
            "--baseline", str(baseline),
            "--save-baseline",
        ]
    )

    assert exit_code == 2
    assert baseline.exists()
    saved = json.loads(baseline.read_text(encoding="utf-8"))
    assert saved["근교"]["condition_fit"] == 4.0


# ── Test (h): verdict FAIL이면 --save-baseline이 저장을 거부한다 ──────────────

@patch("eval.run_judge.judge_course")
@patch("eval.run_judge.generate_course")
def test_save_baseline_blocked_when_verdict_fail(mock_generate, mock_judge, tmp_path):
    dataset = tmp_path / "cases.jsonl"
    _write_dataset(dataset, [_case_dict("case-bad")])
    out = tmp_path / "report.json"
    baseline = tmp_path / "baseline.json"  # 존재하지 않음 → save 시 새로 만들어질 수 있는 상태

    mock_generate.return_value = _failing_course()  # 구조 FAIL → verdict FAIL
    mock_judge.return_value = _judge_result(5)

    exit_code = main(
        [
            "--dataset", str(dataset),
            "--out", str(out),
            "--baseline", str(baseline),
            "--save-baseline",
        ]
    )

    assert exit_code == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["regression"]["verdict"] == "FAIL"
    assert not baseline.exists()  # FAIL 상태로는 baseline이 저장되지 않아야 함


# ── Test (i): --out 부모 디렉토리가 없어도 자동 생성된다 ──────────────────────

@patch("eval.run_judge.judge_course")
@patch("eval.run_judge.generate_course")
def test_out_parent_directory_is_auto_created(mock_generate, mock_judge, tmp_path):
    dataset = tmp_path / "cases.jsonl"
    _write_dataset(dataset, [_case_dict("case-1")])
    out = tmp_path / "reports" / "nested" / "report.json"  # 존재하지 않는 하위 디렉토리
    baseline = tmp_path / "baseline.json"
    _write_baseline(baseline, mode="타지", score=5.0)

    mock_generate.return_value = _passing_course()
    mock_judge.return_value = _judge_result(5)

    exit_code = main(
        ["--dataset", str(dataset), "--out", str(out), "--baseline", str(baseline)]
    )

    assert exit_code == 0
    assert out.exists()


# ── Test (j): 한 모드의 judge 표본이 0건이면 INCONCLUSIVE ────────────────────

@patch("eval.run_judge.judge_course")
@patch("eval.run_judge.generate_course")
def test_zero_judge_samples_in_mode_is_inconclusive(mock_generate, mock_judge, tmp_path):
    """
    judge_course가 계속 None을 반환하면(파싱 실패 등) 구조는 통과해도 그 모드의
    judge 표본이 0건이라 델타 비교를 할 수 없다. 이걸 조용히 PASS로 묻지 않고
    INCONCLUSIVE로 표시해야 한다(PM 리뷰 반영).
    """
    dataset = tmp_path / "cases.jsonl"
    _write_dataset(dataset, [_case_dict("case-1", mode="타지")])
    out = tmp_path / "report.json"
    baseline = tmp_path / "baseline.json"
    _write_baseline(baseline, mode="타지", score=5.0)

    mock_generate.return_value = _passing_course()
    mock_judge.return_value = None  # judge 응답 파싱 실패를 흉내냄

    exit_code = main(
        ["--dataset", str(dataset), "--out", str(out), "--baseline", str(baseline)]
    )

    assert exit_code == 2
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["regression"]["verdict"] == "INCONCLUSIVE"
    assert any("judge 표본 0건" in n for n in report["regression"]["notes"])
