"""
골든셋 회귀 테스트 CLI — Sprint 5 Task 4.

[설계 근거] docs/superpowers/specs/2026-08-03-sprint5-eval-pipeline-design.md §5
`python eval/run_judge.py --dataset eval/cases.jsonl --out eval/report.json` 한 줄로
생성(course_generator) → 구조 측정(metrics) → 채점(judge) → 집계 → baseline 비교까지
수행한다.

[회귀 판정 기준(§5.2 결정 A) — 혼합]
  - 구조 메트릭: 절대 기준. 케이스 중 하나라도 structural.passed == False면
    baseline 유무와 무관하게 그 자체로 verdict="FAIL" (JSON이 깨지는 데
    "이전보다 덜 깨짐"이라는 개념은 없다).
  - judge 점수: baseline 대비 델타. 모드별 axis 평균이 baseline 대비 -0.3 이상
    하락하면 FAIL.

[verdict 3단계 — 2026-08-10, PM 리뷰 반영]
  PASS/FAIL 2단이던 구 버전은 "판정 불가"(baseline 없음/손상, judge 표본 0건)를
  조용히 PASS로 묻어 CI에서 "검사를 못 했음"과 "통과함"을 구분할 수 없었다.
  이제 셋으로 나눈다 — verdict/exit code:
    PASS(0)         구조·judge 델타 모두 문제 없음
    FAIL(1)         구조 FAIL이 하나라도 있거나 judge 델타가 -0.3 이상 하락
    INCONCLUSIVE(2) 판정 자체를 못 함 — baseline 없음/손상, 또는 어떤 모드의
                    judge 표본이 0건(--no-judge 명시 시는 제외)
  FAIL 사유는 report.json의 regression.failures, INCONCLUSIVE 사유는
  regression.notes에 각각 담긴다. --save-baseline은 FAIL일 때만 저장을 거부한다
  — INCONCLUSIVE(특히 baseline 최초 생성)는 --save-baseline의 정상 사용 시나리오다.

이 스크립트는 pytest 스위트에 포함되지 않는다 — 실 API를 호출하는 수동 배치이지
CI 게이트가 아니다(§5.2). API 호출 없는 로직(집계·회귀 판정·exit code)은
tests/test_eval_run_judge.py에서 generate_course/judge_course를 목업해 검증한다.
"""

import argparse
import asyncio
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# WHY sys.path 삽입: 이 스크립트를 어느 위치에서 실행하든(예: python eval/run_judge.py)
#   리포 루트가 import 경로에 들어가야 `from eval...`/`from traveler...`가 풀린다.
#   parents[1] = eval/의 부모 = 리포 루트. scripts/smoke_test.py와 동일한 패턴.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.dataset import GoldenCase, load_cases  # noqa: E402  (sys.path 설정 후 import)
from eval.judge import judge_course  # noqa: E402
from eval.judge_prompt import JUDGE_PROMPT_VERSION  # noqa: E402
from eval.metrics import measure  # noqa: E402
from traveler.core.course_generator import generate_course  # noqa: E402
from traveler.core.llm_client import JUDGE_MODEL, MODEL  # noqa: E402
from traveler.core.prompt import PROMPT_VERSION  # noqa: E402

logger = logging.getLogger(__name__)

# WHY 0으로 고정: eval 실행은 재현성이 목적이다(D3). generate_course()의
# temperature 기본값(0.3, 프로덕션용)과 달리 회귀 러너는 항상 0을 명시적으로
# 전달해 "프롬프트가 바뀌어서 결과가 바뀐 것"과 "그냥 샘플링이 달라서 바뀐 것"을
# 구분할 수 있게 한다.
EVAL_TEMPERATURE = 0

# 회귀 판정 임계값(§5.2): 모드별 axis 평균이 baseline 대비 이 값 이상 하락하면 FAIL.
REGRESSION_THRESHOLD = -0.3

AXES = ("condition_fit", "mood_match", "route_sanity", "diversity", "completeness")

# WHY 동시성 제한(PM 리뷰 반영): 케이스 1개가 생성(gen)+채점(judge) 최대 2회의 API
# 호출을 순차로 만든다. asyncio.gather로 30케이스를 무제한 동시에 던지면 그 시점에
# 최대 60개 호출이 동시에 나갈 수 있어 429(rate limit)를 유발한다. 429는
# _process_case의 except Exception에 걸려 "구조 FAIL"로 기록되므로, 세마포어 없이는
# rate limit이 곧 거짓 회귀로 이어진다.
CONCURRENCY_LIMIT = 5


def _failed_structural_dict() -> dict:
    """
    케이스 처리 자체가 예외로 죽었을 때(예: generate_course 호출 실패) 채워 넣는
    구조 메트릭 자리표시자. eval/metrics.py의 StructuralMetrics와 동일한 필드
    구성으로, 전부 "실패"를 뜻하는 값이다 — passed=False라 회귀 판정에서
    자연스럽게 구조 FAIL로 집계된다(Graceful Degradation: 리포트 스키마는
    깨지지 않고, 대신 이 케이스가 실패로 정직하게 기록된다).
    """
    return {
        "json_valid": False,
        "is_fallback": False,
        "place_count": 0,
        "place_count_ok": False,
        "grounded": False,
        "hallucinated_ids": [],
        "estimated_total_min": 0,
        "time_budget_ok": False,
        "passed": False,
    }


async def _process_case(case: GoldenCase, no_judge: bool, semaphore: asyncio.Semaphore) -> dict:
    """
    케이스 1개를 생성→구조측정→(게이트 통과 시) 채점까지 처리한다.

    # WHY 전체를 하나의 try/except로 감싸는가?
    #   judge_course()는 이미 내부에서 예외를 흡수해 None을 반환하지만,
    #   generate_course()나 measure()가 예상 밖 예외(네트워크 오류 등)를 던지면
    #   이 케이스 하나 때문에 asyncio.gather 전체가 죽는다. 여기서 흡수해야
    #   나머지 29개 케이스가 계속 처리된다(Graceful Degradation).
    # WHY semaphore: CONCURRENCY_LIMIT 참고 — 동시 API 호출 수를 제한해 rate limit이
    #   거짓 구조 FAIL로 오판되는 것을 막는다.
    """
    async with semaphore:
        try:
            course = await generate_course(
                case.ctx, travel_mode=case.mode, temperature=EVAL_TEMPERATURE
            )
            structural = measure(case.ctx, case.expectations, course.model_dump_json())

            judge_result = None
            skipped_reason = None
            if not structural.passed:
                skipped_reason = "structural_fail"
            elif no_judge:
                skipped_reason = "no_judge_flag"
            else:
                judge_result = await judge_course(case.ctx, case.mode, course)
                if judge_result is None:
                    skipped_reason = "judge_failed"

            return {
                "case_id": case.case_id,
                "mode": case.mode,
                "structural": structural.model_dump(),
                "judge": judge_result.model_dump() if judge_result is not None else None,
                "skipped_reason": skipped_reason,
            }
        except Exception as e:
            logger.warning("케이스 %s 처리 중 예외 발생, 실패로 기록: %s", case.case_id, e)
            return {
                "case_id": case.case_id,
                "mode": case.mode,
                "structural": _failed_structural_dict(),
                "judge": None,
                "skipped_reason": "error",
                "error": str(e),
            }


def _axis_means(judge_dicts: list[dict]) -> dict[str, float | None]:
    """judge 결과 목록에서 5축 각각의 평균 점수를 계산한다. 데이터가 없는 축은 None."""
    means: dict[str, float | None] = {}
    for axis in AXES:
        scores = [j[axis]["score"] for j in judge_dicts]
        means[axis] = round(sum(scores) / len(scores), 3) if scores else None
    return means


def _aggregate_group(results: list[dict]) -> dict:
    """결과 목록 하나(overall/모드/태그 단위)를 구조 pass rate + judge axis 평균으로 집계."""
    total = len(results)
    passed = sum(1 for r in results if r["structural"]["passed"])
    judge_dicts = [r["judge"] for r in results if r["judge"] is not None]
    return {
        "case_count": total,
        "structural_pass_rate": round(passed / total, 3) if total else 0.0,
        "judge_case_count": len(judge_dicts),
        "axis_means": _axis_means(judge_dicts),
    }


def _aggregate_by_mode(results: list[dict]) -> dict:
    modes = sorted({r["mode"] for r in results})
    return {mode: _aggregate_group([r for r in results if r["mode"] == mode]) for mode in modes}


def _aggregate_by_tag(results: list[dict], cases: list[GoldenCase]) -> dict:
    """
    §5.3 — by_tag 집계. GoldenCase.tags의 각 태그별로 결과를 묶어 집계한다.
    이게 Task 1 factor grid를 실제 진단력으로 바꾸는 지점(예: "비 오는 날만 하락").
    """
    tags_by_case_id = {c.case_id: c.tags for c in cases}
    all_tags = sorted({tag for tags in tags_by_case_id.values() for tag in tags})
    grouped = {}
    for tag in all_tags:
        matching = [r for r in results if tag in tags_by_case_id.get(r["case_id"], [])]
        if matching:
            grouped[tag] = _aggregate_group(matching)
    return grouped


def _check_regression(by_mode: dict, baseline_path: Path) -> tuple[dict, list[str], list[str]]:
    """
    baseline.json과 이번 실행의 모드별 axis 평균을 비교한다.

    Returns:
        (deltas, failures, notes)
        failures: baseline 대비 실제로 -0.3 이상 하락한 항목 — verdict를 FAIL로 만든다.
        notes: 회귀 판정 자체를 할 수 없었던 사유(baseline 없음/손상/평가 버전 불일치) — FAIL은
        아니지만 PASS로 조용히 넘기지도 않는다(호출부가 INCONCLUSIVE로 취급, PM 리뷰 반영).
    """
    if not baseline_path.exists():
        return {}, [], ["baseline 파일 없음(최초 실행) — judge 델타 비교 생략"]

    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("baseline 파일 로드 실패, 델타 비교 생략: %s", e)
        return {}, [], [f"baseline 파일 손상({e}) — judge 델타 비교 생략"]

    # 채점하지 않은 실행은 버전 비교도 필요 없다. 표본 누락은 호출부에서 판정한다.
    if not any(
        mean is not None for agg in by_mode.values() for mean in agg["axis_means"].values()
    ):
        return {}, [], []

    metadata = baseline.get("_meta") if isinstance(baseline, dict) else None
    baseline_version = metadata.get("judge_prompt_version") if isinstance(metadata, dict) else None
    if baseline_version != JUDGE_PROMPT_VERSION:
        return {}, [], [
            f"baseline judge_prompt_version 불일치 또는 누락 "
            f"({baseline_version!r} → {JUDGE_PROMPT_VERSION}) — "
            "같은 judge 버전으로 기준 코스를 다시 평가해야 하므로 델타 비교 생략"
        ]

    deltas: dict = {}
    failures: list[str] = []
    notes: list[str] = []
    for mode, agg in by_mode.items():
        base_axis = baseline.get(mode, {})
        mode_deltas = {}
        for axis, current_mean in agg["axis_means"].items():
            if current_mean is None:
                continue
            base_mean = base_axis.get(axis) if isinstance(base_axis, dict) else None
            if base_mean is None:
                notes.append(f"모드 {mode} 축 {axis}: baseline 점수 없음 — 회귀 판정 불가")
                continue
            delta = round(current_mean - base_mean, 3)
            mode_deltas[axis] = delta
            if delta <= REGRESSION_THRESHOLD:
                failures.append(
                    f"모드 {mode} 축 {axis} 하락: baseline {base_mean} → 현재 {current_mean} (delta {delta})"
                )
        if mode_deltas:
            deltas[mode] = mode_deltas
    return deltas, failures, notes


def _save_baseline(by_mode: dict, baseline_path: Path) -> None:
    """모드별 judge axis 평균과 평가 기준 버전을 baseline 파일로 저장한다."""
    baseline = {mode: agg["axis_means"] for mode, agg in by_mode.items()}
    baseline["_meta"] = {"judge_prompt_version": JUDGE_PROMPT_VERSION}
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(
        json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _git_sha() -> str:
    repo_root = Path(__file__).resolve().parent.parent
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_root,
            timeout=5,
        )
        sha = result.stdout.strip()
        return sha if sha else "unknown"
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.warning("git_sha 조회 실패: %s", e)
        return "unknown"


async def run(args: argparse.Namespace) -> dict:
    """
    회귀 러너 본체. report.json 스키마(§5.3)에 맞는 dict를 반환한다.

    # WHY 파일 쓰기를 여기서 하지 않는가?
    #   run()이 순수하게 dict를 반환해야 테스트가 asyncio 이벤트 루프 안에서
    #   generate_course/judge_course를 목업한 채로 결과를 직접 검증할 수 있다.
    #   파일 I/O와 exit code 결정은 main()이 담당한다.
    """
    cases = load_cases(args.dataset, mode=args.mode)
    if args.limit is not None:
        cases = cases[: args.limit]

    # WHY asyncio.gather: 30개 케이스를 병렬로 처리해 순차 호출보다 빠르게 끝낸다.
    # return_exceptions는 필요 없다 — _process_case가 자체 try/except로 예외를
    # 이미 흡수해 항상 dict를 반환하기 때문이다. CONCURRENCY_LIMIT 세마포어가
    # 실제 동시 API 호출 수를 제한한다.
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    results = list(
        await asyncio.gather(
            *(_process_case(c, args.no_judge, semaphore) for c in cases)
        )
    )

    structural_failures = [r for r in results if not r["structural"]["passed"]]
    overall = _aggregate_group(results)
    by_mode = _aggregate_by_mode(results)
    by_tag = _aggregate_by_tag(results, cases)

    deltas, judge_failures, notes = _check_regression(by_mode, args.baseline)

    # WHY 모드별 judge 표본 0건 감지(PM 리뷰 반영): axis_means가 전부 None이면
    # _check_regression의 델타 비교가 전부 continue로 넘어가 아무 failure도 안 남고
    # 조용히 PASS가 된다. --no-judge를 명시한 게 아닌데 표본이 0건이면 "회귀 없음"이
    # 아니라 "판정할 데이터가 없음"이므로 notes에 남겨 INCONCLUSIVE로 취급한다.
    if not args.no_judge:
        for mode, agg in by_mode.items():
            if agg["case_count"] > 0 and agg["judge_case_count"] == 0:
                notes.append(f"모드 {mode}: judge 표본 0건 — 회귀 판정 불가")

    failures: list[str] = [
        f"case {r['case_id']} ({r['mode']}): structural FAIL"
        for r in structural_failures
    ]
    failures.extend(judge_failures)

    # WHY 3단 verdict(PM 리뷰 반영): 기존엔 PASS/FAIL 2단이라 "검사를 못 했음"과
    # "검사를 통과함"이 exit code 0으로 동일했다. baseline 없음/손상, judge 표본
    # 부족처럼 판정 자체가 불가능한 경우는 INCONCLUSIVE로 분리해 CI가 구분할 수 있게 한다.
    if structural_failures or judge_failures:
        verdict = "FAIL"
    elif notes:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "PASS"

    return {
        "run": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_sha": _git_sha(),
            "prompt_version": PROMPT_VERSION,
            "judge_prompt_version": JUDGE_PROMPT_VERSION,
            "judge_model": JUDGE_MODEL,
            "gen_model": MODEL,
            "temperature": EVAL_TEMPERATURE,
            "case_count": len(cases),
        },
        "cases": results,
        "aggregate": {
            "overall": overall,
            "by_mode": by_mode,
            "by_tag": by_tag,
        },
        "regression": {
            "baseline_ref": str(args.baseline),
            "deltas": deltas,
            "verdict": verdict,
            "failures": failures,
            "notes": notes,
        },
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="골든셋 회귀 테스트 CLI (Sprint 5 Task 4)")
    parser.add_argument("--dataset", type=Path, required=True, help="eval/cases.jsonl 경로")
    parser.add_argument("--out", type=Path, required=True, help="report.json 출력 경로")
    parser.add_argument("--mode", type=str, default=None, help="근교|타지|장거리 중 하나만 필터")
    parser.add_argument(
        "--baseline", type=Path, default=Path("eval/baseline.json"), help="baseline.json 경로"
    )
    parser.add_argument(
        "--save-baseline", action="store_true", help="이번 실행 결과를 baseline으로 저장"
    )
    parser.add_argument("--limit", type=int, default=None, help="앞에서 N개만 처리(스모크 테스트용)")
    parser.add_argument(
        "--no-judge", action="store_true", help="judge를 호출하지 않고 구조 메트릭만 측정"
    )
    return parser.parse_args(argv)


# WHY 3단 exit code(PM 리뷰 반영): PASS=0(정상), FAIL=1(회귀 확인), INCONCLUSIVE=2
# (판정 불가 — baseline 없음/손상, judge 표본 부족). CI에서 FAIL만 막고
# INCONCLUSIVE는 경고로 다루는 등 분리된 처리를 하려면 exit code가 갈라져야 한다.
EXIT_CODE_BY_VERDICT = {"PASS": 0, "FAIL": 1, "INCONCLUSIVE": 2}


def main(argv: list[str] | None = None) -> int:
    """
    CLI 진입점. report.json을 쓰고, verdict에 따른 exit code
    (0=PASS, 1=FAIL, 2=INCONCLUSIVE)를 반환한다.
    """
    args = _parse_args(argv)
    report = asyncio.run(run(args))

    # WHY parent.mkdir(PM 리뷰 반영): --out에 존재하지 않는 디렉토리(예:
    # eval/reports/report.json)를 넘기면 API 호출을 전부 마친 뒤 마지막에
    # write_text가 FileNotFoundError로 죽는다. _save_baseline()은 이미 mkdir을
    # 하고 있었는데 여기만 빠져 있었다.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    verdict = report["regression"]["verdict"]

    # WHY FAIL일 때 저장 거부(PM 리뷰 반영): --save-baseline은 "이번 실행을 새
    # 기준으로 삼겠다"는 뜻인데, 회귀가 난 실행을 그대로 저장하면 다음 실행부터
    # 진짜 회귀가 "이미 baseline"이라 통과해버린다. INCONCLUSIVE(baseline 최초
    # 생성 등)는 저장을 막지 않는다 — 오히려 그게 --save-baseline의 정상적인
    # 첫 사용 시나리오다.
    if args.save_baseline:
        if verdict == "FAIL":
            logger.warning(
                "verdict=FAIL 상태라 baseline을 저장하지 않습니다. "
                "회귀를 해결한 뒤 --save-baseline으로 다시 실행하세요."
            )
        else:
            _save_baseline(report["aggregate"]["by_mode"], args.baseline)

    print(f"[run_judge] verdict={verdict} case_count={report['run']['case_count']} → {args.out}")
    if verdict == "FAIL":
        for failure in report["regression"]["failures"]:
            print(f"  - {failure}")
    elif verdict == "INCONCLUSIVE":
        for note in report["regression"]["notes"]:
            print(f"  - {note}")

    return EXIT_CODE_BY_VERDICT[verdict]


if __name__ == "__main__":
    sys.exit(main())
