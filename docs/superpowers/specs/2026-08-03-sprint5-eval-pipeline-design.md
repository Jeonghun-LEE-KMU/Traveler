# Sprint 5 — 프롬프트 자동 평가 파이프라인 + 계절 L3 통합 설계 spec

- 작성일: 2026-08-03
- 프로젝트: 트래블러 (mobility_rag) — 한이음 드림업 공모전 LLM 파이프라인
- 담당: 이정훈 (LLM 엔지니어)
- 대상 브랜치: `feat/jeonghun/sprint5-eval-pipeline` (base: `feat/jeonghun/sprint4-llm-core`, 미병합 상태 위에서 이어감 — 결정 6)
- Sprint 5 기간: 2026-07-25 ~ 2026-08-07 (로드맵 HTML 기준. `info/LLM_sprint.md`는 8/13로 다르게 적혀 있었으나 사용자가 8/7로 확인함)
- 단계: 설계(브레인스토밍) — **이 spec은 구현 착수 시점의 기준 문서다. 코드는 이 문서에 정의된 인터페이스를 따른다.**

---

## 0. 배경 — 코드베이스 실사에서 확정된 전제

설계 전 실사로 다음이 확인됐다. 이후 모든 결정은 이 전제 위에서 내려졌다.

| # | 사실 | 근거 |
|---|---|---|
| F1 | Sprint 4 결과물(`rrf.py`·`persona.py`·prompt v2.0)은 `feat/jeonghun/sprint4-llm-core`에만 있고 develop 미병합(PR #14 OPEN) | `git log`, `gh pr view` |
| F2 | `traveler/core/llm_client.py`에 `JUDGE_MODEL = "gpt-4o"`가 "Phase 8 골든셋 평가용" 주석과 함께 이미 선언돼 있음 | 코드 직접 확인 |
| F3 | `build_context_message()`가 이미 L1(`ctx.weather`)·L2(`get_season(today)`)를 주입 중 — Task 5는 신규 구현이 아니라 L3 한 층 추가 | `traveler/core/prompt.py` |
| F4 | `SpotCandidate.place_id: int`(SQLite PK 가정) ↔ 실제 `HybridSearchResult`(`content_id: str`, `place_id`/`distance_min` 필드 자체가 없음) — 타입·필드 불일치 | `traveler/rag/types.py` vs `traveler/rag/hybrid_search.py:181-194` |
| F5 | `places.db` 로컬에 없음. `.env`(`code/.env`)에 `PINECONE_API_KEY` 없음(`OPENAI/WEATHER/TMAP/TOURAPI/NAVER`만 존재) | `find`, `.env` 키 이름 확인 |
| F6 | `generate_course()`는 `temperature=0.3` 하드코딩, 외부 주입 지점 없음 | `course_generator.py` |
| F7 | `_build_fallback()` 출력도 `StreamingCourseModel` 스키마 검증을 통과함 — "스키마 통과 = 성공"이 아님 | `course_generator.py` |
| F8 | README가 이미 `eval/run_judge.py --dataset eval/cases.jsonl --out eval/report.json`을 공표(단일 파일 가정) | `트래블러/README.md:113` |
| F9 | `hybrid_search.estimate_travel_minutes()`(haversine)가 존재하나 `rag/` 소속(송가영 관할) | `hybrid_search.py:638` |

**F4·F5는 "이정훈의 작업이 아니다."** `SpotCandidate` 생성은 `types.py` docstring에 "이수연 담당"으로 명시돼 있고, `places.db`·Pinecone 키도 이수연/송가영 소관이다. 이번 스프린트는 이 두 가지를 기다리거나 대신 만들지 않고, 아래처럼 **합성 fixture로 완전히 우회**한다.

---

## 1. 결정 로그 (사용자 확인 완료 — 이 값들로 확정하고 구현 진행)

| # | 질문 | 결정 | 근거 |
|---|---|---|---|
| D0 | Sprint 5 마감 | **8/7** | 사용자 확인 |
| D1 | 골든셋 규모 | **30개(모드별 10개)** | 60개는 Sprint 8에 100개로 확장 예정이 이미 `info/LLM_sprint.md`에 있어 지금 60을 채울 이유가 약함. 4일 제약에서 "완주"가 "물량"보다 우선 |
| D2 | Judge 모델 | **gpt-4o** (`JUDGE_MODEL` 그대로) | self-preference bias는 리포트에 명시하고 감수. Sprint 7에서 타 모델 교차검증 예정(별도 태스크로 관리하지 않고 이 문서에 기록만) |
| D3 | `generate_course()` temperature 노출 | **keyword-only 파라미터 추가, 기본값 0.3 유지** | 프로덕션 동작 무변경, eval만 0 전달로 재현성 확보 |
| D4 | L3 데이터 공급 경로 | **이번 스프린트는 순수 함수(조립 로직)만 완결.** 실데이터 연결(`ContextData.festivals` 필드 추가)은 다음 스프린트로 이관, 이수연에게 별도 요청만 걸어둠 | `season.py` 설계 철학(외부 의존 없는 건 core/, 있는 건 계약 경유)과 일관 + 4일 제약 |
| D5 | places.db/PINECONE_API_KEY 확보 | **불필요.** 골든셋은 합성 fixture로 진행, 실 DB 의존 없음 | F4, F5 |
| D6 | 브랜치 기준점 | **`feat/jeonghun/sprint4-llm-core` 위에 `feat/jeonghun/sprint5-eval-pipeline` 신설**(완료) | develop이 언제 갱신될지 불확실, Sprint4 산출물(RRF·persona·prompt v2) 위에서 평가해야 의미 있음 |
| D7 | 축제(L3)를 `recommended_places`에 실제로 넣을 것인가 | **아니오.** 컨텍스트 문자열로만 제공(bot_message 언급 참고용). `place_id` grounding 계약(F4) 위배 방지 | 기획팀이 "축제를 코스에 직접 추천"을 요구하면 별도 계약 변경 필요 — 이번 스프린트 범위 아님(가정, 틀리면 PM 확인 필요) |
| D8 | 골든셋에 "정답 예시" 포함 여부 | **선택 필드 `reference_example`을 넣되 채점에는 사용하지 않음** | 발표 자료용 최소 비용 절충안 |

> D7·D8은 사용자가 명시적으로 확답하지 않은 항목이라 **가정으로 진행**한다. 틀렸으면 구현 중 언제든 정정 가능하도록 D7은 컨텍스트 조립 함수 안에 격리해뒀다(아래 5-2 참조).

---

## 2. Task 1 — 골든셋 30개 (`eval/cases.jsonl`)

### 2.1 범위와 완료 정의
모드별 고정 입력 케이스 10개(근교/장거리/타지) = 30개를 `eval/cases.jsonl`에 동결. 완료 정의: `pytest`가 API 호출 없이 30개 전부 `ContextData` 복원에 성공하고, 각 축(날씨/동행자/가용시간)의 모든 수준이 모드별 최소 2회 이상 등장함을 테스트로 강제한다.

### 2.2 설계 결정

**골든셋 생성 방식**: 사람이 factor grid를 설계하고 LLM이 문장을 채운 뒤 사람이 검수하는 하이브리드. 무작위 30개보다 훨씬 적은 수로 "어떤 조건에서 프롬프트가 깨지는지" 진단 가능(stratified eval set).

**`candidate_spots`**: 합성 `SpotCandidate` fixture로 동결(D5). 검색 품질(송가영 담당)과 프롬프트 품질(이정훈 담당)을 한 실험에 섞지 않기 위한 component-level eval.

**`expectations`**: "정답 코스 전체"가 아니라 "검증 가능한 제약"만 기술. 생성형은 정답이 하나가 아니라 매 실행이 다르므로, 코스 전체를 정답으로 두면 회귀 테스트가 노이즈만 낸다.

### 2.3 Factor grid (모드별 10개 배분)

축: `날씨 4종(맑음/비/흐림/눈) × 동행자 5종(혼자/커플/가족유아/가족고령/친구) × 가용시간 3종(2h/4h/6h)`.
10개로는 3축 풀 조합(60가지)을 다 못 채우므로, **날씨×동행자를 우선 교차**시키고 가용시간은 순환 배정한다. 각 모드에서 "비/눈 + 실내 제약이 걸리는 동행자(가족유아/가족고령)" 조합을 최소 2개는 포함시켜 Graceful Degradation 경로(날씨 API 무자료 포함)를 반드시 커버한다.

### 2.4 파일/모듈 구조

```
eval/
├── cases.jsonl          # 골든셋 30줄
├── dataset.py           # 로더
├── metrics.py           # Task 3
├── judge.py             # Task 2
├── judge_prompt.py      # Task 2 (버전 헤더)
├── run_judge.py         # Task 4 CLI
└── baseline.json        # Task 4 (생성물, 최초 실행 시 --save-baseline로 생성)
tests/test_eval_dataset.py
```

### 2.5 데이터 계약 (`cases.jsonl` 1줄)

```jsonc
{
  "case_id": "nearby-001",
  "mode": "근교",
  "today": "2026-08-07",
  "ctx": {
    "user_query": "...", "current_lat": 0.0, "current_lng": 0.0,
    "weather": "비", "duration_hours": 4, "transport": "자차",
    "companion": "가족(유아)", "mood": "",
    "candidate_spots": [
      {"place_id": 1, "place_name": "...", "category": "관광",
       "distance_min": 12.0, "indoor": true}
    ]
  },
  "expectations": {
    "min_indoor_places": 1,
    "forbidden_place_ids": [],
    "max_total_minutes": 300,
    "expect_places_range": [2, 5]
  },
  "reference_example": null,
  "tags": ["rain", "family_infant", "short"]
}
```

`ctx`는 `ContextData` 필드명과 1:1 — `ContextData(**row["ctx"])`로 바로 복원. 계약이 어긋나면 로더가 즉시 실패해야 드리프트를 잡는다.

### 2.6 시그니처

```python
# eval/dataset.py
@dataclass(frozen=True)
class GoldenCase:
    case_id: str
    mode: str
    today: date
    ctx: ContextData
    expectations: dict
    reference_example: str | None
    tags: list[str]

def load_cases(path: Path, *, mode: str | None = None) -> list[GoldenCase]: ...
```

### 2.7 테스트/수용 기준
- 30줄, 모드별 정확히 10줄, `case_id` 중복 0
- 전부 `ContextData` 생성 성공
- `expectations` 키 화이트리스트 검증(오타 방지)
- 각 축 수준이 모드별 2회 이상 등장 — 커버리지를 테스트로 강제
- API 키 없이 전부 통과

---

## 3. Task 2 — LLM-as-Judge 5축 채점기

### 3.1 범위와 완료 정의
코스 1개 + 입력 컨텍스트를 받아 5축(조건반영도·감성일치·동선합리성·다양성·정보완결성)을 1~5점 + 근거로 채점. 완료 정의: 골든셋 5개 케이스에 대해 동일 입력 2회 실행 시 점수가 동일(결정론).

### 3.2 설계 결정

**모델**: gpt-4o(D2). 비용 실측 — 회귀 1회(생성 30건+채점 30건) 약 1,000원 이하로 결정 변수 아님. 진짜 쟁점은 생성자=심판 self-preference bias이며, 이는 감수하고 리포트에 명시.

**채점 방식**: 5축 1콜(비용·지연 1/5, 서술 일관성) + halo effect 완화 장치 2개.
1. **reason-then-score**: 각 축에서 근거 문장을 먼저, 점수를 나중에 출력하도록 필드 순서 강제(점수 먼저 내면 근거가 사후 정당화됨).
2. **anchor rubric**: 1점/3점/5점이 각각 무엇인지 구체적으로 명시 — "좋다/나쁘다"만 주면 3~4점에 몰리는 중심화 경향 발생.

**척도**: 1~5 Likert + anchor. Pairwise 비교는 baseline 대비 회귀 감지에 기준점이 없어 부적합(Sprint 7 A/B용으로 남김).

**temperature**: 0 고정(일관된 판단용) + `seed` 고정.

### 3.3 파일/모듈 구조 및 시그니처

```python
# eval/judge_prompt.py
# judge_prompt_id: course_quality_judge / version: v1.0 / author: 이정훈
JUDGE_SYSTEM_PROMPT: str   # 5축 anchor rubric 포함
def build_judge_user_message(case: GoldenCase, course: StreamingCourseModel) -> str: ...

# eval/judge.py
class AxisScore(BaseModel):
    reason: str                      # 필드 순서: 근거 먼저
    score: int = Field(ge=1, le=5)

class JudgeResult(BaseModel):
    condition_fit: AxisScore    # 조건반영도
    mood_match: AxisScore       # 감성일치
    route_sanity: AxisScore    # 동선합리성
    diversity: AxisScore        # 다양성
    completeness: AxisScore     # 정보완결성

    @property
    def mean(self) -> float: ...

async def judge_course(
    case: GoldenCase, course: StreamingCourseModel, *, model: str = JUDGE_MODEL
) -> JudgeResult | None:
    """None = 채점 실패(Graceful Degradation) — 러너가 skip 처리."""
```

judge 프롬프트를 별도 파일 + 버전 헤더로 분리하는 이유: judge 프롬프트가 바뀌면 점수가 통째로 이동한다. "프롬프트가 나빠짐"과 "심판이 엄해짐"을 구분하려면 judge 자체도 버전 관리 대상이어야 한다(eval versioning).

### 3.4 테스트/수용 기준
- `JudgeResult` Pydantic 검증 통과, 5축 전부 1~5 정수
- 동일 입력 2회 → 동일 점수
- 의도적으로 망친 코스(비 오는 날 전부 야외)에 `condition_fit ≤ 2`
- judge 호출 실패 시 `None` 반환, 러너 비정상 종료 안 함

---

## 4. Task 3 — 구조적 메트릭 (`eval/metrics.py`)

### 4.1 범위와 완료 정의
LLM 없이 결정론적으로 코스를 검사하는 순수 함수 집합. 완료 정의: API 호출 0회로 전 항목 pytest 통과.

### 4.2 핵심 설계 결정 — Judge와의 역할 분리

```
코스 출력 → [구조 메트릭: 결정론·무료·즉시] → FAIL이면 여기서 중단, judge 호출 안 함
                                            → PASS만 [LLM Judge: 비결정·유료]로 진행
```

JSON이 깨졌는지, 장소 수가 범위 밖인지, 없는 place_id를 지어냈는지는 규칙으로 100% 판정 가능하다. 이를 LLM에게 물으면 비용·시간을 버리며 정확도까지 떨어진다(팀 원칙 "LLM as tiebreaker"의 적용). 구조 FAIL 케이스는 judge 점수 집계에서 제외해 비용도 절감한다.

### 4.3 측정 항목

| 항목 | 판정 |
|---|---|
| JSON 유효성 | `StreamingCourseModel.model_validate_json()` 통과 여부 |
| 장소 수 | 스키마 1~10 + 케이스별 `expect_places_range` |
| place_id grounding | `candidate_spots` 밖 `place_id` 사용 = 환각 → FAIL |
| fallback 탐지(F7) | `bot_message`가 `_build_fallback()` 문구와 일치하면 FAIL. 없으면 API 전멸 시에도 "전부 통과"로 오판됨 |
| 이동거리/시간 예산 | 아래 참조 |

**이동거리 합리성**: `hybrid_search.estimate_travel_minutes()` 재사용(rag/ 의존 추가) 대신, fixture의 `SpotCandidate.distance_min`(현재 위치→각 장소, 이미 존재하는 필드)만 사용. 장소 간 거리 데이터는 없으므로 "동선 최적성"은 이 메트릭으로 판정 불가 — 그 역할은 Judge의 `route_sanity` 축이 맡는다. 정직한 정의:

```
체류시간 합(duration_min) + max(distance_min) × 2  ≤  duration_hours × 60 × 1.2
```

### 4.4 시그니처

```python
# eval/metrics.py — 전부 순수 함수, LLM·네트워크 의존 0
class StructuralMetrics(BaseModel):
    json_valid: bool
    is_fallback: bool
    place_count: int
    place_count_ok: bool
    grounded: bool
    hallucinated_ids: list[int]
    estimated_total_min: int
    time_budget_ok: bool
    passed: bool   # 위 전부 AND → judge 게이트

def measure(ctx: ContextData, expectations: dict, raw_output: str) -> StructuralMetrics: ...
```

`case: GoldenCase`가 아니라 `ctx`+`expectations`를 직접 받는 이유(구현 중 조정) — T3는 T1(`eval/dataset.py`의 `GoldenCase`)에 완전 독립이라고 명시했는데, `GoldenCase`를 시그니처에 넣으면 그 독립성이 깨진다. `measure()`가 실제로 필요한 건 `ctx.candidate_spots`/`ctx.duration_hours`와 `expectations` 딕셔너리뿐이라 이 둘만 받는 게 §7 의존성 매트릭스와 일치한다.

`raw_output: str`을 받는 이유 — JSON 파싱 실패 자체가 측정 대상이므로 파싱된 객체가 아니라 원문 문자열을 받아야 한다.

### 4.5 테스트/수용 기준
정상 코스는 전 항목 PASS. 깨진 JSON/장소 11개/후보 밖 place_id/시간 3배 초과/폴백 문구는 각각 해당 항목만 FAIL. 전 테스트 API 키 없이 실행(CI 안전).

---

## 5. Task 4 — 골든셋 회귀 테스트 CLI (`eval/run_judge.py`)

### 5.1 범위와 완료 정의
`python eval/run_judge.py --dataset eval/cases.jsonl --out eval/report.json` 한 줄로 생성→구조측정→채점→집계→baseline 비교까지 수행. 완료 정의: 프롬프트를 의도적으로 훼손했을 때 스크립트가 회귀를 탐지하고 non-zero exit.

### 5.2 설계 결정

**회귀 판정 기준(혼합)**: 구조 메트릭은 절대 기준(`passed` 비율 100% 요구 — JSON이 깨지는 데 "이전보다 덜 깨짐"은 없다), judge 점수는 baseline 대비 델타(모드별 평균 -0.3 이상 하락 시 FAIL). LLM 평가 점수의 절대값은 rubric·모델이 바뀌면 이식성이 없어, 동일 조건 하의 델타만 신뢰할 수 있다.

**pytest 분리**: 독립 CLI로 두고 pytest에는 API 호출 없는 부분(metrics 순수함수, 골든셋 스키마)만 포함. eval은 "프롬프트를 고쳤을 때 수동으로 도는 배치"이지 CI 게이트가 아니다.

**재현성(D3)**: `generate_course()`에 `temperature: float = 0.3` keyword-only 인자 추가. 프로덕션 기본값 무변경, eval만 0 전달. 팀 계약 문서(`types.py`)에 시그니처 변경을 주석으로 고지.

### 5.3 데이터 계약 (`report.json`)

```jsonc
{
  "run": {"timestamp": "...", "git_sha": "...", "prompt_version": "v2.0",
          "judge_prompt_version": "v1.0", "judge_model": "gpt-4o",
          "gen_model": "gpt-4o", "temperature": 0, "case_count": 30},
  "cases": [{"case_id": "...", "mode": "...", "structural": {...},
             "judge": {...}, "skipped_reason": null}],
  "aggregate": {
    "overall": {...},
    "by_mode": {"근교": {"structural_pass_rate": 1.0, "axis_means": {...}}},
    "by_tag": {"rain": {...}, "family_infant": {...}}
  },
  "regression": {"baseline_ref": "eval/baseline.json", "deltas": {...},
                 "verdict": "PASS", "failures": []}
}
```

`by_tag` 집계가 Task 1 factor grid를 실제 진단력으로 바꾸는 지점 — "비 오는 날 케이스만 평균 하락"을 즉시 확인 가능.

`prompt_version`을 리포트에 찍기 위해 `prompt.py`에 `PROMPT_VERSION = "v2.0"` 상수 노출 필요(현재 주석에만 존재, MODE_WEIGHTS와 같은 SSOT 패턴).

### 5.4 CLI 시그니처

```
python eval/run_judge.py --dataset eval/cases.jsonl --out eval/report.json
                         [--mode 근교] [--baseline eval/baseline.json]
                         [--save-baseline] [--limit N] [--no-judge]
```
`--no-judge`: 구조 메트릭만(무료·빠름, 프롬프트 수정 중 자주 사용). `--limit N`: 스모크 테스트용.

### 5.5 테스트/수용 기준
30케이스 완주, `report.json` 스키마 유효. 의도적 훼손 테스트(예: "비/흐림→실내" 지시 제거) 시 `condition_fit` 하락 탐지 + exit code 1. 중간 API 1건 실패해도 나머지 완주(Graceful Degradation). `--no-judge`는 API 키 없이도 동작.

---

## 6. Task 5 — 계절 컨텍스트 L3 통합 (`traveler/core/season_context.py`)

### 6.1 범위와 완료 정의
`build_context_message()`가 L1(날씨)·L2(계절)에 더해 L3(축제)를 주입. 완료 정의: 축제 데이터가 있으면 컨텍스트 블록 추가, 없으면 기존과 100% 동일 출력(Graceful Degradation).

### 6.2 설계 결정

**데이터 공급 경로(D4)**: `core/`가 SQLite에 직접 의존하지 않는다(F5로 로컬 검증도 불가능). `season.py`의 설계 철학 — "외부 의존 없는 건 core/ 순수 함수, 있는 건 ContextData 계약 경유" — 을 그대로 따라, 이번 스프린트는 **조립 순수 함수만 완결**하고 데이터는 파라미터로 주입받는다. 실 데이터 연결(`ContextData.festivals` 필드)은 다음 스프린트로 이관.

**SystemPrompt vs user 메시지(결정 C)**: 로드맵 문구("SystemPrompt에 동적 삽입")와 달리, 동적 데이터는 기존처럼 user 메시지로 넣는다. SystemPrompt는 정적 유지 — 요청마다 바뀌면 prompt caching이 깨지고 버전 관리가 불가능해진다.

```
SystemPrompt(_COMMON_BASE): "축제·행사 정보가 제공되면 계절 특화 추천에 우선 반영하십시오." ← 규칙(정적, 1줄 추가)
user 메시지:                "[진행 중인 축제·행사] ..." ← 데이터(동적)
```

**축제를 코스 장소에 넣을지(D7)**: 넣지 않는다. 컨텍스트 블록으로만 제공해 LLM이 `bot_message`에 언급하거나 후보 선택 참고로만 쓰게 한다. `place_id` grounding 계약 위배를 피하기 위함(가정 — 기획 요건과 다르면 정정 필요).

**선별 기준**: `is_active` AND `event_start_date ≤ today ≤ event_end_date` AND 인근 → 상위 3건(토큰 예산).

### 6.3 시그니처

```python
# traveler/core/season_context.py — 신설, season.py는 무수정(외과적 변경)
@dataclass(frozen=True)
class FestivalInfo:
    name: str
    event_start_date: str   # "YYYYMMDD"
    event_end_date: str
    event_place: str = ""

def filter_active_festivals(
    festivals: list[FestivalInfo], today: date, *, limit: int = 3
) -> list[FestivalInfo]: ...

def build_season_context(
    today: date | None = None, festivals: list[FestivalInfo] | None = None
) -> str:
    """L2(계절)+L3(축제) 컨텍스트 블록 조립.
    festivals가 None/빈 리스트면 기존 '현재 계절: X' 한 줄만 반환 → 회귀 없음."""
```

```python
# traveler/core/prompt.py — 최소 변경
def build_context_message(
    ctx: ContextData, today: date | None = None,
    festivals: list["FestivalInfo"] | None = None,
) -> str:
    # 기존 f"현재 계절: {get_season(today)}" 자리를
    # build_season_context(today, festivals)로 교체
```

`festivals`를 세 번째 선택 파라미터로 두는 이유 — 호출부(`course_generator`)를 지금 수정하지 않아도 되고, `ContextData.festivals`가 나중에 추가되면 `ctx.festivals`를 넘기도록 한 줄만 바꾸면 된다.

### 6.4 테스트/수용 기준 (`tests/test_season_context.py`, API·DB 불필요)
`festivals=None` → 기존 `build_context_message` 출력과 완전히 동일(회귀 0 증명). 축제 1/3/5개(상한 3 적용) 블록 형태 검증. 날짜 경계(`today == start`, `today == end` 포함/전후 제외). 잘못된 날짜 문자열이 섞여도 해당 건만 스킵. `get_season()` 무수정 — 기존 `test_season.py` 전부 통과.

---

## 7. 의존성 매트릭스 및 실행 순서

```
T3(구조 메트릭, 완전 독립) ──약한 선행─▶ T1(골든셋 30개)
                                          │ 강한 선행(채점 대상 필요)
                                          ▼
                                    T2(Judge) ──┐
                                                 │ 강한 선행(3개 전부 필요)
                                    T3 ──────────┤
                                                 ▼
                                          T4(회귀 러너)
                                                 │
                                    T5(계절 L3, 코드상 독립) ──▶ baseline 무효화 주의
```

| 쌍 | 관계 |
|---|---|
| T3 → T1 | 약한 선행. 메트릭 항목이 정해져야 `expectations` 키 확정. T1 factor grid 설계는 지금 바로 시작 가능 |
| T1 → T2 | 강한 선행. 단 judge 프롬프트·rubric 작성은 T1과 병렬 가능 |
| T1+T2+T3 → T4 | 강한 선행. CLI 골격·report 스키마는 병렬 설계 가능 |
| T2 ↔ T3 | 독립. 러너가 게이트로 연결 |
| T5 ↔ T1~T4 | **구현은 독립, 실행은 순서 제약**: T5는 프롬프트 입력을 바꾸므로 적용 시 T4 baseline이 무효화됨 |

**T5 순서 제약을 서사로 활용**: `T4 완성 → baseline 촬영(L3 적용 전) → T5 적용 → 회귀 러너 재실행 → "L3 도입 전/후 5축 점수 변화" 리포트`. 이러면 이번 스프린트가 "평가 체계를 만들고, 그 체계로 이번 스프린트의 프롬프트 변경(L3)을 실제로 검증했다"는 자기완결적 결과물이 된다.

### 실행 일정 (8/4~8/7, 4일)

| 일자 | 작업 |
|---|---|
| Day 1 (8/4) | T3 `metrics.py` 순수 함수 + pytest 전량 통과 / T1 factor grid 설계 + 근교 10개 |
| Day 2 (8/5) | T1 장거리·타지 20개 완성 / T2 rubric + `judge_prompt.py` + `JudgeResult` 스키마 + 5케이스 시범 채점 |
| Day 3 (8/6) | T4 `run_judge.py` 완성(`--no-judge`부터 → 전체) / **baseline 촬영**(L3 적용 전) / T5 `season_context.py` 순수 함수 + 테스트 |
| Day 4 (8/7) | T5 `prompt.py` 주입 + SystemPrompt 규칙 1줄 / 회귀 러너 재실행 → L3 전/후 비교 리포트 / 의도적 훼손 테스트로 러너 자체 검증 |

**절대 잘라서는 안 되는 것**: T3(구조 메트릭)와 T4의 fallback 탐지 — 없으면 F7 때문에 "전부 성공"이라는 거짓 리포트가 나온다.

---

## 8. 다음 스프린트로 이관되는 항목 (이번 스프린트 범위 아님)

- `ContextData.festivals` 필드 추가 — 이수연에게 요청 필요, 계약 변경이라 팀 통보 필요(`types.py` 규칙)
- 골든셋 30 → 100 확장 (Sprint 8 예정, 이미 `info/LLM_sprint.md`에 명시)
- Judge 모델 교차검증(gpt-4o ↔ 타 모델, self-preference bias 완화, Sprint 7)
- 실 RAG 경로 골든셋 — `SpotCandidate`/`HybridSearchResult` 타입 정합 후(F4), 이수연 쪽 작업 완료가 선행 조건
