# Sprint 7 — 멀티턴 수정 인텐트 5종 설계

- 작성: 2026-08-30 · 이정훈
- 대상 마감: 2026-09-04 (Sprint 7 종료) · 시연: 2026-09-08 한이음 중간발표
- 로드맵 근거: v15 "멀티턴 수정 인텐트 5종 — 9/4(금) · **발표 핵심 시연**"
- CLAUDE.md 근거: "멀티턴 인텐트 5종 `바꿔줘 / 추가 / 빼줘 / 순서변경 / 전체재생성`
  — 해당 부분만 재생성해 Guardrails 재호출 범위 최소화"

---

## §0 실사 사실 (코드 직접 확인, 2026-08-30)

| # | 사실 | 근거 |
|---|---|---|
| F1 | `traveler/core/intent.py`는 **동승자(companion) 파싱 전용**이며 수정 인텐트와 무관하다. 이름이 비슷해 착수된 것으로 오인하기 쉽다. | `intent.py` — `parse_companion()` 단일 함수 |
| F2 | 단발 LLM 호출 + JSON mode + Pydantic 검증 + 실패 시 `None` 반환 패턴의 선례가 이미 있다. 새 분류기는 이 구조를 그대로 따른다. | `eval/judge.py` `judge_course()` |
| F3 | `llm_client.py`에 gpt-4o-mini 상수가 없다(`MODEL`·`JUDGE_MODEL` 둘 다 `"gpt-4o"`). 상수 추가가 선행된다. | `llm_client.py:21,26` |
| F4 | `StreamingCourseModel`은 `visit_sequence`가 1..n 연속임을 검증하고, 정렬된 리스트를 반환한다. 중복·gap은 거부된다. | `schema.py` `check_sequence_sequential` |
| F5 | `recommended_places`는 1~10개여야 한다. 마지막 1곳을 빼면 스키마 위반이다. | `schema.py` `Field(min_length=1, max_length=10)` |
| F6 | `place_id`는 `str`(content_id), `place_name`·`duration_min`은 Optional이다. | `schema.py` `RecommendedPlaceModel` |
| F7 | 후보 풀은 `ContextData.candidate_spots: list[SpotCandidate]`다. 코스는 이 목록 밖 장소를 쓰면 안 된다(grounding). | `types.py` · `prompt.py` `_COMMON_BASE` |
| F8 | 모델 선택 기준상 멀티턴 인텐트 분류는 `gpt-4o-mini`다("5종 분류, 저지연 필수"). | CLAUDE.md 모델 선택 기준표 |
| F9 | `main.py`의 `/api/chat`은 매 메시지마다 `rag_get_context()`를 새로 호출한다. 현재 코스나 `candidate_spots`를 세션에 저장하지 않는다(저장하는 건 `visited`·`prefs`뿐). | `main.py:110-181` (Outside Voice, 2026-08-30) |
| F10 | `generate_course(ctx, travel_mode, *, temperature)`는 `ctx.user_query` 외에 후속 발화를 받는 파라미터가 없다. | `course_generator.py:54-59` (Outside Voice, 2026-08-30) |

---

## §1 결정 로그

**D1 — 모듈명은 `traveler/core/edit_intent.py`.**
F1의 `intent.py`(동승자 파싱)와 한 파일에 섞으면 두 개의 무관한 책임이 공존한다(SRP 위반).
파일명에 `edit`를 넣어 "코스 수정 인텐트"임을 이름만으로 구분한다.

**D2 — 분류·대상 식별·신규 장소 선택을 LLM 호출 1회로 처리한다.**
"두 번째를 더 조용한 데로 바꿔줘"는 ①의도(replace) ②대상(2번) ③대체 장소 선택이 한 문장에 들어있다.
세 가지를 따로 부르면 호출 3회에 지연도 3배인데, gpt-4o-mini는 이미 후보 목록을 컨텍스트로
받으므로 한 번에 셋 다 답할 수 있다. 규칙 기반 선택(후보 중 첫 번째)도 검토했으나
"더 조용한 데로" 같은 사용자 의도를 전혀 반영하지 못해 시연 품질이 떨어진다.

**D3 — 실제 코스 수정은 LLM 없는 순수 함수로 한다.**
5종 중 4종(replace·add·remove·reorder)은 리스트 조작이다. 새 코스를 통째로 다시 생성하면
사용자가 유지하길 원한 장소까지 바뀐다 — "해당 부분만 재생성"이라는 설계 의도와 정반대다.
부수 효과로 API 없이 테스트 가능해진다.

**D4 — `regenerate_all`만 `generate_course()`를 재호출한다.**
전체 재생성은 정의상 부분 수정이 아니다. 기존 함수를 그대로 재사용하고 새 코드를 만들지 않는다.

**D5 — 분류 실패 시 `None`을 반환한다(예외 전파 금지).**
`judge_course()`와 동일한 Graceful Degradation. 호출부가 "무슨 말씀인지 잘 모르겠어요"로
되묻게 하고, 파이프라인은 죽지 않는다.

**D6 — 적용 불가한 수정은 원본 코스를 그대로 돌려주고 사유를 문장으로 전달한다.**
마지막 한 곳 삭제(F5 위반), 범위 밖 대상, 남은 후보 없음 등은 예외가 아니라 정상적인
"거절"이다. 예외를 던지면 호출부가 매번 try/except를 써야 하고, 조용히 무시하면 사용자가
왜 안 바뀌었는지 모른다. 반환값에 사유 문장을 실어 그대로 `bot_message`로 쓴다.

**D7 — 수정 후 `visit_sequence`는 항상 1..n으로 재부여한다.**
F4 때문에 삭제·삽입 후 번호를 그대로 두면 gap이 생겨 스키마 검증에서 코스 전체가 거부된다.
재시퀀싱을 실행 함수의 책임으로 못박아, 호출부가 잊어버릴 여지를 없앤다.

**D8 — 새로 코스에 들어가는 장소의 `duration_min`은 60(분) 고정값이다** (/plan-eng-review
2026-08-30, Outside Voice 지적으로 `replace`까지 범위 확정 — 최초 문서에서 `add`만 적었던 건
누락이었다). `SpotCandidate`에는 `distance_min`만 있고 `duration_min`이 없다 — 지금까지
`duration_min`은 항상 GPT가 코스 생성 시점에 즉석으로 정하던 값이다. `add`뿐 아니라
`replace`로 들어가는 새 장소도 후보 풀(`SpotCandidate`)에서 오므로 동일하게 `duration_min`이
없다 — 기존 대상의 체류시간을 그대로 물려주면 새 장소에 거짓 정보이므로 두 action 모두
고정값을 쓴다. LLM을 다시 부르지 않고 이번 스프린트 안에 끝내는 가장 단순한 방법이고,
원래 값도 GPT의 추정치였으므로 정밀도를 잃는 게 아니다.

**D9 — `revise_course()`는 재조립 실패 시 원본 코스로 폴백한다** (/plan-eng-review 2026-08-30).
`apply_edit()`이 리스트를 조작한 뒤 `StreamingCourseModel(...)`로 재구성하는 지점은
D7 재시퀀싱에 버그가 있거나 예상 못한 입력이 들어오면 `ValidationError`를 던질 수 있다.
코드베이스 전체가 이미 `_build_fallback()` 패턴(API가 죽어도 서비스는 안 죽음)으로
일관되어 있는데, 여기서만 예외가 새면 발표 핵심 시연 기능이 그 한 지점에서 서버를
죽인다. §6을 이 폴백을 명시하도록 갱신한다.

**D10 — "미사용 후보 추출"은 공유 헬퍼 `_unused_candidates()`로 뺀다** (/plan-eng-review 2026-08-30).
§4(분류기 프롬프트)와 §5(replace/add 거절 판정)가 둘 다 사실상
`candidate_spots - {p.place_id for p in course.recommended_places}`를 각자 계산하게
설계돼 있었다. 계산이 한 줄이라도 두 곳에 복제되면, 판정 기준이 바뀔 때(예: 이미
방문한 곳도 제외) 한쪽만 고치는 전형적인 버그 지점이 된다. 헬퍼로 빼는 비용은
사실상 0에 가깝다.

**D11 — 현재 코스·후보 풀은 `main.py`에 메모리 딕셔너리로 임시 저장한다** (Outside Voice
2026-08-30 — 가장 심각한 지적, F9). `revise_course(user_message, course, ctx, ...)`는
`course`·`ctx`를 파라미터로 받을 뿐 어디서 오는지는 모른다 — 지금 F9 때문에 실제로는
어디서도 안 온다. `main.py`에 `_session_courses: dict[str, tuple[StreamingCourseModel,
ContextData]]`를 두고 `/api/chat` 마지막에 저장, 후속 요청에서 조회해 `revise_course()`에
넘긴다. 이수연의 `session.py`(Redis+SQLite)에 정식 필드로 추가하는 안도 검토했으나, 그
파일이 이번 스프린트에 PR #30으로 활발히 바뀌는 중이라 지금 필드를 더하면 그쪽 머지
일정(8/28)이 더 밀릴 위험이 있다. 발표는 일단 작동하는 게 최우선이라 가장 작고
되돌리기 쉬운 방법을 먼저 쓴다. **⚠️ 데모 전용 스캐폴딩**이다 — 서버 재시작하면
날아가고, 여러 워커 프로세스 간 공유도 안 된다. 정식 영속화는 Sprint 8 이후 별도
과제(#28/`session.py`)로 이관한다.

**D12 — `regenerate_all`은 사용자의 후속 발화를 `generate_course()`에 그대로 전달한다**
(Outside Voice 2026-08-30, F10). `generate_course()`에 keyword-only `refinement: str | None
= None`을 추가하고, 있으면 `build_context_message()`가 조립하는 프롬프트에 "추가 요청:
{refinement}" 같은 섹션으로 덧붙인다. `revise_course()`가 `action == "regenerate_all"`일 때
`user_message`를 그대로 `refinement`로 넘긴다 — `EditIntent`에 별도 필드가 필요 없다,
사용자 원문 자체가 곧 refinement이기 때문이다. 기본값 `None`이라 기존 호출부(스모크
테스트 등)는 무변경.

**D13 — `add`는 삽입 위치를, `reorder`는 부분 이동을 지원한다**(완전판, Outside Voice
2026-08-30 지적에 사용자가 완전판으로 결정). `EditIntent`에 `insert_after: int | None`
(add 전용 — 대상 `visit_sequence` 뒤에 삽입, `None`이면 기존과 동일하게 끝에 추가)과
`move_sequence`·`move_to_position: int | None` 쌍(reorder 전용 — 장소 하나만 새 위치로
이동)을 추가한다. `reorder`는 이제 두 하위 모드다 — `new_order`가 채워지면 전체 순열
재배열(기존 동작), `move_sequence`/`move_to_position`이 채워지면 부분 이동. 프롬프트가
"전체 순서를 말했으면 new_order, 특정 장소 하나만 옮기라고 했으면 move_sequence·
move_to_position을 채우라"고 명시적으로 분기 지시한다. "카페를 먼저 가고 싶어" 같은
부분 요청에서 LLM이 나머지 장소 순서까지 임의로 추측해 바꾸는 것을 막는다(T3 스테이크).

---

## §2 파이프라인 내 위치

```
[1차 코스 생성 완료] ── main.py: _session_courses[session_id] = (course, ctx) 저장 (D11)
      │
      ▼
사용자 후속 발화 ("두 번째 빼줘")
      │
      ▼
main.py: _session_courses[session_id]에서 (course, ctx) 조회 (D11, 데모 전용)
      │
      ▼
★ classify_edit_intent()  ── gpt-4o-mini 1회 ──▶ EditIntent | None
      │
      ├─ None ─────────────▶ "무슨 말씀인지 잘 모르겠어요" (재질문)
      ├─ regenerate_all ───▶ generate_course(ctx, refinement=user_message) 재호출 (D4·D12)
      └─ 나머지 4종 ───────▶ ★ apply_edit()  ── LLM 없음 ──▶ 수정된 장소 목록
                                    │
                                    ▼
                          StreamingCourseModel 재조립 → Guardrails 1회
                                    │
                                    ▼
                    main.py: _session_courses[session_id] 갱신 (D11)
```

Guardrails가 코스 전체에 대해 한 번만 돌아간다 — CLAUDE.md의 "Guardrails 재호출 범위 최소화"가
여기서 충족된다.

**`course`·`ctx`가 어디서 오는지가 F9로 드러난 진짜 구멍이었다.** `revise_course()` 자체는
이 둘을 파라미터로만 받으므로 모듈 내부 설계는 바뀌지 않는다 — 조회·저장은 전부 `main.py`의
책임이다(D11). `edit_intent.py`는 세션이라는 개념을 전혀 몰라도 된다.

---

## §3 스키마

```python
class EditIntent(BaseModel):
    action: Literal["replace", "add", "remove", "reorder", "regenerate_all"]
    target_sequence: int | None      # replace·remove 대상의 visit_sequence
    new_place_id: str | None         # replace·add 시 넣을 후보 place_id
    insert_after: int | None         # add 전용(D13) — 이 visit_sequence 뒤에 삽입.
                                      #   None이면 끝에 추가(기존 동작과 동일)
    new_order: list[int] | None      # reorder 전용, 전체 순열 — 새 순서(기존 visit_sequence 배열)
    move_sequence: int | None        # reorder 전용(D13), 부분 이동 — 옮길 장소
    move_to_position: int | None     # reorder 전용(D13), 부분 이동 — 새 위치(1부터)
    reason: str                      # 사용자에게 보여줄 근거 한 문장
```

`reason`을 마지막이 아니라 스키마에 명시적으로 둔 이유는 `AxisScore`(judge.py)의
reason-then-score와 같은 맥락이다 — 모델이 근거를 먼저 쓰게 하면 분류가 안정된다.
다만 여기서는 action이 먼저 와야 파싱이 단순하므로, 프롬프트 지시문에서 "reason을 먼저
생각하고 action을 결정하라"고 명시하는 쪽으로 처리한다.

**`reorder`가 필드 2쌍으로 갈리는 이유(D13)**: "전체 순서를 다 새로 정해줘"는 순열
전체(`new_order`)가 필요하지만, "카페를 먼저 가고 싶어"는 장소 하나만 옮기면 된다.
후자를 전체 순열로 강제하면 LLM이 요청하지 않은 나머지 순서까지 추측해서 만들어야
한다 — 프롬프트가 두 표현 중 어느 쪽인지 보고 한 쌍만 채우도록 명시적으로 분기한다.

| action | 채워지는 필드 | 예시 발화 |
|---|---|---|
| `replace` | target_sequence, new_place_id | "두 번째를 더 조용한 데로 바꿔줘" |
| `add` | new_place_id (+ 선택: insert_after) | "카페 하나 추가해줘" / "카페를 세 번째에 넣어줘" |
| `remove` | target_sequence | "불국사는 빼줘" |
| `reorder`(전체) | new_order | "순서 완전히 다시 짜줘" |
| `reorder`(부분) | move_sequence, move_to_position | "카페를 먼저 가고 싶어" |
| `regenerate_all` | (없음, `user_message`가 곧 refinement — D12) | "다 마음에 안 들어, 카페 중심으로 다시 짜줘" |

```python
# WHY: §4(분류기 프롬프트)와 §5(replace/add 거절 판정) 둘 다 "아직 코스에 없는 후보"가
# 필요하다(D10). 계산이 한 줄이라도 두 곳에서 각자 재구현하면 판정 기준이 바뀔 때
# 한쪽만 고치는 게 이런 버그의 전형적인 발생 지점이라, 여기 하나로 SSOT를 둔다.
def _unused_candidates(
    course: StreamingCourseModel, ctx: ContextData
) -> list[SpotCandidate]:
    used = {p.place_id for p in course.recommended_places}
    return [s for s in ctx.candidate_spots if s.place_id not in used]
```

---

## §4 분류기 — `classify_edit_intent()`

```python
# WHY: 후속 발화가 5종 중 무엇인지, 어느 장소를 대상으로 하는지 판정한다.
# INPUT: 사용자 후속 발화 + 현재 코스 + ContextData(후보 풀)
# OUTPUT: EditIntent | None  (None = 분류 실패 → 호출부가 재질문)
# COST: ~700 tokens · gpt-4o-mini · 약 $0.15 per 1,000 calls
async def classify_edit_intent(
    user_message: str,
    course: StreamingCourseModel,
    ctx: ContextData,
    *,
    model: str = INTENT_MODEL,
) -> EditIntent | None:
```

- `temperature=0` — 분류는 판단이므로 결정론적 출력이 필요하다.
- `response_format={"type": "json_object"}` — 파싱 안정성.
- 프롬프트에 **현재 코스(순번·장소명)와 미사용 후보 목록**(`_unused_candidates()`, D10)을
  함께 넣는다. 후보를 안 주면 `new_place_id`를 지어내(환각) grounding이 깨진다(F7).
- **`reorder` 분기 지시(D13)**: "사용자가 전체 순서를 다 말했으면 `new_order`를, 장소
  하나만 옮기라고 했으면 `move_sequence`·`move_to_position`을 채워라"를 프롬프트에
  명시한다. 두 쌍을 동시에 채우거나 둘 다 비우면 §5가 거절한다(잘못된 조합).
- 예외는 `judge_course()`와 동일하게 전부 흡수하고 `None`을 반환한다(D5).
- 프롬프트는 `prompt.py`가 아니라 `edit_intent_prompt.py`에 버전 헤더와 함께 둔다
  (CLAUDE.md 프롬프트 버전 관리 규칙 · `judge_prompt.py` 선례).

---

## §5 실행기 — `apply_edit()`

```python
# WHY: LLM 없이 코스를 실제로 수정한다. 4종 모두 리스트 조작이라 순수 함수로 충분하다.
# OUTPUT: (수정된 장소 목록, 사용자에게 보여줄 설명 한 문장)
#         적용 불가 시 원본 목록 + 거절 사유를 돌려준다(D6).
def apply_edit(
    course: StreamingCourseModel,
    edit: EditIntent,
    ctx: ContextData,
) -> tuple[list[RecommendedPlaceModel], str]:
```

Dispatch Table 패턴으로 action → 핸들러를 매핑한다(`get_system_prompt()` 선례).

| action | 동작 | 거절 조건 |
|---|---|---|
| `remove` | 대상 제거 후 재시퀀싱 | 장소가 1곳뿐(F5) · 대상 없음 |
| `replace` | 같은 자리에 `new_place_id` 교체, `duration_min=60`(D8) | 대상 없음 · `new_place_id`가 `_unused_candidates()` 밖 · 이미 코스에 있음 |
| `add` | `insert_after` 뒤에 삽입(없으면 끝), `duration_min=60`(D8·D13) | 이미 10곳(F5) · `new_place_id`가 `_unused_candidates()` 밖 · 중복 · `insert_after`가 존재하지 않는 순번 |
| `reorder`(전체) | `new_order` 순서로 재배열 | 기존 순번의 순열이 아님 |
| `reorder`(부분, D13) | `move_sequence` 장소를 `move_to_position`으로 이동, 나머지는 순서 유지 | `move_sequence`가 존재하지 않는 순번 · `move_to_position`이 범위 밖(1~장소 수) · 두 쌍(`new_order`/`move_*`)이 동시에 채워지거나 둘 다 빔 |
| `regenerate_all` | 여기서 처리하지 않음 | 호출부가 `generate_course(ctx, refinement=user_message)` 재호출(D4·D12) |

거절 조건의 "후보 밖" 판정은 §3의 `_unused_candidates()` 하나만 쓴다(D10) — §4의
프롬프트 주입과 같은 함수라 두 곳의 판정 기준이 어긋날 수 없다.

모든 경로의 마지막에서 `visit_sequence`를 1..n으로 재부여한다(D7).

---

## §6 오케스트레이터 — `revise_course()`

```python
# WHY: 시연·API가 호출할 단일 진입점. 분류 → 분기 → 코스 재조립까지 한 번에.
# OUTPUT: StreamingCourseModel — 항상 유효한 코스. 실패해도 원본 + 설명 bot_message.
async def revise_course(
    user_message: str,
    course: StreamingCourseModel,
    ctx: ContextData,
    travel_mode: str = "타지",
) -> StreamingCourseModel:
```

분류 실패면 원본 코스에 "무슨 말씀인지 잘 모르겠어요…" `bot_message`를 붙여 돌려준다.

**`regenerate_all`은 후속 발화를 흘려보낸다(D12).** `action == "regenerate_all"`이면
`generate_course(ctx, travel_mode, refinement=user_message)`을 호출한다. `EditIntent`에
별도 필드가 없는 이유는 사용자 원문 그 자체가 이미 refinement이기 때문이다 —
`classify_edit_intent()`가 요약하거나 가공할 필요가 없다.

**`course`·`ctx`의 출처는 이 함수의 관심사가 아니다(D11).** `main.py`가
`_session_courses[session_id]`에서 조회해 넘겨주고, 반환값으로 다시 저장한다.
`revise_course()`·`edit_intent.py`는 세션이라는 개념을 모른 채로 남는다 — D11이
데모 전용 스캐폴딩으로 격리된 채 유지되는 지점이다.

**재조립도 명시적으로 방어한다(D9).** `apply_edit()`의 결과로 `StreamingCourseModel(...)`을
다시 만드는 지점은 D7 재시퀀싱 버그나 예상 못한 입력으로 `ValidationError`를 던질 수 있다.
이 지점을 감싸지 않으면 코드베이스 전체가 지켜온 `_build_fallback()` 패턴(API가 죽어도
서비스는 안 죽음)이 여기서만 깨진다 — 발표 핵심 시연 기능이라 리스크를 더 낮게 잡는다.

```python
try:
    revised = StreamingCourseModel(
        bot_message=explanation, recommended_places=updated_places
    )
except ValidationError as e:
    logger.warning("코스 수정 재조립 실패, 원본 유지: %s", e)
    revised = course.model_copy(
        update={"bot_message": "수정을 반영하지 못했어요, 원래 코스를 유지할게요."}
    )
```

어떤 경로로도 예외가 밖으로 나가지 않는다.

---

## §7 테스트 전략

API 키 없이 전부 도는 것을 목표로 한다(현재 팀 워크트리에 키가 없고, CI에서도 없을 것).

- **분류기**: `@patch("traveler.core.edit_intent.get_client")` + `AsyncMock` — `test_eval_judge.py`와 동일한 패턴. 5종 정상 분류 / JSON 깨짐 / API 예외 / 스키마 위반 4케이스.
- **실행기**: 순수 함수라 목업 불필요. action별 정상 1개 + 거절 조건 1개씩.
- **재시퀀싱 회귀**: 삭제·삽입 후 결과가 `StreamingCourseModel` 검증을 실제로 통과하는지 확인한다(F4가 무너지면 코스 전체가 폴백으로 강등되므로 이게 가장 중요한 테스트다).
- **`_unused_candidates()`**(D10): 코스가 후보 전부를 쓴 경우(빈 리스트) / 일부만 쓴 경우 2케이스.
  분류기·실행기 양쪽이 이 하나를 호출하는지도 각 테스트에서 간접 확인된다.
- **`add`의 `duration_min`**(D8): 새로 추가된 장소의 `duration_min == 60`을 명시적으로 단언.
- **재조립 실패 폴백**(D9, /plan-eng-review 신규 — 이게 없으면 코드베이스 전체의
  Graceful Degradation 일관성이 깨지는 지점이라 우선순위가 가장 높다): D7 재시퀀싱이
  gap 있는 결과를 만들도록 강제로 깨뜨린 뒤(예: `apply_edit`을 몽키패치해 잘못된 순번
  반환), `revise_course()`가 예외를 던지지 않고 원본 코스 + "수정을 반영하지 못했어요"
  `bot_message`를 반환하는지 확인한다.
- **`regenerate_all` refinement 전달**(D12, Outside Voice — 이것도 우선순위 높음: 발표
  핵심 시연 5종 중 하나가 사용자 말을 무시하는지 여부를 가른다): `generate_course`를
  목업해 `revise_course()`가 `refinement=user_message`로 호출하는지 단언. 별도로
  `build_context_message(ctx, refinement="...")`가 실제로 그 문구를 프롬프트에
  포함하는지도 `test_prompt.py` 쪽에 1개 추가.
- **`insert_after`**(D13): 중간 삽입 1개 · 존재하지 않는 순번 거절 1개 · 생략 시
  기존처럼 끝에 추가되는지 회귀 1개.
- **부분 이동 `move_sequence`/`move_to_position`**(D13): 정상 이동(나머지 순서 유지되는지
  까지 확인) 1개 · `move_sequence` 없음 거절 1개 · `move_to_position` 범위 밖 거절 1개 ·
  `new_order`와 동시에 채워진 경우 거절 1개.
- **`main.py` 세션 딕셔너리**(D11): `edit_intent.py` 범위 밖(main.py 통합 테스트)이라
  이 스프린트에선 수동 스모크로 확인하고, 자동 테스트는 Sprint 8로 미룬다 — 지금
  `main.py`엔 pytest 대상 테스트가 하나도 없어(§0 사실 밖 관찰) 새 테스트 인프라
  자체가 별도 과제다.

목표: 신규 테스트 약 26개, 기존 171개 무회귀.

---

## §8 작업 순서와 의존성

| 순서 | 작업 | 선행 |
|---|---|---|
| 1 | `llm_client.py`에 `INTENT_MODEL = "gpt-4o-mini"` 추가 | 없음 (F3) |
| 2a | `course_generator.py`에 `refinement` 파라미터(D12) · `prompt.py`에 주입 지점 | 없음 |
| 2b | `edit_intent_prompt.py` — 분류 프롬프트(reorder 2분기 지시 포함, D13) + 버전 헤더 | 1 |
| 3 | `edit_intent.py` — 스키마(D13 필드 포함) · `_unused_candidates()`(D10) · 분류기 · 실행기(insert_after·부분이동 포함) · 오케스트레이터(D9 폴백·D12 전달·D11 인터페이스) | 2a, 2b |
| 4 | `tests/test_edit_intent.py` (§7, 약 26개) | 3 |
| 5 | `main.py`에 `_session_courses` 딕셔너리(D11) + `/api/chat` 조회·저장 배선 | 3 |

2a·2b는 서로 다른 파일이라 **병렬 진행 가능**하고, 골든셋 30→60 확장도 이 작업 전체와
파일이 겹치지 않아 **병렬 진행 가능**하다. 5번(main.py)은 팀 공용 파일이라 착수 전에
채팅으로 짧게 알리는 게 안전하다 — 계약을 바꾸는 게 아니라 순수 추가(dict 하나, 조회·저장
두 줄)라 이수연 승인까지는 필요 없다고 판단했다.

---

## §9 이번 스프린트에서 다루지 않는 것

- **정식 세션 영속화**: `_session_courses`(D11)는 메모리 딕셔너리 데모용 스캐폴딩이다.
  Redis/SQLite로 정식 이관하는 건 이수연 담당(#28·`session.py`)이고, 이번 스프린트엔
  손대지 않는다 — PR #30 머지 일정(8/28)과 부딪히지 않게 격리한다.
- **`main.py`의 메인 코스 생성 경로 통합**: `main.py`가 여전히 `generate_course()`를 안
  쓰고 LangChain을 직접 호출하는 문제(#25)는 이 설계와 별개다. D11·D12·D13은 "코스가
  이미 있다고 가정하고 그걸 수정하는" 기능이라, 최초 생성 경로가 어느 쪽이든 동작한다.
  다만 `main.py`가 `edit_intent.py`를 실제로 호출하는 새 엔드포인트/분기는 이번
  스프린트에 추가한다(§8 순서 5) — #25와 달리 이건 "붙이지 않으면 시연 자체가 안
  된다"는 D11의 결론이라 범위 밖에 둘 수 없었다.
- **"더 조용한 데로" 같은 의미 기반 재선택의 품질 평가**: 골든셋 확장(별도 과제) 이후
  A/B 리포트에서 함께 측정한다.
