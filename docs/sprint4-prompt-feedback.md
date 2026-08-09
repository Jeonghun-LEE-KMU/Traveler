# Sprint 4 프롬프트 피드백 반영 기록

## 목적과 검증 범위

이 문서는 v1 프롬프트와 PR11 검토에서 확인된 부정확 패턴을 Sprint 4
Constraint-Aware CoT v2에서 어떻게 보완했는지 기록한다.

현재 근거는 프롬프트 문자열의 정적 계약과 단위 테스트 결과다. 실제 LLM을
호출한 Golden 평가나 live 평가는 미실시했으며, 정확도·성공률·환각 감소율 같은
실사용 수치는 산출하지 않았다.

## 피드백과 v2 변경 매핑

| 확인된 위험 | v1/PR11 피드백 | Prompt v2 변경 | 검증 테스트 |
| --- | --- | --- | --- |
| 후보 밖 `place_id` 또는 장소 생성 | 후보 장소가 제공돼도 모델이 목록 밖 장소나 DB에 없는 ID를 만들 가능성이 있었다. 이는 SQLite enrichment 및 지도 핀 연결 실패로 이어질 수 있다. | 후보 목록이 제공된 경우 그 `place_id` 허용 목록을 절대 제약으로 지정했다. `[제약 충돌 우선순위]`와 내부 검토 첫 단계에 grounding을 배치하고, “후보 장소 목록 밖의 장소를 새로 만들지 마십시오”를 명시했다. 후보가 없는 graceful path에는 이 제한을 적용하지 않는다. `build_context_message()`의 후보 목록 제한 문장도 유지했다. | `test_sprint4_prompt_forbids_ungrounded_place_generation`, `test_build_context_message_contains_prompt_injection_attempt_within_section_and_constraint_intact` |
| 제약 충돌 우선순위 부재 | 사용자 조건, 시간·동선, 날씨·동행자, 모드별 구성 비율이 충돌할 때 무엇을 먼저 지킬지 명확하지 않았다. | `[제약 충돌 우선순위]`를 추가해 `안전 > 사용자 명시 조건 > 시간·동선 > 날씨·동행자 적합성 > 모드별 장소 구성 비율` 순으로 판단하도록 했다. | `test_sprint4_prompt_defines_constraint_conflict_priority` |
| 체류시간과 이동시간을 합친 예산 검토 누락 | 장소별 체류시간만 고려하거나 이동시간을 별도로 보면서 전체 일정이 가용 시간을 초과할 위험이 있었다. | CoT v2 내부 검토에 “총 체류시간과 이동시간의 합이 가용 시간을 넘지 않는지 확인” 단계를 추가했다. | `test_sprint4_prompt_requires_time_budget_check` |
| JSON 외 텍스트 또는 CoT 노출 | 모델이 코드 펜스, 머리말, 설명, 내부 체크리스트를 함께 출력하면 스트리밍 JSON 파싱이 깨지거나 내부 추론이 사용자 응답에 노출될 수 있었다. | 내부 추론은 검토에만 사용하고 출력하지 않도록 지시했다. 최종 출력은 `bot_message`가 첫 필드인 JSON 객체 하나로 제한하고 코드 펜스·머리말·꼬리말·추론 과정 등 JSON 외 텍스트를 금지했다. | 기존 `StreamingCourseModel` 계약 테스트와 `test_streaming_course_field_order_is_stable`, `test_coursejson_required_and_optional_place_fields_are_stable`로 구조를 검증한다. JSON-only 문구 자체는 현재 정적 프롬프트 검토로 확인했다. |

## 모드별 Few-shot 보완

공통 Few-shot 3개는 회귀 방지를 위해 모든 모드에 유지했다. 여기에 근교 3개,
장거리 2개, 타지 3개를 각각 해당 모드 프롬프트에만 주입했다. 모든 예시는
현재 `StreamingCourseModel`과 동일하게 `bot_message`를 먼저 두고,
`recommended_places`의 각 항목에 필수 `place_id`와 `visit_sequence`를 포함한다.

모드별 예시의 개수와 교차 주입 방지는 다음 단위 테스트로 검증한다.

- `test_nearby_prompt_contains_only_three_nearby_few_shots`
- `test_longdrive_prompt_contains_only_two_longdrive_few_shots`
- `test_tourist_prompt_contains_only_three_tourist_few_shots`

## 남은 검증

현재 완료된 검증은 정적 계약/단위 테스트 검증이며 live 평가는 미실시했다.
추후 실제 모델을 연결한 Golden 평가에서는 최소한 다음 항목을 별도로 측정해야
한다.

- 후보 목록 밖 `place_id` 생성 여부
- 총 체류시간과 이동시간의 가용 시간 초과 여부
- 날씨·동행자 제약 반영 여부
- JSON 파싱 성공 여부와 JSON 외 텍스트 발생 여부
- 근교·장거리·타지 모드별 추천 성격과 Few-shot 오염 여부
