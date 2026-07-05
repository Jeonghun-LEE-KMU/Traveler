# MVP1 기준 정훈 담당 작업 범위

## 결론

정훈 담당 범위는 `main.py` 전체 연결이 아니라, LLM 코어 쪽의 Triple Mode 정책을 코드와 테스트로 고정하는 것이다.

현재 `travel_mode`와 3종 SystemPrompt 라우팅은 `traveler/core/course_generator.py`, `traveler/core/prompt.py`에 이미 일부 구현되어 있다. 다만 `main.py`가 아직 `generate_course()`를 호출하지 않기 때문에 실제 FastAPI E2E 경로에는 붙어 있지 않다.

따라서 정훈이 먼저 끝내야 할 일은 아래 3가지다.

1. `MODE_WEIGHTS` 상수 추가
2. `MODE_WEIGHTS`와 `get_system_prompt()` 단위 테스트 추가
3. 터미널 smoke test에서 근교 모드 실행 가능하게 정리

## 수정해야 할 파일

### 1. `traveler/core/prompt.py`

`MODE_WEIGHTS` 상수를 추가한다.

현재 근교/타지/장거리 추천 비율은 `MODE_WEIGHTS` 한 곳에 두고, 프롬프트 문자열은 이 값을 주입받도록 만든다.

```text
근교: 숨은 명소 70%, 대표 명소 20%, 창의 추천 10%
타지: 대표 명소 45%, 숨은 명소 45%, 창의 추천 10%
장거리: 대표 명소 45%, 숨은 명소 45%, 창의 추천 10%
```

이를 코드 상수로 분리한다.

```python
MODE_WEIGHTS = {
    "근교": {
        "hidden": 0.70,
        "iconic": 0.20,
        "creative": 0.10,
    },
    "타지": {
        "iconic": 0.45,
        "hidden": 0.45,
        "creative": 0.10,
    },
    "장거리": {
        "iconic": 0.45,
        "hidden": 0.45,
        "creative": 0.10,
    },
}
```

### 왜 필요한가

프롬프트 안의 비율은 사람이 읽을 수는 있지만 테스트가 검증하기 어렵다. `MODE_WEIGHTS`처럼 구조화된 상수로 빼면 LLM 정책을 코드로 검증할 수 있다.

LLM 엔지니어 관점에서는 이것을 **prompt policy를 structured config로 분리하고 실제 prompt rendering에 연결하는 작업**으로 볼 수 있다. 프롬프트 품질 기준을 문자열에만 숨겨두지 않고, 코드와 테스트가 확인 가능한 형태로 만드는 것이다.

## 2. `tests/test_prompt.py`

새 테스트 파일을 추가한다.

검증할 내용은 다음과 같다.

1. `MODE_WEIGHTS`에 `"근교"`, `"타지"`, `"장거리"`가 모두 있는지
2. 각 모드의 가중치 합이 `1.0`인지
3. `get_system_prompt("근교")`가 근교 프롬프트를 반환하는지
4. `get_system_prompt("장거리")`가 장거리 경유지 안내를 포함하는지
5. `get_system_prompt("이상한값")`이 타지 프롬프트로 fallback 되는지

예상 테스트 이름:

비율 테스트는 단순히 `"70%"`가 어딘가에 있는지만 보지 않고, `숨은 명소`, `대표 명소`, `창의적 추천` 라벨별 비율 라인이 `MODE_WEIGHTS`와 일치하는지 확인한다.

```python
def test_mode_weights_include_all_travel_modes():
    ...

def test_mode_weights_sum_to_one():
    ...

def test_get_system_prompt_returns_nearby_prompt():
    ...

def test_get_system_prompt_returns_longdrive_prompt():
    ...

def test_get_system_prompt_unknown_mode_falls_back_to_tourist():
    ...
```

특히 장거리 프롬프트는 타지와 같은 템플릿을 공유하더라도 `MODE_WEIGHTS["장거리"]`를 주입받아야 한다.
그래야 나중에 장거리 비율만 바뀌어도 프롬프트와 테스트가 함께 따라간다.

## 3. `scripts/smoke_test.py`

선택 작업이지만 MVP1 내부 테스트를 위해 수정하는 것을 권장한다.

현재는 기본값으로 `generate_course(ctx)`를 호출한다. `generate_course()`의 기본 `travel_mode`는 `"타지"`이므로, PM이 말한 `"경주 5시간 자차"` 근교 모드 테스트를 하려면 아래처럼 명시한다.

```python
course = await generate_course(ctx, travel_mode="근교")
```

이렇게 하면 터미널에서 정훈 LLM 코어 파이프라인을 근교 모드로 직접 확인할 수 있다.

현재 로컬 환경에서는 `python`이 miniforge 3.12를 가리키고 의존성이 설치되어 있으므로, 실행 명령은 `python scripts/smoke_test.py`를 기준으로 둔다. `python3`는 시스템 Python 3.13이라 `pydantic`이 없어 실패할 수 있다.

## 정훈이 건드리지 않아도 되는 파일

아래 작업은 현재 정훈 담당 범위가 아니다.

```text
main.py
traveler/rag/retriever.py 실제 Pinecone 연결
traveler/data/scripts/*
Iconic Score v2
FastAPI travel_mode 파라미터 연결
```

`main.py` 연결은 MVP1 전체 병목이지만, 백엔드 orchestration 작업에 가깝다. 정훈은 `generate_course(ctx, travel_mode)`가 안정적으로 동작하고 테스트로 보장되게 만드는 데 집중하면 된다.

## PM에게 답장 예시

```text
확인했습니다. 큰 방향 동의합니다.

코드 기준으로 travel_mode와 3종 SystemPrompt 라우팅은 course_generator/prompt 쪽에는 이미 들어가 있지만,
main.py 엔드포인트가 아직 generate_course()를 호출하지 않아서 실제 E2E 경로에는 붙어 있지 않은 상태로 보입니다.

저는 우선 MODE_WEIGHTS 상수와 관련 단위 테스트,
그리고 get_system_prompt/travel_mode 라우팅 테스트를 보강하겠습니다.
그 다음 main.py에서 ContextData → generate_course(ctx, travel_mode)로 연결되는 흐름에 맞춰 같이 붙이면 될 것 같습니다.
```

## 완료 기준

정훈 담당 작업은 아래 조건을 만족하면 완료로 본다.

```text
1. traveler/core/prompt.py에 MODE_WEIGHTS 추가
2. tests/test_prompt.py 추가
3. MODE_WEIGHTS 가중치 합 테스트 통과
4. get_system_prompt() 모드 라우팅 테스트 통과
5. 근교/타지/장거리 프롬프트가 각자의 MODE_WEIGHTS 비율을 실제로 반영
6. 전체 pytest 통과
7. scripts/smoke_test.py에서 근교 모드 실행 가능
```
