# CourseJSON 계약

CourseJSON은 LLM 코스 생성 결과와 백엔드·지도 UI 사이의 최소 데이터 계약이다.
런타임의 기준 구현은 `traveler.validation.schema.StreamingCourseModel`이다.

## 최상위 구조와 필드 순서

```json
{
  "bot_message": "사용자에게 먼저 보여줄 안내 문장",
  "recommended_places": [
    {
      "place_id": 17,
      "visit_sequence": 1,
      "place_name": "예시 장소",
      "duration_min": 60
    }
  ]
}
```

`bot_message`는 반드시 첫 번째 필드이고 `recommended_places`는 두 번째 필드다.
이 순서는 단순한 표현 규칙이 아니라, bot_message 텍스트를 먼저 스트리밍하고
장소 배열이 완성된 뒤 지도 핀을 갱신하기 위한 프론트엔드 계약이다.

## 장소 필드

- `place_id`: 필수 정수이며 1 이상이다. SQLite 장소 PK와 연결된다.
- `visit_sequence`: 필수 정수이며 1부터 시작한다.
- `place_name`: 선택 문자열이다. 없으면 백엔드가 place_id로 보강한다.
- `duration_min`: 선택 정수다. 없으면 백엔드 체류시간 데이터로 보강할 수 있다.

`recommended_places`에는 장소가 1~10개 있어야 한다. 빈 배열이나 11개 이상은
스키마 검증 오류다.

## 정규화와 검증

- `duration_min`이 있으면 10~480분 범위로 clamp한다. 예를 들어 5는 10으로,
  600은 480으로 정규화한다.
- 장소 배열은 `visit_sequence` 기준으로 정렬한다.
- 정렬 후 순서는 1, 2, 3처럼 연속이어야 한다. 중복 또는 누락된 순번은
  검증 오류다.
- `place_name`이 제공됐다면 공백 문자열일 수 없다.

## 실패와 fallback

LLM 응답이 JSON 파싱에 실패하거나 위 스키마를 통과하지 못하거나 API 호출이
실패하면 코스 생성기는 예외를 사용자에게 전파하지 않고 fallback
`StreamingCourseModel`을 반환한다. 후보 장소가 있으면 첫 후보의 place_id를,
없으면 안전 기본값 1을 사용하고 `visit_sequence=1`로 구성한다. 이 동작은
서비스 중단을 피하기 위한 Graceful Degradation이며, 잘못된 원본 응답을
정상 데이터로 간주한다는 뜻은 아니다.

## LLM, 백엔드, 지도 enrichment 경계

LLM의 책임은 후보 목록에 grounding된 `place_id`, 방문 순서, 사용자용 문장을
생성하는 것이다. 좌표를 만들거나 존재하지 않는 장소 ID를 추측하지 않는다.

백엔드는 place_id를 기준으로 SQLite/TourAPI에서 `place_name`, 위도·경도,
필요한 경우 `duration_min`을 enrichment한다. 지도/프론트엔드는 백엔드가
검증하고 보강한 좌표와 장소 정보를 표시하며, LLM 원문을 좌표 데이터로 직접
사용하지 않는다. 이 경계는 장소명·좌표 환각이 지도 핀으로 전파되는 것을 막는다.
