# 트래블러 — LLM 여행 코스 추천 서비스

날씨·동행자·가용 시간 같은 조건을 자연어로 받아 여행 코스를 만드는 GPT-4o 기반 서비스입니다.
2026 한이음 드림업 팀 프로젝트(2026.04 – 2026.10)이며, 이 저장소는 팀 저장소 `Traveler-map/mobility_rag`의 `develop` 브랜치를 커밋 이력째 옮겨 정리한 개인 포트폴리오용 사본입니다.

## 역할

| 담당 | 내용 |
|---|---|
| **이정훈 (팀장)** | 출력 검증과 평가 체계, LLM 코어 안정화 |
| 팀원 | 관광 데이터 수집·전처리, 검색(RAG), 지도·경로 API, 백엔드 최적화, 프론트엔드 |

커밋 작성자 기록은 원본 그대로 유지했습니다. 팀원별 작업은 `git log --author`로 확인할 수 있습니다.

## 본인이 만든 것

### 1. 평가 파이프라인 (`eval/`)
- **평가 세트 60건**: 근교·장거리·타지 각 20건 (`eval/cases.jsonl`)
- **구조 검사**: JSON 스키마 통과, 대체 응답 아님, 장소 수 충족, 후보에 없는 장소 생성 없음 (`eval/metrics.py`). 이 검사를 통과한 코스만 채점
- **LLM-as-Judge 5축 채점**: 조건 반영, 분위기 일치, 동선, 다양성, 완결성 (0–5점, `eval/judge.py`)
- **회귀 판정**: 모드별 5축 평균이 기준 버전보다 0.3점 이상 떨어지면 FAIL. 기준 버전이 없거나 채점 표본이 0건이면 PASS로 묻지 않고 INCONCLUSIVE로 분리 (`eval/run_judge.py`)
- 재현성을 위해 평가 실행은 temperature 0으로 고정

**첫 정식 평가 (2026-10-07)**: 60건 중 56건이 구조 검사 통과 (93.3%). 기준 버전을 처음 만든 실행이라 회귀 비교는 없습니다.

### 2. 출력 검증 (`traveler/validation/`, `traveler/core/`)
- Pydantic 스키마 검증과 실패 시 대체 응답
- 가용 시간 파싱, 장소 유형별 체류 시간표, 이동·체류 시간 기반 코스 검증
- 사용자 입력을 프롬프트 안에서 구획 표시로 분리하고 인젝션 문장 테스트 추가
- 날씨 등 외부 정보가 없거나 실패해도 LLM 코어가 멈추지 않도록 처리

### 3. 서비스 오류 수정
- 프롬프트 템플릿 f-string 충돌로 서버가 멈추던 문제 수정
- 검증·대체 응답 경로를 하나로 합치고, 사용자 화면에 원시 JSON이 노출되던 문제 제거
- 코스 장소 수 하한 3곳, 생성 temperature 0.3 고정

## 기술 스택

Python, FastAPI, OpenAI API (GPT-4o), LangChain, Pinecone, PostgreSQL, Pydantic, pytest

## 구조

```
main.py          FastAPI 진입점 (SSE 스트리밍)
config.py        환경 변수 로드
traveler/
  core/          코스 생성, 프롬프트, 의도·시간 파싱
  rag/           하이브리드 검색 (팀원)
  api/           외부 API 래퍼 (팀원)
  data/          관광 데이터 수집·전처리 (팀원)
  validation/    응답 스키마 검증
eval/            평가 파이프라인
tests/           단위 테스트
docs/            설계 문서, 원 팀 README (TEAM_README.md)
```

## 실행

```bash
pip install -r requirements.txt
# config.py의 변수 이름을 참고해 .env 작성
uvicorn main:app --reload

pytest                                                    # 단위 테스트
python eval/run_judge.py --dataset eval/cases.jsonl --out eval/report.json   # 평가 (OpenAI API 호출)
```

API 키(OpenAI, Pinecone, 공공데이터포털, Tmap 등)와 관광 데이터 DB는 저장소에 포함하지 않습니다.
