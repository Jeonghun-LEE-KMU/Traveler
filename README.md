# 트래블러 — LLM 여행 코스 추천 서비스

날씨, 동행자, 가용 시간 같은 조건을 자연어로 받아 여행 코스를 만드는 GPT-4o 기반 서비스입니다.
2026 한이음 드림업 팀 프로젝트(2026.04 – 2026.10)로 진행했습니다. 저는 팀장으로 LLM 코어와 출력 검증, 평가 파이프라인을 맡았습니다.

## 이 저장소

팀 저장소 `Traveler-map/mobility_rag`가 비공개라서 제 작업을 보여 드리기 위해 개인 저장소로 옮긴 사본입니다.

- **포함**: 팀 저장소 `develop` 브랜치의 2026-10-29 시점 커밋 105개. 커밋 작성자 기록은 원본 그대로입니다.
- **제외**: 팀원 기능 브랜치, 프론트엔드 저장소(`Traveler-map/traveler-front`), API 키와 관광 데이터 DB, 평가 실행 리포트
- **추가**: 이 README 작성, IDE 설정 파일 정리. 원래 팀 README는 [`docs/TEAM_README.md`](docs/TEAM_README.md)에 그대로 두었습니다.
- 보관용 사본이라 팀 저장소와 동기화하지 않습니다.
- 팀원들의 동의를 받아 공개했습니다.

## 서비스 흐름과 담당

```
사용자 요청 (/api/chat, SSE 스트리밍)
  → ① 조건 파싱: 이동 모드, 동행자 의도, 가용 시간, 계절        이정훈
  → ② 후보 장소 검색: 하이브리드 검색, 순위 결합(RRF)            gayoung · thatsgoodd · 이정훈
  → ③ 코스 생성: 모드별 프롬프트 + GPT-4o                          이정훈
  → ④ 검증: 스키마, 이동·체류 시간, 대체 응답                      이정훈
  → 응답 스트리밍 → 프론트엔드                                     ekgus040
  ↺ 멀티턴 수정 (/api/chat/revise): 바꿔줘·추가·빼줘·순서 변경·재생성   이정훈 · 수연
```

| 팀원 (GitHub 계정) | 주 담당 |
|---|---|
| **이정훈 (팀장)** | LLM 코어(`traveler/core/`), 출력 검증(`traveler/validation/`), 평가 파이프라인(`eval/`), 테스트(`tests/`) |
| gayoung | 관광 데이터 수집·전처리(`traveler/data/`), 임베딩·검색(`traveler/rag/`) |
| thatsgoodd | 외부 API(`traveler/api/`, Tmap 경로·장소 검색), 하이브리드 검색 |
| 수연 | API 서버(`main.py`), 코스 수정 응답 개선, 운영시간·휴무 정보 반영 |
| ekgus040 | 프론트엔드(`traveler-front`), 지역 좌표 |

팀원별 작업은 `git log --author="<계정>"`이나 `git blame`으로 확인할 수 있습니다.

## 제가 맡은 부분

### 1. LLM 코어 (`traveler/core/`)
- 이동 모드(근교·장거리·타지)별 시스템 프롬프트와 소수 예시(few-shot) 설계 (`prompt.py`)
- 코스 생성 흐름과 LLM 호출 (`course_generator.py`, `llm_client.py`)
- 동행자 의도, 가용 시간, 계절 정보 파싱 (`intent.py`, `persona.py`, `season_context.py`)
- 멀티턴 수정 의도 5종 분류와 수정 프롬프트 (`edit_intent.py`)
- 응답을 끊김 없이 보내는 SSE 텍스트 스트리밍 (`stream_text.py`)

### 2. 출력 검증 (`traveler/validation/`, `traveler/core/course_schedule.py`)
- Pydantic 스키마 검증과 실패 시 대체 응답
- 장소 유형별 체류 시간표와 이동·체류 시간 기반 코스 검증
- 사용자 입력을 프롬프트 안에서 구획으로 분리하고 인젝션 문장 테스트 추가
- 날씨 등 외부 정보가 없거나 조회에 실패해도 코스 생성이 멈추지 않도록 처리

### 3. 평가 파이프라인 (`eval/`)
- **평가 세트 60건**: 근교·장거리·타지 각 20건 (`eval/cases.jsonl`)
- **구조 검사**: JSON 스키마 통과, 대체 응답 아님, 장소 수 충족, 후보에 없는 장소 생성 없음. 이 검사를 통과한 코스만 채점합니다 (`eval/metrics.py`).
- **LLM-as-Judge 5축 채점**: 조건 반영, 분위기 일치, 동선, 다양성, 완결성 (0–5점, `eval/judge.py`)
- **회귀 판정**: 모드별 5축 평균이 기준 버전보다 0.3점 이상 떨어지면 FAIL. 기준 버전이 없거나 채점 표본이 0건이면 PASS로 넘기지 않고 INCONCLUSIVE로 따로 표시합니다 (`eval/run_judge.py`).
- 재현성을 위해 평가 실행은 temperature 0으로 고정했습니다.

**첫 정식 평가 (2026-10-07)**: 60건 중 56건이 구조 검사를 통과했습니다(93.3%). 기준 버전을 처음 만든 실행이라 회귀 비교는 없습니다. 실행 리포트는 저장소에 포함하지 않았습니다.

### 4. 서비스 오류 수정
- 프롬프트 템플릿의 f-string 충돌로 서버가 멈추던 문제 수정
- 검증·대체 응답 경로를 하나로 합치고 사용자 화면에 원시 JSON이 노출되던 문제 제거
- 코스 장소 수 하한(3곳)과 생성 temperature(0.3) 고정

## 기술 스택

Python, FastAPI, OpenAI API (GPT-4o), LangChain, Pinecone, PostgreSQL, Pydantic, pytest

## 폴더 구조

```
main.py          FastAPI 서버, SSE 스트리밍            수연 · 이정훈
config.py        환경 변수 로드
traveler/
  core/          코스 생성, 프롬프트, 의도 파싱          이정훈
  validation/    응답 스키마 검증                       이정훈
  rag/           임베딩, 하이브리드 검색                 gayoung · thatsgoodd · 이정훈
  api/           외부 API 래퍼                          thatsgoodd · 이정훈
  data/          관광 데이터 수집·전처리                 gayoung
eval/            평가 파이프라인                        이정훈
tests/           단위 테스트                            이정훈
docs/            설계 문서, 원래 팀 README
```

## 실행

서버를 실제로 띄우려면 PostgreSQL, Pinecone 인덱스, 관광 데이터 적재, 각종 API 키가 필요합니다. 이 저장소에는 키와 데이터가 없어 코드만으로는 바로 실행되지 않습니다.

```bash
pip install -r requirements.txt
# config.py의 변수 이름을 참고해 .env 작성 (OpenAI, Pinecone, 공공데이터포털, Tmap 등)
uvicorn main:app --reload

pytest                                                                       # 단위 테스트
python eval/run_judge.py --dataset eval/cases.jsonl --out eval/report.json   # 평가 (OpenAI API 호출)
```
