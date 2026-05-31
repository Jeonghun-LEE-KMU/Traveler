# 트래블러 — 프로젝트 현황 문서

> **목적**: 팀원 누구나 이 파일 하나만 읽고 프로젝트를 이해하고 기여할 수 있도록.  
> **대상**: 신규 팀원, 복귀 팀원, 멘토, 공모전 심사 준비  
> **최종 업데이트**: 2026-06-01 (스키마 불일치 해결 — place_name/duration_min Optional) | 작성: 이정훈

---

## 한 줄 요약

**SDV 차량 안의 맥락(날씨·기분·동승자·시간)을 읽어 실행 가능한 여행 코스를 1:1로 추천하는 LLM 큐레이터.**  
"그럴듯한 거짓말(환각)"이 아닌 RAG + Guardrails로 검증된 코스를 생성한다.

---

## 공모전 & 일정 개요

| 공모전 | 상태 | 마감 |
|--------|------|------|
| **한이음 드림업 AI 공모전** | 진행 중 | 2026.10.30 (한이음 엑스포 11월) |
| **2026 관광데이터 활용 공모전 — 웹·앱 개발 부문** | ✅ 예선 통과 → 본격 개발 | 10월 기능심사 |
| **제14회 산업통상부 공공데이터 아이디어 공모전** | 진행 중 | 2026.07.06 접수 마감 |

- 본격 개발 시작일: **2026-05-22**
- 멘토: 이여랑 (42dot, 현대차그룹 글로벌 SW 센터)
- **하나의 서비스로 두 공모전 동시 제출** — 10월 동시 마감

---

## 팀 구성 & 현재 상황 (2026-05-27 기준)

### 이정훈 — LLM 엔지니어 + 팀장

**담당**: 프롬프트 설계, 외부 API 래퍼, Guardrails, LLM-as-Judge, 코스 품질 검증

| 구분 | 내용 |
|------|------|
| ✅ 완료 | Phase 1~4 (환경, 첫 LLM 호출, 프롬프트 엔지니어링, Pydantic 검증) |
| ✅ 완료 | Triple Mode SystemPrompt 3종 골격 (`prompt.py` v1.2, 2026-05-27) |
| ✅ 완료 | Router 패턴 설계 확정 (구현은 Sprint 6) |
| ✅ 완료 | **T2** `course_generator.py` 분리 — `generate_course()` main.py에서 독립 모듈로 이동 |
| ✅ 완료 | **T1** `generate_course(ctx: ContextData)` 시그니처 전환 + `build_context_message()` 추가 |
| ✅ 완료 | **T6** API key Fail Fast — `OPENAI_API_KEY` 미설정 시 시작 즉시 RuntimeError |
| ✅ 완료 | **T7+T8** 경계값 테스트 2개 추가 (총 12개) + JUDGE_MODEL 주석 정리 |
| ⏳ Sprint 1 잔여 | `traveler/api/weather.py` — 기상청 단기예보 API PoC |
| ⏳ Sprint 1 잔여 | `traveler/rag/retriever.py` — 더미 5개 반환 stub |
| ⏳ Sprint 1 잔여 | `docs/api_contract.md` — RAG 출력 ↔ LLM 입력 스키마 계약 (**D-2 마감 5/29**) |
| ⏳ Sprint 1 잔여 | Git 브랜치 `feat/jeonghun/test-schema` → PR to develop |

**현재 브랜치 상황**: `code/` 폴더가 예전 `개발/`에서 이름 바뀐 것 — Git 추적 수정 필요

---

### 이수연 — RAG · 백엔드 엔지니어

**담당**: RAG 파이프라인, FastAPI 서버 메인, LangChain, Pinecone 연동

| 구분 | 내용 |
|------|------|
| ✅ 완료 | `/api/chat` SSE 엔드포인트 골격 (공용 레포: `Traveler-map/mobility_rag`) |
| ✅ 완료 | Git 브랜치 전략 확정 (`develop` base, `feat/이름/작업명`) |
| ⏳ Sprint 1 | LangChain RetrievalQA 체인 구현 |
| ⏳ Sprint 1 | 검색 결과 → 프롬프트 컨텍스트 포매팅 로직 |
| ⏳ Sprint 1 | ConversationBufferMemory 연동 |
| ✅ 해결 | 이정훈 스키마를 이수연 포맷에 맞게 수정 — `place_name`, `duration_min` Optional 처리 (2026-06-01) |

**이수연 `/api/chat` 현재 출력 포맷 (SSE)**:
```
스트리밍 중:  data: {"bot_message": "텍스트 청크", "recommended_places": []}
마지막 청크: data: {"bot_message": "", "recommended_places": [{"place_id": 1, "visit_sequence": 1}]}
```
> ⚠️ `place_name`, `duration_min` 없음 — 이정훈 `RecommendedPlaceModel` 필수 필드와 충돌

---

### 송가영 — 데이터 · ML 엔지니어

**담당**: TourAPI 수집, Pinecone 인덱싱, SQLite 메타데이터, Iconic Score ETL

| 구분 | 내용 |
|------|------|
| ⏳ Sprint 1 | 하이브리드 검색 로직 (Pinecone 시맨틱 + SQLite 메타데이터 필터) |
| ⏳ Sprint 1 | 노키즈존·주차·휴무일 필터 쿼리 작성 |
| ⏳ Sprint 1 | Iconic Score 데이터 소스 조사 |
| ⏳ Sprint 2 | TourAPI 데이터 5,000건 확장 + Iconic Score 컬럼 추가 |

---

### 신규 멤버 (TBD) — 프론트엔드 · 백엔드 보조

**담당**: React + Naver Map 마커/Polyline, 이수연 FastAPI 보조

| 구분 | 내용 |
|------|------|
| ⏳ Sprint 1 | React 프로젝트 초기화 (Vite + TypeScript) |
| ⏳ Sprint 1 | 기본 페이지 라우팅 및 레이아웃 |

---

### PM (TBD) — 신규 합류 예정

**담당**: Sprint 운영, AWS 배포, 발표자료

> 강병헌 이탈로 공석. 합류 전까지 이정훈이 팀장 역할 병행.

---

## Sprint 1 DoD 체크리스트 (5/22 ~ 6/05, 마감 D-9)

> Sprint Review Output: "노트북에서 RAG 기반 장소 추천 데모 + SystemPrompt 3종 골격 초안 + React 프로젝트 초기 세팅"

| 항목 | 담당 | 상태 |
|------|------|------|
| Triple Mode SystemPrompt 3종 골격 | 이정훈 | ✅ 완료 (2026-05-27) |
| Few-shot 예시 3쌍 (모드 비종속) | 이정훈 | ✅ 완료 |
| course_generator.py 분리 (T2) | 이정훈 | ✅ 완료 (2026-05-27) |
| ContextData 인터페이스 전환 (T1) | 이정훈 | ✅ 완료 (2026-05-27) |
| API key Fail Fast (T6) | 이정훈 | ✅ 완료 (2026-05-27) |
| 경계값 테스트 추가 (T7) — 테스트 12개 | 이정훈 | ✅ 완료 (2026-05-27) |
| 기상청 단기예보 API PoC | 이정훈 | ⏳ 미착수 |
| RAG retriever.py 더미 stub | 이정훈 | ⏳ 미착수 |
| **api_contract.md (D-2 마감 5/29)** | 이정훈 + 이수연 | ⏳ 미착수 |
| Git 브랜치 + PR | 이정훈 | ⏳ 미착수 |
| LangChain RetrievalQA 체인 | 이수연 | ⏳ 미착수 |
| 하이브리드 검색 로직 | 송가영 | ⏳ 미착수 |
| React 초기 세팅 | 신규 TBD | ⏳ 대기 |

---

## 시스템 아키텍처

```
사용자 채팅 입력 (여행지, 일정, 취향, 이동 수단)
  │
  ├─ [Router] 여행 모드 분류 — gpt-4o-mini          ← Sprint 6 (mode_classifier.py)
  │     ├─ 명확 감지: 칩 표시("🏡 근교로 감지됨") + 자동 진행 + "변경하기" 버튼
  │     └─ 불명확:   UI 선택 카드 (🏡 근교 / ✈️ 타지 / 🚗 장거리)
  │
  ├─ 감성 파싱·취향 추출 — GPT-4o + Triple Mode System Prompt
  │     (Constraint-Aware CoT: 시간·날씨·동선·특수상황 4대 제약 선점검)
  │
  ├─ RAG 파이프라인 — Pinecone 벡터 검색 + SQLite 메타데이터 필터
  │
  ├─ 실시간 컨텍스트 — 기상청 단기예보 API + Tmap 경로 계산
  │
  ├─ 코스 생성 — GPT-4o (RAG 컨텍스트 + 날씨/교통 삽입, 모드별 장소 비율 적용)
  │
  ├─ 코스 품질 검증 (3-Stage Guardrails)
  │     ① Pydantic 스키마 검증  ② 비즈니스 룰  ③ Tmap API 교차 검증
  │
  └─ FastAPI → React + Naver Map (마커·Polyline 시각화)
```

---

## Triple Mode — 핵심 설계 결정 (2026-05-27 확정)

사용자의 여행 목적에 따라 추천 로직을 3가지로 분기한다.

| 모드 | 아이콘 | 장소 구성 비율 | 특이사항 |
|------|--------|--------------|---------|
| 근교 드라이브 | 🏡 | 숨은명소 70% · 대표명소 20% · 창의 10% | visited_spots 방문지 제외 |
| 타지 여행 | ✈️ | 대표명소 45% · 숨은명소 45% · 창의 10% | — |
| 장거리 여행 | 🚗 | 타지 목적지 로직 동일 + 경유지 안내 | bot_message에 휴게소·주유소 포함 |

> **DRY 설계**: 타지·장거리는 목적지 추천 로직 공유. `_DESTINATION_PERSONA` 한 곳만 수정하면 두 모드 동시 반영.

**UX 플로우**:
1. 사용자가 채팅 입력
2. Router(gpt-4o-mini)가 모드 분류 시도
3. 명확하면 칩 표시 후 자동 진행 / 불명확하면 선택 카드 표시
4. 코스 생성

---

## 현재 코드 파일 구조 (실제 존재하는 파일만)

```
code/
├── main.py                        # 얇은 CLI 데모 러너 (ContextData 기반, T1/T2)
├── .env.example                   # API 키 템플릿
├── requirements.txt               # Python 3.11+ 상용 의존성
├── requirements-dev.txt           # 개발 의존성 (pytest 등)
├── tests/
│   ├── test_schema.py             # Pydantic Guardrails 단위 테스트 (7개 통과, T7)
│   └── test_course_generator.py  # build_context_message() 단위 테스트 (5개 통과)
└── traveler/
    ├── core/
    │   ├── course_generator.py    # ★ 코스 생성 핵심 로직 (T1+T2, 새로 분리)
    │   ├── llm_client.py          # AsyncOpenAI 싱글턴 + Fail Fast (T6)
    │   └── prompt.py              # Triple Mode v1.3 + build_context_message() (T1)
    ├── rag/
    │   └── types.py               # 이수연↔이정훈 인터페이스 계약 (ContextData, SpotCandidate)
    └── validation/
        └── schema.py              # Pydantic v2 StreamingCourseModel (Guardrails Stage 1)
```

**아직 없는 파일 (구현 예정)**:
- `traveler/api/weather.py` — 기상청 API 래퍼 (Sprint 1)
- `traveler/rag/retriever.py` — RAG retriever stub (Sprint 1)
- `traveler/core/mode_classifier.py` — Router 패턴 (Sprint 6)
- `docs/api_contract.md` — 팀 인터페이스 계약서 (Sprint 1, 5/29 마감)

---

## 팀 인터페이스 계약 (이수연 ↔ 이정훈)

**이수연 → 이정훈**: `ContextData` 객체 전달 (`traveler/rag/types.py`)

```python
@dataclass
class ContextData:
    user_query: str          # 사용자 자연어 요청
    current_lat: float       # 현재 차량 위도
    current_lng: float       # 현재 차량 경도
    weather: str             # 기상청 날씨 ("맑음", "비", "흐림")
    candidate_spots: list[SpotCandidate]  # SQLite+Pinecone 필터 결과 (3~10개)
    duration_hours: int = 3
    transport: str = "자차"
    companion: str = "혼자"
    mood: str = ""
    route_summary: str = ""
```

**이정훈 → 프론트엔드**: `StreamingCourseModel` JSON 스트리밍

```json
{
  "bot_message": "큐레이터 멘트 (스트리밍으로 먼저 전송)",
  "recommended_places": [
    {"place_id": 1, "place_name": "첨성대", "visit_sequence": 1, "duration_min": 40}
  ]
}
```

**✅ 스키마 불일치 해결 (2026-06-01)**: `place_name`, `duration_min`을 Optional로 변경 → 이수연 포맷 그대로 수용. 백엔드가 `place_id`로 SQLite 조회해 채우는 방식으로 확정.

---

## 스트리밍 구조 — A 방식 (확정)

| 필드 | 전송 방식 | 프론트 처리 |
|------|----------|------------|
| `bot_message` | 첫 번째 필드, 글자 단위 스트리밍 | 채팅창에 실시간 타이핑 표시 |
| `recommended_places` | 두 번째 필드, JSON 완성 후 일괄 전송 | 지도 핀 일괄 표시 |

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| LLM | GPT-4o (코스 생성, Judge) · GPT-4o-mini (Router 분류, Guardrails) |
| 프롬프트 | Constraint-Aware CoT · Triple Mode Few-shot · Prompt Version Header |
| RAG | LangChain · LlamaIndex · BM25+Vector Hybrid Search · Pinecone |
| 검증 | Pydantic v2 · 비즈니스 룰 엔진 · 3-Stage Guardrails |
| DB | Pinecone (벡터) · SQLite (메타데이터) · Redis (캐시, Sprint 2) |
| 백엔드 | FastAPI (async, Sprint 5~) |
| 프론트엔드 | React + Naver Map API (마커·Polyline, Sprint 5~) |
| 외부 API | 한국관광공사 TourAPI · 기상청 단기예보 · Tmap · Naver Map |
| 인프라 | AWS EC2 / S3 / ElastiCache (Sprint 7) |

---

## MVP 마일스톤

| Sprint | 기간 | 목표 | 핵심 산출물 |
|--------|------|------|------------|
| **1** ← 현재 | 05.22~06.05 | 스펙 통일 + LLM 프롬프트 v1 + RAG 기본 인터페이스 | Triple Mode SystemPrompt ✅ · API 계약서 · 환경 세팅 |
| 2 | 06.13~06.26 | 맥락 인지 추론 체인 + 데이터 5,000건 확장 | 날씨 L1·계절 L2 CoT 체인 · TourAPI 5천건 |
| 3 | 06.27~07.10 | **MVP v1 터미널 E2E** | "경주 5시간 자차" → 코스 JSON 터미널 출력 |
| 4 | 07.11~07.24 | LLM-as-Judge 5축 + 모드별 Few-shot 보강 | 골든셋 10개 자동 평가 리포트 |
| 5 | 07.25~08.07 | FastAPI 서버 + React 기본 화면 | REST API + 정적 지도 |
| 6 | 08.08~08.21 | **MVP v2 웹+지도** + Router 패턴 구현 + 멀티턴 인텐트 5종 | 브라우저 채팅 → 지도 Polyline |
| 7 | 08.22~09.04 | AWS 배포 + 성능 튜닝 | 외부 URL, 응답 ≤3초 |
| 8 | 09.05~09.18 | **MVP v3 퍼블릭 베타** + 골든셋 50개 회귀 평가 | 퍼블릭 베타 + t-검정 리포트 |
| 9 | 09.19~10.02 | 버그 수정 + 시연 영상 → **기능심사 제출** | 제출용 패키지 |

---

## 한국관광공사 TourAPI 활용 전략 (Tier 1 — 반드시 통합)

| API | 핵심 활용 |
|-----|---------|
| KorService2 (국문 관광정보) | Pinecone corpus ETL · Naver Map 연동 · RAG fallback |
| DataLabService — 혼잡도 | 관광지 혼잡도 예측 → 시간 슬롯 배치 |
| DataLabService — 체류시간 | 평균 체류시간 → LLM 비현실적 일정 방지 |
| TarRlteTarService1 (연관 관광지) | Pinecone 시드 → 연관 50개 → GPT-4o re-ranking |
| 관광지 집중률 30일 예측 | 여행 2~4주 전 쿼리, Redis 캐시 6h |
| 반려동물 동반여행 | Pinecone namespace="pet_friendly" |
| 무장애 여행 정보 | Pinecone namespace="barrier_free" (공모전 가산점) |

---

## Git & 협업 규칙

- **공용 레포**: `Traveler-map/mobility_rag` (이수연 메인)
- **이정훈 개인**: `Jeonghun-LEE-KMU/Traveler`
- **브랜치 전략**: `develop` base → `feat/이름/작업명` → PR → develop merge
- **develop에 직접 push 금지**
- 브랜치 예시: `feat/jeonghun/test-schema`, `feat/suyeon/rag-chain`

---

## 링크 모음

| 항목 | 링크 |
|------|------|
| Notion 홈 | https://www.notion.so/35dd359e315781779da7c6709df9feb5 |
| 공용 GitHub | https://github.com/Traveler-map/mobility_rag |
| 이정훈 GitHub | https://github.com/Jeonghun-LEE-KMU/Traveler |
| 로드맵 시각화 | `info/Traveler_RoadMap_v12.html` (로컬) |
