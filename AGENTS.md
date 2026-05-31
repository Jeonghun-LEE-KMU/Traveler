# AGENTS.md — 트래블러

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

- **Notion**: https://www.notion.so/35dd359e315781779da7c6709df9feb5
- **GitHub (공용)**: https://github.com/Traveler-map — 메인 레포: `mobility_rag`
- **GitHub (정훈 개인)**: https://github.com/Jeonghun-LEE-KMU/Traveler
- **성격**: 동일 서비스로 두 공모전 동시 진행 (10월 동시 마감)
  - **한이음 드림업 AI 공모전** (2026.04~10) — 멘토: 이여랑 (42dot)
  - **2026 관광데이터 활용 공모전 — ① 웹·앱 개발 부문** (예선 통과, 10월 기능심사)
  - **제14회 산업통상부 공공데이터 활용 아이디어 공모전** (접수~7/6, 사전교육 5/20, 1차심사 7/7~10, 2차심사 7/22~7/31, 시상 8월)
- **이정훈**: LLM 엔지니어 + 팀장 (역할 병행)
- **본격 개발 시작일**: 2026-05-22

---

## 🎓 LLM 엔지니어 학습 모드 — 이 프로젝트의 대화 규칙

이 프로젝트에서 코드를 작성하거나 개념을 설명할 때 반드시 아래 규칙을 따른다.
이정훈은 LLM 파이프라인을 처음부터 직접 설계·구현하면서 LLM 엔지니어로 성장 중이다.

### 📌 설명 규칙 (모든 코드 작성 시 필수)

1. **개념 먼저, 코드는 나중에**
   - 코드를 짜기 전에 "이게 무엇인지", "왜 필요한지"를 먼저 설명한다
   - 예: `from pinecone import Pinecone` 한 줄을 쓰기 전에 "벡터 DB가 무엇이고 왜 필요한지"부터

2. **코드 한 줄 한 줄 설명**
   - 모든 함수, 변수, 파라미터에 "왜 이렇게 썼는지" 설계 의도를 주석 또는 설명으로 포함
   - "이렇게도 쓸 수 있는데 이게 더 나은 이유"까지

3. **파이프라인 내 위치 명시**
   - 지금 짜는 코드가 전체 파이프라인 중 어느 단계에 해당하는지 항상 먼저 짚는다
   - 예: "이 코드는 [사용자 입력] → **[감성 파싱]** → [RAG] → [코스 생성] 중 두 번째 단계입니다"

4. **LLM 엔지니어 관점 필수 포함**
   - 업계에서 실제로 어떻게 쓰이는지, 왜 이 패턴이 표준인지
   - 대안은 무엇이고, 이걸 선택한 이유는 무엇인지
   - 예: "temperature=0 vs 0.3 — 판단에는 0, 자연어 생성에는 0.3이 관행인 이유"

5. **설계 의사결정 사고 과정 공유**
   - 무엇을 먼저 생각해야 하는지 (입력/출력 정의 → 경계 케이스 → 에러 처리 순)
   - LLM 파이프라인 설계 시 고려해야 할 트레이드오프 명시

### 📌 빌드 순서 규칙

아래 Phase 순서를 반드시 지킨다. 앞 단계가 완전히 이해된 후 다음으로 넘어간다.

```
Phase 0: Mental Model — LLM이 뭔지, 파이프라인이 뭔지 개념 정립
Phase 1: 환경 설정 — 프로젝트 구조, .env, 패키지
Phase 2: 첫 LLM 호출 — messages 구조, 토큰, temperature
Phase 3: 프롬프트 엔지니어링 — System Prompt, CoT, Few-shot
Phase 4: Structured Output — JSON mode, Pydantic, Guardrails
Phase 5: RAG — 임베딩, Pinecone, 검색 → 주입 → 생성
Phase 6: FastAPI — async 서버, 요청/응답 모델
Phase 7: LangGraph — 상태 기반 파이프라인, 멀티턴
Phase 8: 평가 — LLM-as-Judge, 골든셋, 회귀 테스트
```

### 📌 코드 파일 작성 규칙

- 파일을 새로 만들 때: 파일이 하는 역할 1줄 요약 docstring 필수
- 함수를 새로 만들 때: "WHY 이 함수가 필요한가"를 주석으로 먼저 작성
- 외부 라이브러리 import 시: 이 라이브러리가 무엇인지, 왜 이걸 선택했는지 설명
- 설계 원칙(SRP, DRY, Graceful Degradation 등)이 적용될 때 반드시 명시

### 📌 코드 주석 형식 (모든 코드에 통일 적용)

코드를 작성할 때 아래 주석 태그를 일관되게 사용한다.

```python
# WHY: 이 함수/클래스가 파이프라인에서 존재하는 이유
# INPUT: 입력 타입 및 출처 (예: Pydantic schema, Pinecone 결과)
# OUTPUT: 출력 타입 및 전달 대상 (예: → 코스 생성 프롬프트에 주입)
# COST: ~N tokens | est. $X per 1,000 calls  ← LLM 호출 함수에 필수

async def parse_user_preference(user_input: str) -> PreferenceSchema:
    # ★ 핵심: temperature=0 — 취향 파싱은 판단이므로 결정론적 출력 필요
    response = await openai_client.chat.completions.acreate(
        model="gpt-4o-mini",      # 대안: gpt-4o → 품질 높지만 비용 3배, 여기선 불필요
        temperature=0,
        ...
    )
    # ⚠️ 주의: JSON 파싱 실패 시 전체 파이프라인 중단 — try/except 필수
    return PreferenceSchema.model_validate_json(response.choices[0].message.content)
```

| 태그 | 사용 위치 | 의미 |
|------|---------|------|
| `# ★ 핵심:` | 함수 내 가장 중요한 줄 | 이 줄이 왜 핵심인지 |
| `# ⚠️ 주의:` | 실수하기 쉬운 줄 | 무엇이 잘못될 수 있는지 |
| `# 대안:` | 다른 방법이 있는 줄 | 대안과 선택 이유 |
| `# COST:` | 모든 LLM API 호출 | 토큰 수 + 1,000회당 예상 비용 |

### 📌 모델 선택 기준

코드 작성 시 아래 기준을 따른다. 임의로 gpt-4o를 쓰지 않는다.

| 작업 유형 | 모델 | 이유 |
|---------|------|------|
| 감성 파싱·취향 추출 | `gpt-4o` | 복잡한 자연어 해석, 품질이 핵심 |
| Guardrails 1단계 (스키마 검증) | `gpt-4o-mini` | 규칙 기반, 저비용 고속 처리 |
| Guardrails 2단계 (비즈니스 규칙) | `gpt-4o-mini` | 판단 단순, mini로 충분 |
| 코스 생성 (RAG 컨텍스트 포함) | `gpt-4o` | 창의적 추론 + 긴 컨텍스트 처리 |
| 여행 모드 분류 Router (근교·타지·장거리) | `gpt-4o-mini` | 3종 분류, 저지연·저비용 필수. 스포츠카(4o)로 마트 장보지 않음 |
| 멀티턴 인텐트 분류 | `gpt-4o-mini` | 5종 분류, 저지연 필수 |
| LLM-as-Judge (5축 평가) | `gpt-4o` | 일관된 판단력, 평가 품질이 핵심 |

### 📌 프롬프트 버전 관리

모든 프롬프트 파일 상단에 버전 헤더를 붙인다. 버전 없이 프롬프트를 수정하지 않는다.

```python
# prompt_id: travel_course_generator
# version: v1.2
# updated: 2026-MM-DD
# changed: Few-shot 예시 3개 추가, 동선합리성 지시문 강화
# author: 이정훈
# tested_on: 골든셋 10개 | judge_score 평균 4.1 → 4.4
COURSE_GENERATION_PROMPT = """..."""
```

프롬프트 파일은 `code/traveler/prompts/` 디렉토리에서 관리. 인라인 하드코딩 금지.

### 📌 질문 처리 규칙

- "이게 뭐야?" → 개념 설명 + 이 프로젝트에서의 역할 + 코드 예시
- "왜 이렇게 해?" → 설계 의사결정 과정 + 대안과 비교 + 선택 이유
- "어디서 써?" → 파이프라인 내 위치 + 실제 업계 사용 사례
- "다음 뭐 해?" → 현재 Phase 확인 → 다음 단계 안내

---

## 에이전트 자동 라우팅 규칙

사용자가 질문하면 아래 기준에 따라 적합한 에이전트를 자동 판단하여 Agent 툴로 호출한다.

| 에이전트 | 모델 | 이런 질문에 자동 호출 |
|---------|------|-------------------|
| `/project:lead` | opus | 아키텍처 설계, 방향 결정, 스프린트 전략, 복잡한 기술 의사결정, 애매한 경우 |
| `/project:prompt` | sonnet | System Prompt, CoT/Few-shot, 프롬프트 디버깅·개선, 감성 파싱, JSON 후처리, 개인화 로직, LLM 응답 품질 |
| `/project:api` | sonnet | 기상청/Tmap/OpenAI/Naver Map API 연동 코드, 래퍼 모듈, 캐싱, 에러 처리, FastAPI |
| `/project:map` | sonnet | Naver Map 마커·Polyline, React 컴포넌트, 채팅-지도 동기화, 장소 카드 UI |
| `/project:qa` | sonnet | 코스 품질 검증, 엣지 케이스, LLM-as-Judge, Guardrails, 골든셋 테스트 |

**라우팅 원칙**: 단일 영역 → 해당 에이전트 1개. 복합 질문 → `/project:lead` 먼저. 코드 한 줄 수정 등 간단한 것 → 직접 처리.

**에이전트 호출 시**: 사용자 질문 + 아래 프로젝트 컨텍스트를 함께 전달. 응답은 요약 없이 그대로 전달.

**파일 저장 규칙**: Codex가 생성하는 코드 파일은 `code/` 폴더 내에 저장. 프롬프트 파일은 `code/traveler/prompts/` 디렉토리에서 관리.

---

## 폴더 구조 (2026-05-27 기준)

```
트래블러/
├── code/                          # 개발 코드 메인 (공용 GitHub: Traveler-map/mobility_rag)
│   ├── main.py                    # ★ 얇은 CLI 데모 러너 (ContextData로 전환 완료 — T1/T2)
│   │                              #   비즈니스 로직은 course_generator.py로 분리됨
│   ├── requirements.txt           # 상용 의존성
│   ├── requirements-dev.txt       # 개발 의존성
│   ├── tests/
│   │   ├── test_schema.py         # Pydantic Guardrails 단위 테스트 (7개 — T7 경계값 추가)
│   │   └── test_course_generator.py  # build_context_message() 단위 테스트 (5개)
│   └── traveler/
│       ├── api/                   # 외부 API 래퍼 (기상청, Tmap, TourAPI) — 미구현
│       ├── core/
│       │   ├── course_generator.py  # ★ 코스 생성 핵심 로직 (T1+T2 — 2026-05-27)
│       │   │                        #   generate_course(ctx: ContextData, travel_mode)
│       │   │                        #   Sprint 2 FastAPI도 여기서 import
│       │   ├── llm_client.py        # Lazy Init AsyncOpenAI + API key Fail Fast (T6)
│       │   ├── prompt.py            # Triple Mode SystemPrompt 3종 + 메시지 조립 (v1.3)
│       │   │                        #   build_context_message(ctx) — 팀 인터페이스 브릿지 ★
│       │   │                        #   build_user_message(req) — 하위 호환성용 유지
│       │   │                        #   get_system_prompt(mode) — Dispatch Table 패턴
│       │   └── mode_classifier.py   # ⏳ Sprint 6 구현 예정 — Router 패턴
│       │                            #   gpt-4o-mini로 여행 모드 자동 분류
│       │                            #   명확 → 칩 표시 + 자동 진행 / 불명확 → UI 선택 카드
│       ├── rag/
│       │   └── types.py             # 이수연↔정훈 인터페이스 계약 (ContextData, SpotCandidate)
│       └── validation/
│           └── schema.py            # Pydantic v2 StreamingCourseModel (Guardrails Stage 1)
├── info/                          # 프로젝트 정보 문서
│   ├── ROADMAP_v10.md             # Sprint 1~9 통합 일정 (SSoT)
│   ├── Traveler_RoadMap_v12.html  # 로드맵 시각화
│   ├── 한국관광공사_OpenAPI_활용전략.md
│   └── DEMO/                      # 데모 영상/시뮬레이션
├── etc/                           # 기타 자료
│   ├── Meeting/                   # 회의록 (0407 등)
│   ├── 공모전/                    # 공모전 관련 서류 (관광데이터, 공공데이터)
│   └── 사진정보/                  # 서비스 사진/아키텍처 이미지
├── AGENTS.md                      # 이 파일
└── README.md
```

### 공용 레포 구조 (Traveler-map/mobility_rag — 이수연 메인)

```
mobility_rag/                      # 공용 팀 레포 (이수연 메인 담당)
├── main.py                        # FastAPI 서버 + LangChain 스트리밍 엔드포인트
├── config.py                      # .env API 키 로드 (KORSERVICE, KMA, OPENAI 등)
├── services/
│   └── nokids.py                  # 노키즈존 서비스 (비어있음 — DB 확정 후 구현)
├── requirements.txt / requirements-dev.txt
├── start.ps1                      # 서버 기동 스크립트 (윈도우 PowerShell)
└── v.ps1                          # 가상환경 생성 스크립트 (윈도우 PowerShell)
```

**이수연 `/api/chat` 엔드포인트 출력 포맷 (SSE)**:
```
스트리밍 중: data: {"bot_message": "텍스트 청크", "recommended_places": []}
마지막 청크: data: {"bot_message": "", "recommended_places": [{"place_id": 1, "visit_sequence": 1}, ...]}
```
⚠️ `place_name`, `duration_min` 필드 없음 → 정훈 `RecommendedPlaceModel`과 스키마 불일치. 합의 필요.

### Git 브랜치 전략 (이수연 확정 — 2026-05-27)
- `develop` 브랜치가 base. **절대 develop에 직접 push 금지**
- 작업 시: 새 브랜치 생성 → push → PR → develop으로 merge
- 브랜치 네이밍: `feat/이름/작업명` (예: `feat/jeonghun/test-schema`)

---

## 프로젝트 컨텍스트

### 서비스 정의
사용자의 여행 취향·이동 수단·일정·실시간 날씨·교통을 종합 분석해 최적화된 여행 코스를 자동 생성하는 초개인화 트래블 큐레이터. RAG로 장소 정보를 실시간 검색하고, LLM이 1:1 맞춤 코스를 생성. Naver Map에 Polyline 시각화.

### 시스템 아키텍처

```
사용자 채팅 입력 (여행지, 일정, 취향, 이동 수단)
  → [Router] 여행 모드 분류 — gpt-4o-mini               ← Sprint 6 (mode_classifier.py)
       ├─ 명확 감지: 칩 표시("🏡 근교로 감지됨") + 자동 진행 + "변경하기" 버튼
       └─ 불명확:   UI 선택 카드 표시 (🏡 근교 / ✈️ 타지 / 🚗 장거리)
  → 감성 파싱·취향 추출 — GPT-4o + System Prompt (Triple Mode CoT/Few-shot)
  → RAG 파이프라인 — Pinecone 벡터 검색 + SQLite 메타데이터
  → 실시간 컨텍스트 — 기상청 단기예보 API + Tmap 경로 계산
  → 코스 생성 — GPT-4o (RAG 컨텍스트 + 날씨/교통 삽입, 모드별 장소 비율 적용)
  → 코스 품질 검증 (Evaluator) — 거리·시간·장소 유형 자동 검사
  → FastAPI → React + Naver Map (마커·Polyline 시각화)
```

### 이정훈 담당 파트
프롬프트 설계, 기상청·Tmap·TourAPI 래퍼 모듈, Naver Map React 컴포넌트, 코스 품질 검증 로직, JSON 후처리, LLM-as-Judge Evaluator 설계

### 팀 인터페이스 (2026-05-22 개편)

| 역할 | 담당 | 인터페이스 |
|------|------|-----------|
| LLM 엔지니어 (팀장) | 이정훈 | 프롬프트·외부 API·LLM-as-Judge·Guardrails·품질검증 총괄 |
| RAG · 백엔드 엔지니어 | 이수연 | RAG 파이프라인 결과 → 이정훈 프롬프트 컨텍스트 삽입, **FastAPI 서버 메인 담당** |
| 데이터 엔지니어 | 송가영 | TourAPI 수집 → Pinecone 인덱싱 + SQLite 메타데이터 → 이정훈 추론 엔진 입력 |
| 프론트엔드 · 백엔드 보조 | 신규 멤버 (TBD) | React + Naver Map 마커/Polyline, 이수연 FastAPI 보조 |
| PM / 인프라 | 신규 PM (TBD) | Sprint 운영, AWS 배포, 발표자료 (강병헌 이탈) |

### 기술 스택
- LLM: GPT-4o | Prompt: CoT, Few-shot
- RAG: LangChain, LlamaIndex | Vector DB: Pinecone | Meta DB: SQLite
- Backend: FastAPI (async) | Frontend: React
- Maps: Naver Map API | Traffic: Tmap API | Weather: 기상청 단기예보 API
- Data: 한국관광공사 TourAPI | Deploy: AWS EC2 + S3

### 한국관광공사 OpenAPI 활용 전략

> 전체 상세 문서: `프로젝트정보/한국관광공사_OpenAPI_활용전략.md`  
> 전수 조사(27개) 후 Opus가 분석, Sonnet이 문서화 (2026-05-23)

**Tier 1 — 반드시 통합 (7개)**

| API | 엔드포인트/ID | 핵심 활용 |
|-----|------------|---------|
| KorService2 (국문 관광정보) | `https://apis.data.go.kr/B551011/KorService2` | Pinecone corpus 구축 (ETL), Naver Map 연동(`locationBasedList2`), RAG fallback(`searchKeyword2`), 반려동물(`detailPetTour2`) |
| DataLabService — tarDecoList | `http://apis.data.go.kr/B551011/DataLabService` | 관광지 혼잡도 예측 → 시간 슬롯 배치 로직 (LangGraph Tool 노드) |
| DataLabService — tarTursmRqmtList | 동일 | 평균 체류시간 → LLM 비현실적 일정 방지 |
| TarRlteTarService1 (연관 관광지) | `http://apis.data.go.kr/B551011/TarRlteTarService1` | Hybrid Retrieval: Pinecone 시드 → 연관 50개 → GPT-4o re-ranking |
| 관광지 집중률 30일 예측 | data.go.kr/15128555 | 여행 계획 2~4주 전 쿼리 처리, Redis 캐시(6h TTL) |
| 반려동물 동반여행 서비스 | data.go.kr/15135102 | namespace="pet_friendly" 별도 임베딩 |
| 무장애 여행 정보 | data.go.kr/15101897 | namespace="barrier_free", 공모전 사회적 가치 가산점 |

**Tier 2 — 여유 시 통합 (8개)**  
PhotoGalleryService1(사진), DataLabService 방문자수/티맵 POI, 기초지자체 중심관광지 100위, 웰니스관광정보, 두루누비(둘레길 GPX), 오디오가이드, 고캠핑

**캐싱 3-Layer**: L1 in-memory(법정동코드, cat3 매핑) → L2 Redis(혼잡도 24h, 예측 6h) → L3 Pinecone(주 1회 Δ-sync)

---

## 핵심 설계 패턴

### Triple Mode SystemPrompt (v1.2 — 2026-05-27 완성)

사용자 여행 목적에 따라 3종 System Prompt를 선택 적용. `get_system_prompt(mode)` 함수로 Dispatch.

| 모드 | 아이콘 | 장소 구성 비율 | 특이사항 |
|------|--------|--------------|---------|
| 근교 드라이브 | 🏡 | 숨은명소 70% · 대표명소 20% · 창의 10% | visited_spots 제외 |
| 타지 여행 | ✈️ | 대표명소 45% · 숨은명소 45% · 창의 10% | — |
| 장거리 여행 | 🚗 | 타지 목적지 로직 동일 + 경유지 안내 추가 | bot_message에 휴게소·주유소 포함 |

**DRY 설계**: `_DESTINATION_PERSONA`를 타지·장거리가 공유. 목적지 추천 로직 변경 시 한 곳만 수정.  
**Graceful Degradation**: 알 수 없는 모드 → 타지로 폴백.  
**하위 호환**: `SYSTEM_PROMPT = SYSTEM_PROMPT_TOURIST` 유지.

### Router 패턴 (Sprint 6 구현 예정 — mode_classifier.py)

사용자 자연어에서 여행 모드를 자동 분류. 메인 코스 생성(gpt-4o) 전에 경량 모델(gpt-4o-mini)로 의도를 먼저 파악.

```
사용자: "근처 드라이브 코스 추천해줘"
  → gpt-4o-mini 분류 → "근교" (고확신)
  → 앱: [🏡 근교로 감지됨] [변경하기]   ← 칩 표시, 바로 코스 생성 시작

사용자: "경주 가고 싶어요"
  → gpt-4o-mini 분류 → None (불명확)
  → 앱: [🏡 근교] [✈️ 타지] [🚗 장거리]  ← UI 선택 카드
```

- **요청 먼저, 카드 나중**: 목적지 맥락이 생긴 뒤에 모드 선택 (Uber, 네이버 지도 동일 패턴)
- **투명한 자동 감지**: 틀렸을 때 탭 1번으로 변경 가능
- `travel_mode: Optional[str] = None` — Router가 채워줌 (Sprint 6). 현재는 기본값 "타지"
- `# COST: ~100 tokens · gpt-4o-mini · $0.00002/call`

### LLM-as-Judge (5축 채점)
생성된 코스를 GPT-4o가 자동 평가. 50개 골든셋 + t-검정으로 프롬프트 버전 간 회귀 방지.

| 축 | 평가 기준 |
|----|---------|
| 조건반영도 | 날씨·동행자·예산·일정 반영 정도 |
| 감성일치 | 사용자 감성 표현과 코스 분위기 일치 |
| 동선합리성 | 이동 경로·시간의 물리적 합리성 |
| 다양성 | 음식·관광·체험·휴식 유형 균형 |
| 정보완결성 | 장소명·주소·영업시간·이동시간 누락 여부 |

### Guardrails 3단계
① Pydantic CourseJSON 스키마 검증 → ② 영업시간·예산·이동시간 비즈니스 규칙 → ③ Tmap 실제 경로 교차 검증. 실패 시 해당 장소 단위 부분 재생성.

### 멀티턴 인텐트 5종
`바꿔줘 / 추가 / 빼줘 / 순서변경 / 전체재생성` — 해당 부분만 재생성해 Guardrails 재호출 범위 최소화.

---

## MVP 마일스톤 (2026-05-22 전면 개편, 통합 일정)

> 한이음(10월) + 관광데이터 기능심사(10월 초) 통합 마감. 09월 말 개발 완료 기준 9개 스프린트.

| Sprint | 기간 | 일수 | 목표 | 산출물 |
|--------|------|-----|------|-------|
| **1** | 05.22~06.05 | 15일 | **스펙 통일 + 기존 로드맵 Sprint 1~3 통합 작업** (개발 환경, 팀 API 계약, LLM 프롬프트 v1 + Guardrails + RAG 기본 인터페이스) | `requirements.txt` + `requirements-dev.txt`, 프로젝트 구조, GPT-4o 기본 코스 생성 동작 · **Triple Mode SystemPrompt 3종 골격 완성 (2026-05-27)** · **course_generator.py 분리 (T2)** · **ContextData 인터페이스 전환 (T1)** · **API key Fail Fast (T6)** · **경계값 테스트 7개 (T7)** · Router 패턴 설계 확정 (구현 Sprint 6) |
| — | 06.06~06.12 | 7일 | 🛌 시험기간 등 공백 | — |
| **2** | 06.13~06.26 | 14일 | RAG 파이프라인 본격 연결 + TourAPI 데이터 수집/임베딩 1차 완료 | Pinecone 5000건 임베딩, RAG 검색 동작 |
| **3** | 06.27~07.10 | 14일 | **MVP v1 — 터미널 E2E** (이정훈 LLM Core ↔ 이수연 RAG ↔ 송가영 DB 완전 연결) | 터미널에서 "경주 5시간 자차" → 코스 JSON 출력 |
| **4** | 07.11~07.24 | 14일 | LLM-as-Judge 5축 + 3-Tier Confidence Pipeline | 골든셋 10개 자동 평가 리포트 |
| **5** | 07.25~08.07 | 14일 | FastAPI 서버 완성 + 프론트엔드 기본 (React + Naver Map 마커) | REST API + 정적 지도 화면 |
| **6** | 08.08~08.21 | 14일 | **MVP v2 — 웹 + 지도 통합** + 멀티턴 인텐트 5종 | 브라우저 채팅 → 지도 Polyline |
| **7** | 08.22~09.04 | 14일 | AWS 배포 (EC2/S3/ElastiCache) + 성능 튜닝 + Redis 캐싱 | 외부 URL 접속, 응답 ≤3초 |
| **8** | 09.05~09.18 | 14일 | **MVP v3 — 퍼블릭 베타** + 50개 골든셋 회귀 평가 + t-검정 | 퍼블릭 베타 + 평가 리포트 |
| **9** | 09.19~10.02 | 14일 | 버그 수정 + 시연 영상 + 발표자료 → **관광데이터 1차 기능심사 제출** | 제출용 패키지 |
| — | 10.03~10월 말 | — | 🎤 한이음 + 관광데이터 PT 발표 준비 (상위 5팀 진입 시) | — |
| — | 11월 | — | 🏆 시상식 / 한이음 엑스포 Live Demo | — |
