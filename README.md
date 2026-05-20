# 트래블러 (Traveler) — LLM·RAG 기반 초개인화 모빌리티 트래블 큐레이터

> SDV 시대의 운전자에게 "지금, 이 차 안의 맥락(날씨·기분·동승자·시간)"을 반영해 신뢰 가능한 여행 동선을 제안하는 LLM 큐레이터.  
> 환각(Hallucination)을 3단계로 차단해 "그럴듯한 거짓말"이 아닌 "실행 가능한 추천"을 보장한다.

**한이음 드림업** (과기부·IITP 주관, 2026.04 ~ 2026.10) | 멘토: 이여랑 (42dot, 현대차그룹 글로벌 SW 센터)

---

## 기술 스택

| 분류 | 사용 기술 |
|---|---|
| LLM / AI | OpenAI GPT-4o · Constraint-Aware CoT · Emotional State Inference · LLM-as-Judge (5-axis) |
| RAG | LangChain · LlamaIndex · BM25+Vector Hybrid Search (α≈0.4) · Cohere Rerank · Recursive Character Splitter |
| Validation | Pydantic v2 · 자체 비즈니스 룰 엔진 · 3-Tier Confidence Pipeline |
| DB / Storage | Pinecone (Vector+Hybrid) · SQLite (초개인화 메모리) · Redis (캐시) |
| Backend | FastAPI (async) · Circuit Breaker 패턴 |
| External APIs | 한국관광공사 TourAPI · 기상청 · Tmap · Naver Map |
| Infra | AWS EC2 / S3 / ElastiCache |

---

## 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│                       User (Driver / In-Vehicle)                    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ Query + Context (시간/날씨/동승자/기분)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  [1] Context Layer                                                  │
│    - Emotional State Inference (목적·에너지·동행 추론)              │
│    - 초개인화 Memory (SQLite, 과거 대화에서 LLM이 취향 자동 추출)   │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  [2] LLM Core — Constraint-Aware CoT (GPT-4o)                       │
│    시간 · 날씨 · 동선 · 특수상황 4대 제약을 "추론 전에" 점검        │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  [3] RAG Retrieval                                                  │
│    Hybrid (BM25 + Vector α≈0.4) → Cohere Rerank → Top-K             │
│    Source: TourAPI / 기상청 / 자체 큐레이션 코퍼스                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  [4] 3-Tier Confidence Pipeline          ← 환각 차단의 핵심         │
│    Tier1: LLM 자기 검증 (신뢰도 점수 자가 산출)                     │
│    Tier2: 외부 API 교차 검증 (Tmap/TourAPI로 사실 확인)             │
│    Tier3: 실행 가능성 확인 (영업시간·도달가능·동선 타당성)          │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  [5] 3-Stage Guardrails                                             │
│    ① Pydantic 스키마 검증  ② 비즈니스 룰  ③ Tmap API 교차 검증     │
└──────────────────────────────┬──────────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  [6] LLM-as-Judge (오프라인 평가 루프, 5축)                         │
│    조건반영도 · 감성일치 · 동선합리성 · 다양성 · 정보완결성         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 주요 구현 (이정훈 — LLM 엔지니어 담당)

### 1. Constraint-Aware CoT 프롬프트
추론 시작 전에 4대 제약(시간·날씨·동선·특수상황)을 명시적 슬롯에 채우도록 강제.  
일반 CoT 대비 "맥락 무시 환각(우천에 야외 카페 추천 등)" 케이스가 유의미하게 감소.

### 2. LLM-as-Judge 5축 채점 파이프라인
조건반영도 / 감성일치 / 동선합리성 / 다양성 / 정보완결성 — temperature=0 + JSON structured output.  
단일 점수 방식 대비 **어느 축이 약한지 분리 진단** 가능 → 프롬프트 반복 개선의 신호 채널.

### 3. 3-Tier Confidence Pipeline
- **Tier1** LLM 자기 검증 → **Tier2** 외부 API 교차 검증 → **Tier3** 실행 가능성 확인
- 통과 못 한 항목은 제거하거나 "확인 불가" 표기 (graceful degradation)

### 4. 3-Stage Guardrails
① Pydantic v2 스키마 검증 + 재프롬프트 ② 비즈니스 룰 엔진 ③ Tmap API 교차 검증

### 5. Emotional State Inference
사용자 발화에서 여행 목적 / 에너지 레벨 / 동행 유형 3차원을 LLM으로 추론.  
키워드 매칭으로는 "조용한 = 한적함 vs 안전한" 같은 맥락 의존성을 다 표현할 수 없어 LLM 추론 채택.

---

## 실행 방법

```bash
# 1. 클론
git clone https://github.com/Jeonghun-LEE-KMU/Traveler.git
cd Traveler

# 2. 의존성 (Python 3.11+)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. 환경 변수
cp .env.example .env  # OPENAI_API_KEY, COHERE_API_KEY, PINECONE_API_KEY 등 입력

# 4. 벡터 인덱스 빌드
python scripts/build_index.py

# 5. API 서버
uvicorn app.main:app --reload --port 8000

# 6. LLM-as-Judge 오프라인 평가
python eval/run_judge.py --dataset eval/cases.jsonl --out eval/report.json
```

---

## 결과 (2026.05 기준 — Sprint 4~5, MVP v1 준비중)

- Constraint-Aware CoT v3까지 반복 개선, LLM-as-Judge "조건반영도" 축 점수 상승 (정식 수치 MVP 완료 후 공개)
- 3-Tier Confidence 골격 구현 완료, Tier2 단계에서 좌표·영업정보 환각의 상당수 사전 차단
- 팀 5회 투표 무합의 → 개별 1:1 면담으로 2시간 내 해결, 멘토(42dot) 피드백으로 발표 구조 전환

---

## 상세 포트폴리오

문제 정의·실험 과정(실패 포함)·배운 점  
🔗 [Notion 포트폴리오](https://www.notion.so/35dd359e315781779da7c6709df9feb5?pvs=1)
