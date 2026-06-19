"""
Pinecone 임베딩 업로드 스크립트 — Sprint 3 준비

역할:
  가영 언니가 수집한 places.db (SQLite) → OpenAI text-embedding-3-small → Pinecone upsert

사용법:
  python -m traveler.rag.embed                          # 기본 경로 (traveler/data/places.db)
  python -m traveler.rag.embed --db 경로/to/places.db   # 커스텀 DB 경로
  python -m traveler.rag.embed --recreate               # 인덱스 삭제 후 재생성 (전체 재임베딩)

[임베딩 전략]
  - 대상 텍스트: f"{name}. {overview}"
    → 장소명을 앞에 붙이면 overview가 짧거나 없을 때도 검색 품질 보완
  - overview가 None이면 name + address 로 fallback
  - 모델: text-embedding-3-small (1536차원, cosine 유사도)

[Pinecone 인덱스]
  - 이름: traveler-places
  - 차원: 1536
  - 메트릭: cosine
  - Vector ID: content_id (TourAPI 고유 ID)

[배치 처리]
  - OpenAI embed: EMBED_BATCH=100 (API limit 고려)
  - Pinecone upsert: UPSERT_BATCH=100 (권장 최대)

[메타데이터]
  Pinecone은 메타데이터로 필터링 가능.
  → content_type_id, area_code, sigungu_code, iconic_score,
     is_pet_allowed, is_nokidszone 저장
  → retriever.py에서 이 값으로 pre-filter 후 벡터 검색
"""

import argparse
import logging
import os
import sqlite3
import sys
import time
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── 설정 ──────────────────────────────────────────────────────────────────────
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

EMBED_MODEL   = "text-embedding-3-small"
EMBED_DIM     = 1536
EMBED_BATCH   = 100   # OpenAI 한 번에 최대 요청 수
UPSERT_BATCH  = 100   # Pinecone upsert 권장 배치

INDEX_NAME    = "traveler-places"
DEFAULT_DB    = os.path.join(os.path.dirname(__file__), "..", "data", "places.db")


# ── 텍스트 조합 ───────────────────────────────────────────────────────────────

def build_embed_text(row: dict) -> Optional[str]:
    """
    임베딩할 텍스트 생성.
    overview가 있으면: "{name}. {overview}"
    없으면: "{name}. {address}" (fallback)
    둘 다 없으면: None → 임베딩 스킵
    """
    name     = (row.get("name") or "").strip()
    overview = (row.get("overview") or "").strip()
    address  = (row.get("address") or "").strip()

    if overview:
        return f"{name}. {overview}" if name else overview
    elif address:
        logger.debug("[%s] overview 없음, address로 fallback", row.get("content_id"))
        return f"{name}. {address}" if name else address
    elif name:
        logger.debug("[%s] overview/address 없음, name만 사용", row.get("content_id"))
        return name
    return None


def build_metadata(row: dict) -> dict:
    """
    Pinecone 메타데이터 빌드.
    숫자형은 float/int, 나머지는 str (None → 빈 문자열).
    Pinecone 메타데이터는 str, int, float, bool, list만 허용.
    """
    def s(v) -> str:
        return str(v) if v is not None else ""

    def f(v) -> float:
        try:
            return float(v) if v is not None else 0.0
        except (ValueError, TypeError):
            return 0.0

    def i(v) -> int:
        try:
            return int(v) if v is not None else 0
        except (ValueError, TypeError):
            return 0

    return {
        # 검색/필터링 핵심 필드
        "content_id":       s(row.get("content_id")),   # 가영 설계 명세 반영
        "content_type_id": s(row.get("content_type_id")),
        "area_code":        s(row.get("area_code")),
        "sigungu_code":     s(row.get("sigungu_code")),
        "iconic_score":     f(row.get("iconic_score")),
        "is_pet_allowed":   i(row.get("is_pet_allowed")),
        "is_nokidszone":    i(row.get("is_nokidszone")),
        # 프론트/RAG에서 표시용
        "name":             s(row.get("name")),
        "address":          s(row.get("address")),
        "category":         s(row.get("category")),
        "lat":              f(row.get("lat")),
        "lng":              f(row.get("lng")),
        "image_url":        s(row.get("image_url")),
        "tel":              s(row.get("tel")),
    }


# ── OpenAI 임베딩 ─────────────────────────────────────────────────────────────

def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """
    OpenAI text-embedding-3-small 배치 임베딩.
    빈 문자열은 API가 거부하므로 미리 필터링해서 넣어야 함.
    """
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )
    # response.data는 index 순서 보장됨
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


# ── Pinecone 인덱스 초기화 ────────────────────────────────────────────────────

def get_or_create_index(pc: Pinecone, recreate: bool = False):
    """
    traveler-places 인덱스 가져오기 또는 생성.
    recreate=True면 기존 인덱스 삭제 후 재생성 (전체 재임베딩 시).
    """
    existing = [idx.name for idx in pc.list_indexes()]

    if recreate and INDEX_NAME in existing:
        logger.warning("[Pinecone] 인덱스 '%s' 삭제 중...", INDEX_NAME)
        pc.delete_index(INDEX_NAME)
        existing = []
        time.sleep(5)  # 삭제 완료 대기

    if INDEX_NAME not in existing:
        logger.info("[Pinecone] 인덱스 '%s' 생성 중 (dim=%d, metric=cosine)...",
                    INDEX_NAME, EMBED_DIM)
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # 인덱스 ready 대기
        for _ in range(30):
            time.sleep(3)
            desc = pc.describe_index(INDEX_NAME)
            if desc.status.get("ready", False):
                break
        logger.info("[Pinecone] 인덱스 생성 완료.")
    else:
        logger.info("[Pinecone] 기존 인덱스 '%s' 사용.", INDEX_NAME)

    return pc.Index(INDEX_NAME)


# ── 메인 ──────────────────────────────────────────────────────────────────────

def run(db_path: str, recreate: bool = False):
    # 환경변수 확인
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY가 .env에 없음!")
        sys.exit(1)
    if not PINECONE_API_KEY:
        logger.error("PINECONE_API_KEY가 .env에 없음!")
        sys.exit(1)

    # DB 경로 확인
    if not os.path.exists(db_path):
        logger.error("DB 파일 없음: %s", db_path)
        sys.exit(1)

    # 클라이언트 초기화
    oai_client = OpenAI(api_key=OPENAI_API_KEY)
    pc         = Pinecone(api_key=PINECONE_API_KEY)
    index      = get_or_create_index(pc, recreate=recreate)

    # SQLite 전체 읽기
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur  = conn.cursor()
    cur.execute("SELECT * FROM places")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    logger.info("[SQLite] 총 %d건 로드 완료: %s", len(rows), db_path)

    # 임베딩할 텍스트 준비
    valid_rows  = []
    valid_texts = []
    skipped     = 0

    for row in rows:
        text = build_embed_text(row)
        if text:
            valid_rows.append(row)
            valid_texts.append(text)
        else:
            logger.debug("[스킵] content_id=%s: 임베딩할 텍스트 없음", row.get("content_id"))
            skipped += 1

    logger.info("[필터] 임베딩 대상: %d건 / 스킵: %d건", len(valid_rows), skipped)

    # 배치 처리
    total_upserted = 0

    for i in range(0, len(valid_rows), EMBED_BATCH):
        batch_rows  = valid_rows[i:i + EMBED_BATCH]
        batch_texts = valid_texts[i:i + EMBED_BATCH]

        # 1. OpenAI 임베딩
        try:
            vectors = embed_texts(oai_client, batch_texts)
        except Exception as e:
            logger.error("[OpenAI] 배치 %d~%d 임베딩 실패: %s", i, i + len(batch_rows), e)
            continue

        # 2. Pinecone upsert 준비
        upsert_items = []
        for row, vec in zip(batch_rows, vectors):
            upsert_items.append({
                "id":       row["content_id"],
                "values":   vec,
                "metadata": build_metadata(row),
            })

        # 3. Pinecone upsert (upsert도 배치 단위로)
        for j in range(0, len(upsert_items), UPSERT_BATCH):
            chunk = upsert_items[j:j + UPSERT_BATCH]
            try:
                index.upsert(vectors=chunk)
                total_upserted += len(chunk)
            except Exception as e:
                logger.error("[Pinecone] upsert 실패 (offset=%d): %s", i + j, e)

        logger.info(
            "[진행] %d/%d 완료 (누적 upsert: %d)",
            min(i + EMBED_BATCH, len(valid_rows)),
            len(valid_rows),
            total_upserted,
        )

        # OpenAI rate limit 대비 짧은 슬립 (RPM 여유 있으면 제거 가능)
        time.sleep(0.3)

    logger.info("=" * 60)
    logger.info("임베딩 완료! 총 %d건 Pinecone upsert", total_upserted)
    logger.info("인덱스: %s | 모델: %s", INDEX_NAME, EMBED_MODEL)

    # 최종 통계 출력
    stats = index.describe_index_stats()
    logger.info("Pinecone 현재 벡터 수: %s", stats.get("total_vector_count", "?"))


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="places.db → Pinecone 임베딩 업로드 스크립트"
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB,
        help=f"SQLite DB 경로 (기본값: {DEFAULT_DB})",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Pinecone 인덱스를 삭제하고 재생성 (전체 재임베딩)",
    )
    args = parser.parse_args()
    run(db_path=args.db, recreate=args.recreate)
