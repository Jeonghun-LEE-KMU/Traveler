# ============================================================
# 임베딩 자동 업데이트 파이프라인
# 프로젝트: LLM-RAG 트래블 큐레이터
# 작성자: gayoung
# Sprint 2 | 2026-06-25
#
# 역할:
#   places.db에서 embedded_at IS NULL인 장소만 골라서
#   OpenAI text-embedding-3-small로 임베딩 후 Pinecone upsert
#   완료된 장소는 embedded_at에 타임스탬프 기록
#
# embed.py(이수연)와의 관계:
#   - embed.py: 임베딩 함수 모음 (build_embed_text, embed_texts 등)
#   - embed_pipeline.py: 자동화 실행기 (embedded_at 기반 증분 처리)
#
# 실행 방법 (mobility_rag 루트에서):
#   python3 traveler/data/scripts/embed_pipeline.py
#   python3 traveler/data/scripts/embed_pipeline.py --db traveler/data/places.db
#
# 크론잡 등록 예시 (매일 새벽 3시):
#   0 3 * * * cd /path/to/mobility_rag && venv/bin/python3 traveler/data/scripts/embed_pipeline.py
# ============================================================

import argparse
import logging
import os
import sqlite3
from datetime import datetime, timezone

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

# embed.py에서 함수 재사용 (mobility_rag 루트에서 실행 필요)
from traveler.rag.embed import (
    EMBED_BATCH,
    EMBED_MODEL,
    INDEX_NAME,
    UPSERT_BATCH,
    build_embed_text,
    build_metadata,
    embed_texts,
    get_or_create_index,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
DEFAULT_DB = os.path.join(
    os.path.dirname(__file__), "..", "places.db"
)


def load_unembedded_places(db_path: str) -> list[dict]:
    """
    embedded_at IS NULL인 장소만 로드.
    이미 Pinecone에 적재된 장소(embedded_at이 채워진 곳)는 스킵.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM places WHERE embedded_at IS NULL")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def update_embedded_at(db_path: str, content_ids: list[str]):
    """
    Pinecone upsert 완료된 장소의 embedded_at을 현재 시각으로 업데이트.
    embedded_at = NULL → 타임스탬프 (처리 완료 표시)
    """
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(db_path)
    conn.executemany(
        "UPDATE places SET embedded_at = ? WHERE content_id = ?",
        [(now, cid) for cid in content_ids],
    )
    conn.commit()
    conn.close()


def run(db_path: str):
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY가 .env에 없음!")
        return
    if not PINECONE_API_KEY:
        logger.error("PINECONE_API_KEY가 .env에 없음!")
        return
    if not os.path.exists(db_path):
        logger.error("DB 파일 없음: %s", db_path)
        return

    # 클라이언트 초기화
    oai_client = OpenAI(api_key=OPENAI_API_KEY)
    pc         = Pinecone(api_key=PINECONE_API_KEY)
    index      = get_or_create_index(pc)

    # embedded_at IS NULL인 장소만 로드
    rows = load_unembedded_places(db_path)
    logger.info("[SQLite] 미처리 장소: %d건 (embedded_at IS NULL)", len(rows))

    if not rows:
        logger.info("임베딩할 장소가 없습니다. 모두 처리 완료 상태입니다.")
        return

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

        # 2. Pinecone upsert
        upsert_items = []
        for row, vec in zip(batch_rows, vectors):
            upsert_items.append({
                "id":       row["content_id"],
                "values":   vec,
                "metadata": build_metadata(row),
            })

        succeeded_ids = []
        for j in range(0, len(upsert_items), UPSERT_BATCH):
            chunk = upsert_items[j:j + UPSERT_BATCH]
            try:
                index.upsert(vectors=chunk)
                succeeded_ids.extend([item["id"] for item in chunk])
                total_upserted += len(chunk)
            except Exception as e:
                logger.error("[Pinecone] upsert 실패 (offset=%d): %s", i + j, e)

        # 3. embedded_at 업데이트 (upsert 성공한 것만)
        if succeeded_ids:
            update_embedded_at(db_path, succeeded_ids)

        logger.info(
            "[진행] %d/%d 완료 (누적 upsert: %d)",
            min(i + EMBED_BATCH, len(valid_rows)),
            len(valid_rows),
            total_upserted,
        )

    logger.info("=" * 60)
    logger.info("임베딩 완료! 총 %d건 Pinecone upsert", total_upserted)
    logger.info("인덱스: %s | 모델: %s", INDEX_NAME, EMBED_MODEL)

    # 최종 통계
    stats = index.describe_index_stats()
    logger.info("Pinecone 현재 벡터 수: %s", stats.get("total_vector_count", "?"))

    # DB 미처리 건수 확인
    conn = sqlite3.connect(db_path)
    cur  = conn.execute("SELECT COUNT(*) FROM places WHERE embedded_at IS NULL")
    remaining = cur.fetchone()[0]
    conn.close()
    logger.info("남은 미처리 장소: %d건", remaining)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="embedded_at IS NULL인 장소만 Pinecone에 임베딩"
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB,
        help=f"SQLite DB 경로 (기본값: {DEFAULT_DB})",
    )
    args = parser.parse_args()
    run(db_path=args.db)