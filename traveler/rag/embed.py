"""
places.db -> Pinecone 임베딩 업로드 스크립트

사용법:
    python -m traveler.rag.embed
    python -m traveler.rag.embed --db 경로/places.db
    python -m traveler.rag.embed --recreate   # 인덱스 삭제 후 재생성
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

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

EMBED_MODEL  = "text-embedding-3-small"
EMBED_DIM    = 1536
EMBED_BATCH  = 100
UPSERT_BATCH = 100

INDEX_NAME  = "traveler-places"
DEFAULT_DB  = os.path.join(os.path.dirname(__file__), "..", "data", "places.db")


def build_embed_text(row: dict) -> Optional[str]:
    name     = (row.get("name") or "").strip()
    overview = (row.get("overview") or "").strip()
    address  = (row.get("address") or "").strip()

    if overview:
        return f"{name}. {overview}" if name else overview
    elif address:
        return f"{name}. {address}" if name else address
    elif name:
        return name
    return None


def build_metadata(row: dict) -> dict:
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
        "content_id":      s(row.get("content_id")),
        "content_type_id": s(row.get("content_type_id")),
        "area_code":       s(row.get("area_code")),
        "sigungu_code":    s(row.get("sigungu_code")),
        "iconic_score":    f(row.get("iconic_score")),
        "is_pet_allowed":  i(row.get("is_pet_allowed")),
        "is_nokidszone":   i(row.get("is_nokidszone")),
        "name":            s(row.get("name")),
        "address":         s(row.get("address")),
        "category":        s(row.get("category")),
        "lat":             f(row.get("lat")),
        "lng":             f(row.get("lng")),
        "image_url":       s(row.get("image_url")),
        "tel":             s(row.get("tel")),
    }


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


def get_or_create_index(pc: Pinecone, recreate: bool = False):
    existing = [idx.name for idx in pc.list_indexes()]

    if recreate and INDEX_NAME in existing:
        logger.warning("[Pinecone] 인덱스 '%s' 삭제 중...", INDEX_NAME)
        pc.delete_index(INDEX_NAME)
        existing = []
        time.sleep(5)

    if INDEX_NAME not in existing:
        logger.info("[Pinecone] 인덱스 '%s' 생성 중 (dim=%d, metric=cosine)...", INDEX_NAME, EMBED_DIM)
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBED_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        for _ in range(30):
            time.sleep(3)
            desc = pc.describe_index(INDEX_NAME)
            if desc.status.get("ready", False):
                break
        logger.info("[Pinecone] 인덱스 생성 완료.")
    else:
        logger.info("[Pinecone] 기존 인덱스 '%s' 사용.", INDEX_NAME)

    return pc.Index(INDEX_NAME)


def run(db_path: str, recreate: bool = False):
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY가 .env에 없음!")
        sys.exit(1)
    if not PINECONE_API_KEY:
        logger.error("PINECONE_API_KEY가 .env에 없음!")
        sys.exit(1)
    if not os.path.exists(db_path):
        logger.error("DB 파일 없음: %s", db_path)
        sys.exit(1)

    oai_client = OpenAI(api_key=OPENAI_API_KEY)
    pc         = Pinecone(api_key=PINECONE_API_KEY)
    index      = get_or_create_index(pc, recreate=recreate)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur  = conn.cursor()
    cur.execute("SELECT * FROM places")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    logger.info("[SQLite] 총 %d건 로드 완료: %s", len(rows), db_path)

    valid_rows  = []
    valid_texts = []
    skipped     = 0

    for row in rows:
        text = build_embed_text(row)
        if text:
            valid_rows.append(row)
            valid_texts.append(text)
        else:
            skipped += 1

    logger.info("[필터] 임베딩 대상: %d건 / 스킵: %d건", len(valid_rows), skipped)

    total_upserted = 0

    for i in range(0, len(valid_rows), EMBED_BATCH):
        batch_rows  = valid_rows[i:i + EMBED_BATCH]
        batch_texts = valid_texts[i:i + EMBED_BATCH]

        try:
            vectors = embed_texts(oai_client, batch_texts)
        except Exception as e:
            logger.error("[OpenAI] 배치 %d~%d 임베딩 실패: %s", i, i + len(batch_rows), e)
            continue

        upsert_items = [
            {"id": row["content_id"], "values": vec, "metadata": build_metadata(row)}
            for row, vec in zip(batch_rows, vectors)
        ]

        for j in range(0, len(upsert_items), UPSERT_BATCH):
            chunk = upsert_items[j:j + UPSERT_BATCH]
            try:
                index.upsert(vectors=chunk)
                total_upserted += len(chunk)
            except Exception as e:
                logger.error("[Pinecone] upsert 실패 (offset=%d): %s", i + j, e)

        logger.info("[진행] %d/%d 완료 (누적 upsert: %d)",
                    min(i + EMBED_BATCH, len(valid_rows)), len(valid_rows), total_upserted)

        time.sleep(0.3)

    logger.info("=" * 60)
    logger.info("임베딩 완료! 총 %d건 Pinecone upsert", total_upserted)
    logger.info("인덱스: %s | 모델: %s", INDEX_NAME, EMBED_MODEL)

    stats = index.describe_index_stats()
    logger.info("Pinecone 현재 벡터 수: %s", stats.get("total_vector_count", "?"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="places.db -> Pinecone 임베딩 업로드")
    parser.add_argument("--db", default=DEFAULT_DB, help=f"SQLite DB 경로 (기본값: {DEFAULT_DB})")
    parser.add_argument("--recreate", action="store_true", help="인덱스 삭제 후 재생성")
    args = parser.parse_args()
    run(db_path=args.db, recreate=args.recreate)
