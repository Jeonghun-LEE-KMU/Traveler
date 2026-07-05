"""
SQLite places 테이블과 Pinecone 벡터를 동기화한다.

실제 프로젝트 위치:
  traveler/data/scripts/embed_pipeline.py

일반 동기화:
  python3 traveler/data/scripts/embed_pipeline.py

비활성 벡터(영업 종료한 식당 또는 끝난 축제 등)만 삭제:
  python3 traveler/data/scripts/embed_pipeline.py --delete-only

신규·변경 활성 장소만 임베딩:
  python3 traveler/data/scripts/embed_pipeline.py --upsert-only

관광지·문화시설만 10건 시험:
  python3 traveler/data/scripts/embed_pipeline.py \
    --upsert-only --content-types 12 14 --limit 10

경주시 문화시설·음식점 신규 데이터만 임베딩:
  python3 traveler/data/scripts/embed_pipeline.py \
    --upsert-only --content-types 14 39 --region 경주시

실행 전 확인(DB/Pinecone 변경 없음):
  python3 traveler/data/scripts/embed_pipeline.py --dry-run

Iconic Score ETL 실행 후 metadata까지 동기화:
  python3 traveler/data/scripts/embed_pipeline.py --sync-metadata

Pinecone에는 있지만 embedded_at이 비어 있는 기존 상태 복구:
  python3 traveler/data/scripts/embed_pipeline.py --reconcile-existing

기존 상태가 불확실한 비활성 벡터를 한 번 정리:
  python3 traveler/data/scripts/embed_pipeline.py --reconcile-inactive
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from traveler.rag.embed import (
    INDEX_NAME,
    build_embed_text,
    build_metadata,
    embed_texts,
)


load_dotenv()

DATA_DIR = SCRIPT_DIR.parent
DB_PATH = DATA_DIR / "places.db"

EMBED_BATCH_SIZE = 50
DELETE_BATCH_SIZE = 100
FETCH_BATCH_SIZE = 100

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def chunked(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def validate_database(conn: sqlite3.Connection) -> None:
    columns = {
        str(row[1])
        for row in conn.execute("PRAGMA table_info(places)").fetchall()
    }
    required = {
        "content_id",
        "is_active",
        "embedded_at",
        "iconic_score",
    }
    missing = required - columns
    if missing:
        raise RuntimeError(
            "places 테이블에 필요한 컬럼이 없습니다: "
            f"{sorted(missing)}"
        )


def load_pending_active_places(
    conn: sqlite3.Connection,
    limit: int | None,
    content_types: list[str] | None = None,
    region: str | None = None,
) -> list[dict]:
    sql = """
        SELECT *
        FROM places
        WHERE is_active = 1
          AND embedded_at IS NULL
    """
    params: list = []
    if content_types:
        placeholders = ", ".join("?" for _ in content_types)
        sql += f" AND content_type_id IN ({placeholders})"
        params.extend(content_types)
    if region:
        sql += " AND address LIKE ?"
        params.append(f"%{region}%")
    sql += " ORDER BY content_id"

    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)

    rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def load_inactive_ids(
    conn: sqlite3.Connection,
    reconcile_all: bool,
    limit: int | None,
    region: str | None = None,
) -> list[str]:
    # 평상시에는 Pinecone에 벡터가 있다고 기록된 비활성 장소만 삭제한다.
    # 최초 정리 시 --reconcile-inactive를 사용하면 상태값과 관계없이 확인한다.
    sql = """
        SELECT content_id
        FROM places
        WHERE is_active = 0
    """
    if not reconcile_all:
        sql += " AND embedded_at IS NOT NULL"
    params: list = []
    if region:
        sql += " AND address LIKE ?"
        params.append(f"%{region}%")
    sql += " ORDER BY content_id"

    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)

    return [
        str(row[0])
        for row in conn.execute(sql, params).fetchall()
    ]


def load_embedded_active_places(
    conn: sqlite3.Connection,
    limit: int | None,
    region: str | None = None,
) -> list[dict]:
    sql = """
        SELECT *
        FROM places
        WHERE is_active = 1
          AND embedded_at IS NOT NULL
    """
    params: list = []
    if region:
        sql += " AND address LIKE ?"
        params.append(f"%{region}%")
    sql += " ORDER BY content_id"

    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)

    rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def build_pipeline_metadata(row: dict) -> dict:
    """
    팀원 담당 build_metadata() 결과에 파이프라인 상태값을 보강한다.

    build_metadata()가 이미 같은 키를 제공하더라도 최신 SQLite 값을 사용한다.
    """
    metadata = dict(build_metadata(row) or {})
    metadata.update(
        {
            "content_id": str(row["content_id"]),
            "is_active": bool(row.get("is_active", 1)),
            "iconic_score": float(row.get("iconic_score") or 0.0),
        }
    )

    # 축제 검색·필터에 필요한 값이다. 빈 값은 Pinecone metadata에서 제외한다.
    optional_fields = [
        "content_type_id",
        "name",
        "area_code",
        "sigungu_code",
        "event_start_date",
        "event_end_date",
        "event_place",
    ]
    for field in optional_fields:
        value = row.get(field)
        if value not in (None, ""):
            metadata[field] = str(value)

    return {
        key: value
        for key, value in metadata.items()
        if value is not None
    }


def mark_embedded(
    conn: sqlite3.Connection,
    content_ids: list[str],
) -> None:
    if not content_ids:
        return
    embedded_at = utc_now_iso()
    conn.executemany(
        """
        UPDATE places
        SET embedded_at = ?
        WHERE content_id = ?
          AND is_active = 1
        """,
        [(embedded_at, content_id) for content_id in content_ids],
    )
    conn.commit()


def mark_vector_absent(
    conn: sqlite3.Connection,
    content_ids: list[str],
) -> None:
    if not content_ids:
        return
    conn.executemany(
        """
        UPDATE places
        SET embedded_at = NULL
        WHERE content_id = ?
          AND is_active = 0
        """,
        [(content_id,) for content_id in content_ids],
    )
    conn.commit()


def reconcile_existing_vectors(
    conn: sqlite3.Connection,
    index,
    *,
    limit: int | None,
    content_types: list[str] | None,
    region: str | None,
    dry_run: bool,
) -> tuple[int, int, int]:
    """
    Pinecone에 이미 존재하지만 embedded_at이 NULL인 활성 장소의 상태를 복구한다.

    과거 임베딩 코드가 Pinecone upsert 뒤 SQLite를 갱신하지 않았던 경우에만
    한 번 사용한다. Pinecone에 없는 ID는 NULL 상태로 남아 이후 재임베딩된다.
    """
    rows = load_pending_active_places(
        conn,
        limit,
        content_types,
        region,
    )
    content_ids = [str(row["content_id"]) for row in rows]
    logger.info("기존 Pinecone 벡터 확인 대상: %d건", len(content_ids))

    existing = 0
    missing = 0
    failed = 0

    for batch_ids in chunked(content_ids, FETCH_BATCH_SIZE):
        try:
            response = index.fetch(ids=batch_ids)
            existing_ids = set(response.vectors.keys())
            missing_ids = set(batch_ids) - existing_ids

            if not dry_run:
                mark_embedded(conn, sorted(existing_ids))

            existing += len(existing_ids)
            missing += len(missing_ids)
            logger.info(
                "기존 벡터 확인: 확인=%d, 존재=%d, 누락=%d",
                existing + missing,
                existing,
                missing,
            )
        except Exception:
            failed += len(batch_ids)
            logger.exception(
                "Pinecone 기존 벡터 확인 실패: ids=%s",
                batch_ids,
            )

    return existing, missing, failed


def delete_inactive_vectors(
    conn: sqlite3.Connection,
    index,
    *,
    reconcile_all: bool,
    limit: int | None,
    region: str | None,
    dry_run: bool,
) -> tuple[int, int]:
    content_ids = load_inactive_ids(
        conn,
        reconcile_all,
        limit,
        region,
    )
    logger.info("Pinecone 삭제 대상 비활성 장소: %d건", len(content_ids))

    if dry_run:
        return 0, 0

    deleted = 0
    failed = 0
    for batch_ids in chunked(content_ids, DELETE_BATCH_SIZE):
        try:
            # Pinecone delete는 존재하지 않는 ID에도 안전하게 반복 실행할 수 있다.
            index.delete(ids=batch_ids)
            mark_vector_absent(conn, batch_ids)
            deleted += len(batch_ids)
            logger.info(
                "비활성 벡터 삭제: %d/%d",
                deleted,
                len(content_ids),
            )
        except Exception:
            failed += len(batch_ids)
            logger.exception(
                "비활성 벡터 삭제 실패: ids=%s",
                batch_ids,
            )
    return deleted, failed


def upsert_pending_vectors(
    conn: sqlite3.Connection,
    openai_client: OpenAI,
    index,
    *,
    limit: int | None,
    content_types: list[str] | None,
    region: str | None,
    dry_run: bool,
) -> tuple[int, int, int]:
    rows = load_pending_active_places(
        conn,
        limit,
        content_types,
        region,
    )
    logger.info("임베딩 대상 활성 장소: %d건", len(rows))

    if dry_run:
        return 0, 0, 0

    upserted = 0
    failed = 0
    skipped = 0

    for start in range(0, len(rows), EMBED_BATCH_SIZE):
        batch_rows = rows[start : start + EMBED_BATCH_SIZE]
        valid_rows: list[dict] = []
        texts: list[str] = []

        for row in batch_rows:
            try:
                text = str(build_embed_text(row) or "").strip()
                if not text:
                    skipped += 1
                    logger.warning(
                        "임베딩 텍스트가 비어 있어 건너뜀: %s",
                        row["content_id"],
                    )
                    continue
                valid_rows.append(row)
                texts.append(text)
            except Exception:
                skipped += 1
                logger.exception(
                    "임베딩 텍스트 생성 실패: %s",
                    row["content_id"],
                )

        if not valid_rows:
            continue

        try:
            vectors = embed_texts(openai_client, texts)
            if len(vectors) != len(valid_rows):
                raise RuntimeError(
                    "임베딩 결과 수와 장소 수가 다릅니다: "
                    f"{len(vectors)} != {len(valid_rows)}"
                )

            items = [
                {
                    "id": str(row["content_id"]),
                    "values": vector,
                    "metadata": build_pipeline_metadata(row),
                }
                for row, vector in zip(valid_rows, vectors)
            ]
            index.upsert(vectors=items)

            successful_ids = [item["id"] for item in items]
            # Pinecone 성공 뒤에만 SQLite 상태를 변경한다.
            mark_embedded(conn, successful_ids)
            upserted += len(successful_ids)
            logger.info(
                "활성 벡터 upsert: %d/%d",
                upserted,
                len(rows),
            )
        except Exception:
            failed += len(valid_rows)
            logger.exception(
                "임베딩/upsert 배치 실패: ids=%s",
                [row["content_id"] for row in valid_rows],
            )

    return upserted, failed, skipped


def sync_active_metadata(
    conn: sqlite3.Connection,
    index,
    *,
    limit: int | None,
    region: str | None,
    dry_run: bool,
) -> tuple[int, int]:
    """
    활성 벡터의 metadata를 SQLite 최신값으로 갱신한다.

    벡터 values는 변경하지 않으므로 OpenAI 재임베딩 비용이 발생하지 않는다.
    Iconic Score ETL 뒤에 --sync-metadata 옵션으로 실행한다.
    """
    rows = load_embedded_active_places(conn, limit, region)
    logger.info("metadata 동기화 대상 활성 장소: %d건", len(rows))

    if dry_run:
        return 0, 0

    updated = 0
    failed = 0
    for row in rows:
        content_id = str(row["content_id"])
        try:
            index.update(
                id=content_id,
                set_metadata=build_pipeline_metadata(row),
            )
            updated += 1
            if updated % 100 == 0:
                logger.info(
                    "metadata 업데이트: %d/%d",
                    updated,
                    len(rows),
                )
        except Exception:
            failed += 1
            logger.exception(
                "metadata 업데이트 실패: %s",
                content_id,
            )

    return updated, failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SQLite places와 Pinecone 벡터 동기화"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="DB와 Pinecone을 변경하지 않고 대상 건수만 확인한다.",
    )
    parser.add_argument(
        "--sync-metadata",
        action="store_true",
        help="활성 벡터 metadata를 SQLite 최신값으로 갱신한다.",
    )
    phase_group = parser.add_mutually_exclusive_group()
    phase_group.add_argument(
        "--delete-only",
        action="store_true",
        help="비활성 장소의 Pinecone 벡터 삭제만 수행한다.",
    )
    phase_group.add_argument(
        "--upsert-only",
        action="store_true",
        help="활성·미임베딩 장소의 임베딩과 upsert만 수행한다.",
    )
    parser.add_argument(
        "--reconcile-existing",
        action="store_true",
        help=(
            "Pinecone에는 있지만 embedded_at이 NULL인 활성 장소를 찾아 "
            "SQLite 상태만 복구하고 종료한다."
        ),
    )
    parser.add_argument(
        "--reconcile-inactive",
        action="store_true",
        help=(
            "embedded_at 상태와 관계없이 모든 비활성 ID를 Pinecone에서 "
            "삭제한다. 최초 전환 때 한 번 사용한다."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="각 단계의 처리 건수를 제한한다.",
    )
    parser.add_argument(
        "--content-types",
        nargs="+",
        default=None,
        help=(
            "임베딩할 content_type_id를 제한한다. "
            "예: --content-types 12 14"
        ),
    )
    parser.add_argument(
        "--region",
        default=None,
        help=(
            "address에 포함될 지역명을 지정해 각 처리 대상을 제한한다. "
            "예: --region 경주시. 생략하면 전국 대상"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    region = args.region.strip() if args.region else None

    if not DB_PATH.exists():
        raise FileNotFoundError(f"places.db를 찾을 수 없습니다: {DB_PATH}")

    openai_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    needs_pinecone = not args.dry_run or args.reconcile_existing
    if needs_pinecone and not pinecone_key:
        raise RuntimeError(".env에 PINECONE_API_KEY가 필요합니다.")
    needs_openai = not args.dry_run and not args.delete_only
    if needs_openai and not openai_key:
        raise RuntimeError(
            ".env에 OPENAI_API_KEY가 필요합니다."
        )

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        validate_database(conn)
        logger.info(
            "동기화 범위: region=%s, content_types=%s, limit=%s",
            region or "전국",
            args.content_types or "전체",
            args.limit if args.limit is not None else "없음",
        )

        if args.reconcile_existing:
            index = Pinecone(api_key=pinecone_key).Index(INDEX_NAME)
            existing, missing, reconcile_failed = reconcile_existing_vectors(
                conn,
                index,
                limit=args.limit,
                content_types=args.content_types,
                region=region,
                dry_run=args.dry_run,
            )
            logger.info(
                (
                    "기존 벡터 상태 복구 완료: 존재=%d, 누락=%d, "
                    "실패=%d, dry_run=%s"
                ),
                existing,
                missing,
                reconcile_failed,
                args.dry_run,
            )
            if reconcile_failed:
                raise RuntimeError(
                    f"기존 벡터 확인 실패: {reconcile_failed}건"
                )
            return

        if args.dry_run:
            index = None
            openai_client = None
        else:
            index = Pinecone(api_key=pinecone_key).Index(INDEX_NAME)
            openai_client = (
                OpenAI(api_key=openai_key)
                if needs_openai
                else None
            )

        deleted = 0
        delete_failed = 0
        if not args.upsert_only:
            deleted, delete_failed = delete_inactive_vectors(
                conn,
                index,
                reconcile_all=args.reconcile_inactive,
                limit=args.limit,
                region=region,
                dry_run=args.dry_run,
            )

        upserted = 0
        upsert_failed = 0
        skipped = 0
        if not args.delete_only:
            upserted, upsert_failed, skipped = upsert_pending_vectors(
                conn,
                openai_client,
                index,
                limit=args.limit,
                content_types=args.content_types,
                region=region,
                dry_run=args.dry_run,
            )

        metadata_updated = 0
        metadata_failed = 0
        if args.sync_metadata:
            metadata_updated, metadata_failed = sync_active_metadata(
                conn,
                index,
                limit=args.limit,
                region=region,
                dry_run=args.dry_run,
            )

    logger.info(
        (
            "동기화 완료: 삭제=%d, 삭제실패=%d, "
            "upsert=%d, upsert실패=%d, 건너뜀=%d, "
            "metadata=%d, metadata실패=%d, dry_run=%s"
        ),
        deleted,
        delete_failed,
        upserted,
        upsert_failed,
        skipped,
        metadata_updated,
        metadata_failed,
        args.dry_run,
    )

    total_failed = delete_failed + upsert_failed + metadata_failed
    if total_failed:
        raise RuntimeError(
            f"일부 동기화가 실패했습니다: {total_failed}건"
        )


if __name__ == "__main__":
    main()
