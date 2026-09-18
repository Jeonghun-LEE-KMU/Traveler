#!/usr/bin/env python3
"""
run_pg_migrations.py — PostgreSQL(RDS)에 migrations/ 폴더의 SQL을 순서대로, 안전하게 적용

사용법:
  # DATABASE_URL 환경변수 사용 (컨테이너 안에서)
  python3 traveler/data/migrations/run_pg_migrations.py

  # DSN 직접 지정
  python3 traveler/data/migrations/run_pg_migrations.py --dsn "postgresql://user:pass@host/db"

  # migrations 폴더 경로 지정 (기본값: 이 스크립트와 같은 폴더)
  python3 traveler/data/migrations/run_pg_migrations.py --migrations-dir traveler/data/migrations

SQLite 전용인 run_migrations.py 와 별도 운영.
멱등성: schema_migrations 테이블에 적용 이력을 기록해 중복 실행 방지.
"""

import argparse
import glob
import os
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras


def _ensure_migrations_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                filename   TEXT PRIMARY KEY,
                applied_at TEXT NOT NULL
            )
        """)
    conn.commit()


def _get_applied(conn) -> set:
    with conn.cursor() as cur:
        cur.execute("SELECT filename FROM schema_migrations")
        return {row[0] for row in cur.fetchall()}


def _split_statements(sql: str) -> list[str]:
    """SQL 텍스트를 개별 구문으로 분리 (주석 제거 후 세미콜론 기준)."""
    lines = []
    for line in sql.splitlines():
        stripped = line.strip()
        if stripped.startswith("--"):
            continue
        lines.append(line)
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]


def run_migrations(dsn: str, migrations_dir: str) -> None:
    conn = psycopg2.connect(dsn, cursor_factory=psycopg2.extras.RealDictCursor)
    _ensure_migrations_table(conn)
    applied = _get_applied(conn)

    migration_files = sorted(glob.glob(os.path.join(migrations_dir, "*.sql")))
    if not migration_files:
        print(f"적용할 SQL 파일 없음 (경로: {migrations_dir})")
        conn.close()
        return

    pending = [f for f in migration_files if os.path.basename(f) not in applied]

    if not pending:
        print("✅ 모든 마이그레이션이 이미 적용되어 있습니다.")
        conn.close()
        return

    print(f"적용 대상: {len(pending)}개 / 전체 {len(migration_files)}개\n")

    for filepath in pending:
        filename = os.path.basename(filepath)
        with open(filepath, encoding="utf-8") as f:
            sql = f.read()

        print(f"▶ 적용 중: {filename}")
        try:
            statements = _split_statements(sql)
            with conn.cursor() as cur:
                for stmt in statements:
                    cur.execute(stmt)
                cur.execute(
                    "INSERT INTO schema_migrations (filename, applied_at) VALUES (%s, %s)",
                    (filename, datetime.now(timezone.utc).isoformat()),
                )
            conn.commit()
            print(f"  ✅ 완료\n")
        except Exception as e:
            conn.rollback()
            print(f"  ❌ 실패, 롤백됨: {e}")
            print("마이그레이션 중단. 이전 마이그레이션까지는 정상 적용된 상태.")
            conn.close()
            return

    conn.close()
    print("🎉 모든 PostgreSQL 마이그레이션 적용 완료")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PostgreSQL 마이그레이션 실행기")
    parser.add_argument(
        "--dsn",
        default=os.getenv("DATABASE_URL"),
        help="PostgreSQL DSN (기본값: DATABASE_URL 환경변수)",
    )
    parser.add_argument(
        "--migrations-dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="마이그레이션 SQL 폴더 경로 (기본값: 이 스크립트와 같은 폴더)",
    )
    args = parser.parse_args()

    if not args.dsn:
        parser.error("--dsn 또는 DATABASE_URL 환경변수가 필요합니다.")

    run_migrations(args.dsn, args.migrations_dir)
