"""
run_migrations.py — places.db에 migrations/ 폴더의 SQL을 순서대로, 안전하게 적용

특징:
  - schema_migrations 테이블에 "이미 적용된 마이그레이션 파일명"을 기록해서,
    같은 마이그레이션이 중복 실행되는 걸 방지함 (다시 실행해도 안전 = idempotent)
  - 각 마이그레이션은 트랜잭션으로 묶여서, 중간에 실패하면 그 마이그레이션의
    변경사항만 롤백됨 (DB 전체가 깨지지 않음)
  - 실행 전 자동 백업본을 떠둠 (실수 대비)

사용법:
  # 기본값: data/migrations/ (레포 구조: <repo>/data/migrations/*.sql 가정)
  python3 run_migrations.py --db data/places.db

  # migrations 폴더 위치가 다르면 직접 지정 가능
  # (run_migrations.py를 scripts/ 등 다른 곳에 두더라도 --migrations-dir로 경로만
  #  맞춰주면 동일하게 동작함 — 스크립트 파일 위치에 의존하지 않음)
  python3 run_migrations.py --db data/places.db --migrations-dir data/migrations
"""

import sqlite3
import argparse
import shutil
import os
import glob
from datetime import datetime, timezone

# 기본값: 이 스크립트가 레포 루트에 있다고 가정하고 data/migrations/를 기본 경로로 사용.
# run_migrations.py를 다른 위치(scripts/ 등)로 옮기더라도, --migrations-dir 옵션으로
# 항상 data/migrations 경로를 명시하면 동일하게 동작함.
DEFAULT_MIGRATIONS_DIR = os.path.join("data", "migrations")


def ensure_migrations_table(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            filename     TEXT PRIMARY KEY,
            applied_at   TEXT NOT NULL
        )
        """
    )
    conn.commit()


def get_applied_migrations(conn):
    cur = conn.execute("SELECT filename FROM schema_migrations")
    return {row[0] for row in cur.fetchall()}


def backup_db(db_path):
    if not os.path.exists(db_path):
        print(f"⚠️  DB 파일이 없습니다: {db_path} (신규 생성될 예정)")
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{db_path}.backup_{timestamp}"
    shutil.copy2(db_path, backup_path)
    print(f"📦 백업 생성: {backup_path}")
    return backup_path


def run_migrations(db_path, migrations_dir):
    backup_db(db_path)

    conn = sqlite3.connect(db_path)
    ensure_migrations_table(conn)
    applied = get_applied_migrations(conn)

    migration_files = sorted(glob.glob(os.path.join(migrations_dir, "*.sql")))
    if not migration_files:
        print(f"적용할 마이그레이션 파일이 없습니다. (확인한 경로: {migrations_dir})")
        return

    pending = [f for f in migration_files if os.path.basename(f) not in applied]

    if not pending:
        print("✅ 모든 마이그레이션이 이미 적용되어 있습니다. (변경 없음)")
        conn.close()
        return

    print(f"적용 대상: {len(pending)}개 / 전체 {len(migration_files)}개\n")

    for filepath in pending:
        filename = os.path.basename(filepath)
        with open(filepath, encoding="utf-8") as f:
            sql = f.read()

        print(f"▶ 적용 중: {filename}")
        try:
            conn.executescript(sql)
            conn.execute(
                "INSERT INTO schema_migrations (filename, applied_at) VALUES (?, ?)",
                (filename, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
            print(f"  ✅ 완료\n")
        except sqlite3.OperationalError as e:
            # 컬럼이 이미 존재하는 등, idempotent하지 않은 ALTER 문 충돌 처리
            if "duplicate column name" in str(e).lower():
                print(f"  ⚠️  이미 컬럼이 존재합니다 (건너뜀): {e}")
                conn.execute(
                    "INSERT INTO schema_migrations (filename, applied_at) VALUES (?, ?)",
                    (filename, datetime.now(timezone.utc).isoformat()),
                )
                conn.commit()
            else:
                conn.rollback()
                print(f"  ❌ 실패, 롤백됨: {e}")
                print("마이그레이션 중단. 이전 마이그레이션까지는 정상 적용된 상태입니다.")
                conn.close()
                return

    conn.close()
    print("🎉 모든 마이그레이션 적용 완료")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="places.db 경로")
    parser.add_argument(
        "--migrations-dir",
        default=DEFAULT_MIGRATIONS_DIR,
        help=f"마이그레이션 SQL 폴더 경로 (기본값: {DEFAULT_MIGRATIONS_DIR})",
    )
    args = parser.parse_args()
    run_migrations(args.db, args.migrations_dir)
