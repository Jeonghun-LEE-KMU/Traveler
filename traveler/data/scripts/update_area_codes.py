# ============================================================
# area_code / sigungu_code 업데이트 스크립트
# 프로젝트: LLM-RAG 트래블 큐레이터
# 작성자: gayoung
# Sprint 2 | 2026-06-25
#
# 목적:
#   TourAPI 원천 데이터에서 areacode/sigungucode가 비어있는 경우,
#   address 텍스트 기반으로 area_code를 일괄 업데이트
#
# TourAPI areaCode 코드표:
#   1=서울, 2=인천, 3=대전, 4=대구, 5=광주, 6=부산, 7=울산, 8=세종
#   31=경기, 32=강원, 33=충북, 34=충남, 35=전북, 36=전남, 37=경북, 38=경남, 39=제주
# ============================================================

import os
import sqlite3

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "..")
DB_PATH  = os.path.join(DATA_DIR, "places.db")

# TourAPI areaCode 매핑 (주소 앞부분 → area_code)
AREA_CODE_MAP = [
    ("서울",  "1"),
    ("인천",  "2"),
    ("대전",  "3"),
    ("대구",  "4"),
    ("광주",  "5"),
    ("부산",  "6"),
    ("울산",  "7"),
    ("세종",  "8"),
    ("경기",  "31"),
    ("강원",  "32"),
    ("충청북도", "33"),
    ("충북",  "33"),
    ("충청남도", "34"),
    ("충남",  "34"),
    ("전라북도", "35"),
    ("전북",  "35"),
    ("전라남도", "36"),
    ("전남",  "36"),
    ("경상북도", "37"),
    ("경북",  "37"),
    ("경상남도", "38"),
    ("경남",  "38"),
    ("제주",  "39"),
]


def update_area_codes(conn: sqlite3.Connection):
    """
    area_code가 비어있는 장소를 address 텍스트 기반으로 일괄 업데이트.
    이미 area_code가 있는 장소는 건드리지 않음.
    """
    print("[area_code 업데이트] 시작...")

    # 전체 대상 건수 확인
    cur = conn.execute("SELECT COUNT(*) FROM places WHERE area_code = '' OR area_code IS NULL")
    total = cur.fetchone()[0]
    print(f"  → area_code 비어있는 장소: {total}건")

    updated = 0
    unmatched = 0

    for keyword, code in AREA_CODE_MAP:
        cur = conn.execute(
            """
            UPDATE places
            SET area_code = ?
            WHERE (area_code = '' OR area_code IS NULL)
              AND address LIKE ?
            """,
            (code, f"{keyword}%"),
        )
        count = cur.rowcount
        if count > 0:
            print(f"  [{keyword}] → area_code={code}: {count}건 업데이트")
            updated += count

    conn.commit()

    # 여전히 비어있는 건수 확인
    cur = conn.execute("SELECT COUNT(*) FROM places WHERE area_code = '' OR area_code IS NULL")
    unmatched = cur.fetchone()[0]

    print()
    print(f"=== 결과 ===")
    print(f"업데이트 완료: {updated}건")
    print(f"매칭 안 됨 (주소 형식 불명확): {unmatched}건")


def check_result(conn: sqlite3.Connection):
    """업데이트 결과 확인"""
    print()
    print("=== area_code별 건수 ===")
    cur = conn.execute(
        "SELECT area_code, COUNT(*) as cnt FROM places GROUP BY area_code ORDER BY cnt DESC"
    )
    code_names = {
        "1": "서울", "2": "인천", "3": "대전", "4": "대구",
        "5": "광주", "6": "부산", "7": "울산", "8": "세종",
        "31": "경기", "32": "강원", "33": "충북", "34": "충남",
        "35": "전북", "36": "전남", "37": "경북", "38": "경남", "39": "제주",
    }
    for row in cur.fetchall():
        code, cnt = row
        name = code_names.get(code, "미분류") if code else "미분류"
        print(f"  {name}({code}): {cnt}건")


def main():
    conn = sqlite3.connect(DB_PATH)
    update_area_codes(conn)
    check_result(conn)
    conn.close()
    print("\n[완료]")


if __name__ == "__main__":
    main()