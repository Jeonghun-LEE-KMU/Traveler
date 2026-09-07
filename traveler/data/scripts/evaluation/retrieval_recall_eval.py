"""
고정 질의셋 기반 Retrieval Recall@K 평가 스크립트.

역할:
1. eval_queries_*.json 로드
2. hybrid_search.search_places() 실행
3. Top-K 결과에서 expected_keywords / expected_content_ids Recall@K 계산
4. query별 결과를 CSV로 저장

평가 목적:
- Search-only 평가
- Basic Search와 Personalized Search를 분리해서 측정한다.
- intent.py/preference.py가 이상적으로 구조화 입력을 추출했다고 가정하고,
  hybrid_search 자체의 검색 품질을 분리해서 본다.

평가 모드:
1. Basic Search
   - --ignore-preferences 사용
   - preferences=None 전달
   - query + region + content_types 기반 검색 품질 측정

2. Personalized Search
   - --ignore-preferences 미사용
   - query + region + content_types + preferences 기반 검색 품질 측정

content_types 우선순위:
1. CLI --content-types가 있으면 전체 질의에 강제 적용
2. 없으면 각 eval item의 content_types 사용
3. 그것도 없으면 None 전달 → hybrid_search 내부 detect_content_types fallback 사용

주의:
- keyword Recall@K는 proxy metric이다.
- category("관광", "문화", "음식")는 매칭 텍스트에서 제외한다.
  content_type만 맞아도 키워드가 맞는 것처럼 부풀려지는 문제를 줄이기 위해서다.
- overview는 HybridSearchResult에 없을 수 있으므로 places.db에서 content_id 기준으로 보강한다.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import sqlite3
import sys
from contextlib import closing
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


# 현재 파일: traveler/data/scripts/evaluation/retrieval_recall_eval.py
# parents[4] = mobility_rag/
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from traveler.rag.hybrid_search import search_places  # noqa: E402


DEFAULT_DB_PATH = REPO_ROOT / "traveler" / "data" / "places.db"


def load_eval_queries(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("eval json은 list 형태여야 합니다.")

    required_keys = {"id", "query"}

    for idx, row in enumerate(data, start=1):
        missing = required_keys - set(row.keys())
        if missing:
            raise ValueError(f"{idx}번째 query에 필수 key가 없습니다: {missing}")

        if "expected_keywords" not in row and "expected_content_ids" not in row:
            raise ValueError(
                f"{idx}번째 query에는 expected_keywords 또는 "
                "expected_content_ids 중 하나가 필요합니다."
            )

    return data


def normalize_text(value: Any) -> str:
    if value is None:
        return ""

    return str(value).lower().replace(" ", "")


def normalize_content_types(value: Any) -> list[str] | None:
    """
    content_types를 hybrid_search.py 내부 비교 기준에 맞춰 str list로 통일한다.

    허용 입력:
    - None
    - [12, 14, 39]
    - ["12", "14", "39"]
    - "12,14,39"
    """
    if value is None:
        return None

    if isinstance(value, str):
        values = [
            item.strip()
            for item in value.split(",")
            if item.strip()
        ]
        return values or None

    if isinstance(value, (list, tuple, set)):
        values = [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]
        return values or None

    raise TypeError(f"지원하지 않는 content_types 형식입니다: {type(value)}")


def resolve_content_types(
    *,
    cli_content_types: list[str] | None,
    item: dict[str, Any],
    ignore_query_content_types: bool,
) -> tuple[list[str] | None, str]:
    """
    content_types 우선순위를 결정한다.

    1. CLI --content-types가 있으면 전체 질의에 강제 적용
    2. --ignore-query-content-types가 켜져 있으면 query별 content_types를 무시
       → None 전달 → hybrid_search 내부 detect_content_types() fallback 평가
    3. eval item의 content_types 사용
    4. 없으면 None 전달 → hybrid_search 내부 fallback
    """
    if cli_content_types is not None:
        return cli_content_types, "cli"

    if ignore_query_content_types:
        return None, "fallback_detect"

    item_content_types = normalize_content_types(item.get("content_types"))
    if item_content_types is not None:
        return item_content_types, "query"

    return None, "fallback_detect"


def normalize_preferences(value: Any) -> dict[str, Any]:
    """
    평가 질의셋의 preferences를 hybrid_search.py가 기대하는 dict 구조로 정규화한다.

    권장 입력:
    {
      "companion": "커플",
      "mood": "카페,디저트,브런치",
      "travel_mode": "자차"
    }

    하위 호환:
    ["카페", "디저트"] 같은 list가 들어오면 mood 문자열로 변환한다.
    """
    if isinstance(value, dict):
        return {
            "companion": str(value.get("companion") or ""),
            "mood": str(value.get("mood") or ""),
            "travel_mode": str(value.get("travel_mode") or ""),
        }

    if isinstance(value, list):
        return {
            "companion": "",
            "mood": ",".join(str(item) for item in value),
            "travel_mode": "",
        }

    return {
        "companion": "",
        "mood": "",
        "travel_mode": "",
    }


def result_to_dict(result: Any) -> dict[str, Any]:
    """
    HybridSearchResult가 dataclass/dict/Pydantic/일반 객체 중 무엇이어도
    평가 스크립트가 최대한 동작하도록 dict로 변환한다.
    """
    if isinstance(result, dict):
        return dict(result)

    if is_dataclass(result):
        return asdict(result)

    if hasattr(result, "model_dump"):
        return result.model_dump()

    if hasattr(result, "dict"):
        return result.dict()

    data: dict[str, Any] = {}

    for name in dir(result):
        if name.startswith("_"):
            continue

        try:
            value = getattr(result, name)
        except Exception:
            continue

        if callable(value):
            continue

        data[name] = value

    return data


def pick_first(data: dict[str, Any], keys: list[str], default: Any = "") -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


def get_result_score(data: dict[str, Any]) -> float:
    score = pick_first(
        data,
        [
            "score",
            "final_score",
            "hybrid_score",
            "rerank_score",
            "similarity_score",
            "semantic_score",
        ],
        default=0.0,
    )

    try:
        return float(score)
    except (TypeError, ValueError):
        return 0.0


def get_result_content_id(data: dict[str, Any]) -> str:
    return str(
        pick_first(
            data,
            ["content_id", "place_id", "id"],
            default="",
        )
    )


def get_result_name(data: dict[str, Any]) -> str:
    return str(
        pick_first(
            data,
            ["name", "title", "place_name"],
            default="",
        )
    )


def get_result_address(data: dict[str, Any]) -> str:
    return str(
        pick_first(
            data,
            ["address", "addr1", "addr"],
            default="",
        )
    )


def get_result_overview(data: dict[str, Any]) -> str:
    return str(
        pick_first(
            data,
            ["overview", "description", "summary"],
            default="",
        )
    )


def load_place_texts_from_db(
    *,
    db_path: Path,
    content_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """
    HybridSearchResult에 overview가 없을 수 있으므로 places.db에서 보강한다.

    평가 텍스트는 name/address/overview를 중심으로 구성한다.
    category는 keyword recall 부풀림을 막기 위해 build_result_text()에서 제외한다.
    """
    unique_ids = sorted({
        str(content_id)
        for content_id in content_ids
        if str(content_id)
    })

    if not unique_ids:
        return {}

    if not db_path.exists():
        return {}

    placeholders = ", ".join("?" for _ in unique_ids)

    query = f"""
        SELECT
            content_id,
            name,
            address,
            overview,
            content_type_id
        FROM places
        WHERE content_id IN ({placeholders})
    """

    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, unique_ids).fetchall()

    return {
        str(row["content_id"]): dict(row)
        for row in rows
    }


def enrich_results_with_db_text(
    *,
    results: list[dict[str, Any]],
    db_path: Path,
) -> list[dict[str, Any]]:
    content_ids = [get_result_content_id(result) for result in results]
    db_rows = load_place_texts_from_db(db_path=db_path, content_ids=content_ids)

    enriched_results: list[dict[str, Any]] = []

    for result in results:
        content_id = get_result_content_id(result)
        db_row = db_rows.get(content_id, {})

        enriched = dict(result)

        # search result 값을 우선하되, 비어 있으면 DB 값으로 보강한다.
        if not get_result_name(enriched) and db_row.get("name"):
            enriched["name"] = db_row["name"]

        if not get_result_address(enriched) and db_row.get("address"):
            enriched["address"] = db_row["address"]

        if not get_result_overview(enriched) and db_row.get("overview"):
            enriched["overview"] = db_row["overview"]

        if not enriched.get("content_type_id") and db_row.get("content_type_id"):
            enriched["content_type_id"] = db_row["content_type_id"]

        enriched_results.append(enriched)

    return enriched_results


def build_result_text(data: dict[str, Any]) -> str:
    """
    keyword Recall 계산용 텍스트를 만든다.

    category/content_type_name은 의도적으로 제외한다.
    CATEGORY_NAMES의 "관광", "문화", "음식" 같은 값이 expected_keywords와
    직접 겹치면 content_type만 맞아도 Recall이 부풀려질 수 있기 때문이다.
    """
    parts = [
        get_result_name(data),
        get_result_address(data),
        get_result_overview(data),
    ]

    return normalize_text(" ".join(parts))


def keyword_recall_at_k(
    results: list[dict[str, Any]],
    expected_keywords: list[str],
    k: int,
) -> tuple[float | None, list[str], list[str]]:
    if not expected_keywords:
        return None, [], []

    top_results = results[:k]
    joined_text = "\n".join(build_result_text(result) for result in top_results)

    matched_keywords: list[str] = []
    missed_keywords: list[str] = []

    for keyword in expected_keywords:
        normalized_keyword = normalize_text(keyword)

        if normalized_keyword and normalized_keyword in joined_text:
            matched_keywords.append(keyword)
        else:
            missed_keywords.append(keyword)

    recall = len(matched_keywords) / len(expected_keywords)
    return recall, matched_keywords, missed_keywords


def content_id_recall_at_k(
    results: list[dict[str, Any]],
    expected_content_ids: list[str],
    k: int,
) -> tuple[float | None, list[str], list[str]]:
    """
    expected_content_ids가 있는 경우에만 계산한다.
    없으면 None 반환.
    """
    if not expected_content_ids:
        return None, [], []

    top_ids = {get_result_content_id(result) for result in results[:k]}
    expected_ids = [str(x) for x in expected_content_ids]

    matched = [content_id for content_id in expected_ids if content_id in top_ids]
    missed = [content_id for content_id in expected_ids if content_id not in top_ids]

    recall = len(matched) / len(expected_ids)
    return recall, matched, missed


def filter_by_score_threshold(
    results: list[dict[str, Any]],
    score_threshold: float | None,
) -> list[dict[str, Any]]:
    if score_threshold is None:
        return results

    return [result for result in results if get_result_score(result) >= score_threshold]


def call_search_places(
    query: str,
    region: str | None,
    top_k: int,
    content_types: list[str] | None,
    preferences: dict[str, Any] | None,
    score_weights: dict[str, float] | None,
) -> list[dict[str, Any]]:
    """
    프로젝트마다 search_places() 시그니처가 조금 달라도 최대한 맞춰 호출한다.

    예상 인자:
    - user_query 또는 query
    - region
    - content_types
    - preferences
    - visited_ids
    - top_k
    - score_weights
    """
    signature = inspect.signature(search_places)
    params = signature.parameters

    kwargs: dict[str, Any] = {}

    if "user_query" in params:
        kwargs["user_query"] = query
    elif "query" in params:
        kwargs["query"] = query
    else:
        raise TypeError("search_places()에 user_query 또는 query 인자가 없습니다.")

    if "region" in params:
        kwargs["region"] = region

    if "content_types" in params and content_types is not None:
        kwargs["content_types"] = content_types

    # Basic mode에서는 preferences=None을 그대로 넘긴다.
    # 빈 dict를 넘기면 search_places() 내부 if preferences 분기를 타서
    # 진짜 Basic이 아니게 된다.
    if "preferences" in params:
        kwargs["preferences"] = preferences

    if "visited_ids" in params:
        kwargs["visited_ids"] = set()

    if "top_k" in params:
        # threshold 후처리와 Top-K 분석을 위해 조금 넉넉히 가져온다.
        kwargs["top_k"] = max(top_k * 2, top_k)

    if "score_weights" in params and score_weights is not None:
        kwargs["score_weights"] = score_weights

    raw_results = search_places(**kwargs)

    if raw_results is None:
        return []

    return [result_to_dict(result) for result in raw_results]


def parse_score_weights(value: str | None) -> dict[str, float] | None:
    """
    CLI score weights 문자열을 dict로 변환한다.

    예:
    --score-weights semantic=0.6,keyword=0.25,preference=0.1,iconic=0.05
    """
    if not value:
        return None

    weights: dict[str, float] = {}

    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue

        if "=" not in chunk:
            raise ValueError(
                "--score-weights는 key=value 형식이어야 합니다. "
                f"잘못된 값: {chunk}"
            )

        key, raw_score = chunk.split("=", 1)
        key = key.strip()
        raw_score = raw_score.strip()

        weights[key] = float(raw_score)

    return weights or None


def format_top_results(results: list[dict[str, Any]], k: int) -> str:
    formatted: list[str] = []

    for idx, result in enumerate(results[:k], start=1):
        content_id = get_result_content_id(result)
        name = get_result_name(result)
        score = get_result_score(result)

        formatted.append(f"{idx}. {name}({content_id}) score={score:.4f}")

    return " | ".join(formatted)


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def optional_float_to_csv(value: float | None) -> str | float:
    if value is None:
        return ""
    return round(value, 4)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="고정 질의셋 기반 hybrid_search Recall@K 평가"
    )
    parser.add_argument("--eval-json", required=True, help="평가 질의셋 JSON 경로")
    parser.add_argument("--out", required=True, help="결과 CSV 저장 경로")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument(
        "--ignore-preferences",
        action="store_true",
        help="질의셋의 preferences를 무시하고 query+region 기반 Basic Search로 평가한다.",
    )
    parser.add_argument(
        "--content-types",
        type=str,
        nargs="+",
        default=None,
        help=(
            "검색 대상 content_type_id 목록. "
            "예: --content-types 12 14 39. "
            "생략하면 query별 content_types를 사용하고, 그것도 없으면 fallback."
        ),
    )
    parser.add_argument(
        "--ignore-query-content-types",
        action="store_true",
        help=(
            "eval JSON의 query별 content_types를 무시하고 "
            "content_types=None을 전달해 hybrid_search 내부 "
            "detect_content_types() fallback을 평가한다."
        ),
    )
    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB_PATH),
        help="overview 보강에 사용할 places.db 경로",
    )
    parser.add_argument(
        "--score-weights",
        default=None,
        help=(
            "score weight 조합. "
            "예: semantic=0.6,keyword=0.25,preference=0.1,iconic=0.05"
        ),
    )
    args = parser.parse_args()

    eval_path = Path(args.eval_json)
    out_path = Path(args.out)
    db_path = Path(args.db)
    cli_content_types = normalize_content_types(args.content_types)
    score_weights = parse_score_weights(args.score_weights)

    queries = load_eval_queries(eval_path)
    ensure_parent_dir(out_path)

    rows: list[dict[str, Any]] = []

    eval_mode = "basic" if args.ignore_preferences else "personalized"

    print(f"[eval] {eval_path}")
    print(f"[mode] {eval_mode}")
    print(f"[query count] {len(queries)}")
    print(f"[top_k] {args.top_k}")
    print(f"[score_threshold] {args.score_threshold}")
    print(f"[cli_content_types] {cli_content_types}")
    print(f"[ignore_query_content_types] {args.ignore_query_content_types}")
    print(f"[db] {db_path}")
    print(f"[score_weights] {score_weights}")

    for idx, item in enumerate(queries, start=1):
        query_id = str(item["id"])
        query = str(item["query"])
        region = item.get("region")

        content_types, content_types_source = resolve_content_types(
            cli_content_types=cli_content_types,
            item=item,
            ignore_query_content_types=args.ignore_query_content_types,
        )

        if args.ignore_preferences:
            preferences = None
        else:
            preferences = normalize_preferences(item.get("preferences"))

        expected_keywords = [str(x) for x in item.get("expected_keywords", [])]
        expected_content_ids = [str(x) for x in item.get("expected_content_ids", [])]
        preferences_json = (
            ""
            if preferences is None
            else json.dumps(preferences, ensure_ascii=False)
        )
        content_types_json = (
            ""
            if content_types is None
            else json.dumps(content_types, ensure_ascii=False)
        )

        print(f"\n[{idx}/{len(queries)}] {query_id} | {query}")
        print(f"  mode={eval_mode}")
        print(f"  preferences={preferences_json or 'None'}")
        print(f"  content_types={content_types_json or 'None'}")
        print(f"  content_types_source={content_types_source}")

        try:
            results = call_search_places(
                query=query,
                region=region,
                top_k=args.top_k,
                content_types=content_types,
                preferences=preferences,
                score_weights=score_weights,
            )
            results = enrich_results_with_db_text(results=results, db_path=db_path)
            results = filter_by_score_threshold(results, args.score_threshold)

            keyword_recall, matched_keywords, missed_keywords = keyword_recall_at_k(
                results=results,
                expected_keywords=expected_keywords,
                k=args.top_k,
            )

            content_id_recall, matched_ids, missed_ids = content_id_recall_at_k(
                results=results,
                expected_content_ids=expected_content_ids,
                k=args.top_k,
            )

            top_results = format_top_results(results, args.top_k)

            if keyword_recall is None:
                print(f"  keyword_recall@{args.top_k}=N/A")
            else:
                print(f"  keyword_recall@{args.top_k}={keyword_recall:.4f}")

            if content_id_recall is not None:
                print(f"  content_id_recall@{args.top_k}={content_id_recall:.4f}")

            print(f"  matched_keywords={matched_keywords}")
            print(f"  top={top_results}")

            rows.append(
                {
                    "id": query_id,
                    "query": query,
                    "region": region or "",
                    "eval_mode": eval_mode,
                    "preferences": preferences_json,
                    "content_types": content_types_json,
                    "content_types_source": content_types_source,
                    "top_k": args.top_k,
                    "score_threshold": args.score_threshold
                    if args.score_threshold is not None
                    else "",
                    "score_weights": args.score_weights or "",
                    "result_count_after_threshold": len(results),
                    "keyword_recall_at_k": optional_float_to_csv(keyword_recall),
                    "matched_keywords": "|".join(matched_keywords),
                    "missed_keywords": "|".join(missed_keywords),
                    "content_id_recall_at_k": optional_float_to_csv(content_id_recall),
                    "matched_content_ids": "|".join(matched_ids),
                    "missed_content_ids": "|".join(missed_ids),
                    "top_results": top_results,
                    "status": "ok",
                    "error_message": "",
                }
            )

        except Exception as exc:
            print(f"  [ERROR] {exc}")

            rows.append(
                {
                    "id": query_id,
                    "query": query,
                    "region": region or "",
                    "eval_mode": eval_mode,
                    "preferences": preferences_json,
                    "content_types": content_types_json,
                    "content_types_source": content_types_source,
                    "top_k": args.top_k,
                    "score_threshold": args.score_threshold
                    if args.score_threshold is not None
                    else "",
                    "score_weights": args.score_weights or "",
                    "result_count_after_threshold": 0,
                    "keyword_recall_at_k": 0.0,
                    "matched_keywords": "",
                    "missed_keywords": "|".join(expected_keywords),
                    "content_id_recall_at_k": "",
                    "matched_content_ids": "",
                    "missed_content_ids": "|".join(expected_content_ids),
                    "top_results": "",
                    "status": "error",
                    "error_message": str(exc),
                }
            )

    fieldnames = [
        "id",
        "query",
        "region",
        "eval_mode",
        "preferences",
        "content_types",
        "content_types_source",
        "top_k",
        "score_threshold",
        "score_weights",
        "result_count_after_threshold",
        "keyword_recall_at_k",
        "matched_keywords",
        "missed_keywords",
        "content_id_recall_at_k",
        "matched_content_ids",
        "missed_content_ids",
        "top_results",
        "status",
        "error_message",
    ]

    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok_rows = [row for row in rows if row["status"] == "ok"]

    keyword_rows = [
        row
        for row in ok_rows
        if row["keyword_recall_at_k"] != ""
    ]
    avg_keyword_recall = (
        sum(float(row["keyword_recall_at_k"]) for row in keyword_rows)
        / len(keyword_rows)
        if keyword_rows
        else 0.0
    )

    content_id_rows = [
        row
        for row in ok_rows
        if row["content_id_recall_at_k"] != ""
    ]
    avg_content_id_recall = (
        sum(float(row["content_id_recall_at_k"]) for row in content_id_rows)
        / len(content_id_rows)
        if content_id_rows
        else None
    )

    print("\n" + "=" * 60)
    print("[완료]")
    print(f"- output: {out_path.resolve()}")
    print(f"- mode: {eval_mode}")
    print(f"- query 수: {len(rows)}")
    print(f"- ok 수: {len(ok_rows)}")
    print(f"- 평균 keyword Recall@{args.top_k}: {avg_keyword_recall:.4f}")
    if avg_content_id_recall is not None:
        print(f"- 평균 content_id Recall@{args.top_k}: {avg_content_id_recall:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()