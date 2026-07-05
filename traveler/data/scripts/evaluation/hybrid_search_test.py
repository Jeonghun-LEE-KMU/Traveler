"""
hybrid_search.py 단독 품질 확인용 대화형 테스트.

Pinecone에서 의미 기반 후보를 조회하고 SQLite의 지역·콘텐츠 유형·선호도·
Iconic Score를 결합한 최종 검색 결과를 출력한다. LLM 답변 생성 테스트가
아니며 실제 서비스 통합 대상은 traveler/rag/hybrid_search.py이다.

[MVP1 경주 고정 시나리오]
Intent/Backend에서 region, content_types, travel_mode를 추출해 전달하는
연결이 아직 없으므로 이 테스트 파일에서만 경주 시나리오 값을 사용한다.

TODO(integration):
  아래 고정값은 이수연 님의 Backend/Intent 연결이 완료되면 제거하고,
  main.py가 추출한 값을 search_places()에 전달한다.

실제 프로젝트 위치:
  traveler/data/scripts/evaluation/hybrid_search_test.py

프로젝트 루트에서 실행:
  python3 traveler/data/scripts/evaluation/hybrid_search_test.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = next(
    (
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "traveler").is_dir()
    ),
    None,
)
if PROJECT_ROOT is None:
    raise RuntimeError("프로젝트 루트(traveler 폴더)를 찾을 수 없습니다.")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from traveler.rag.hybrid_search import search_places


# ============================================================
# MVP1 경주 5시간 자차 E2E 고정 테스트 설정
#
# production hybrid_search.py의 기본값이 아니다.
# Intent/Backend 연결 전 검색 인터페이스를 검증하기 위한 테스트 fixture다.
# ============================================================
TEST_REGION = "경주"
TEST_CONTENT_TYPES = ["12", "14", "39"]
TEST_PREFERENCES = {
    "companion": "친구",
    "mood": "역사, 맛집",
    "travel_mode": "근교",
}
TEST_VISITED_IDS: set[str] = set()
TEST_TOP_K = 10

DEFAULT_QUERY = (
    "친구와 경주에서 5시간 동안 자차로 여행하고 싶어. "
    "역사적인 관광지와 맛집을 포함해줘."
)


def validate_results(results: list) -> list[str]:
    """경주 고정 시나리오 검색 결과의 기본 계약을 검사한다."""
    warnings: list[str] = []
    allowed_types = set(TEST_CONTENT_TYPES)

    wrong_region = [
        result
        for result in results
        if "경주" not in result.address
    ]
    if wrong_region:
        warnings.append(
            "경주 이외 주소가 포함됨: "
            + ", ".join(result.name for result in wrong_region)
        )

    wrong_types = [
        result
        for result in results
        if str(result.content_type_id) not in allowed_types
    ]
    if wrong_types:
        warnings.append(
            "허용하지 않은 콘텐츠 유형이 포함됨: "
            + ", ".join(
                f"{result.name}(type={result.content_type_id})"
                for result in wrong_types
            )
        )

    returned_types = {
        str(result.content_type_id)
        for result in results
    }
    if not returned_types.intersection({"12", "14"}):
        warnings.append(
            "기본 시나리오가 요청한 관광지·문화시설 후보가 없습니다."
        )
    if "39" not in returned_types:
        warnings.append(
            "기본 시나리오가 요청한 음식점·카페(type=39) 후보가 없습니다."
        )

    duplicated_ids = [
        content_id
        for content_id, count in Counter(
            str(result.content_id)
            for result in results
        ).items()
        if count > 1
    ]
    if duplicated_ids:
        warnings.append(
            "중복 content_id가 포함됨: " + ", ".join(duplicated_ids)
        )

    visited_ids = {
        str(content_id)
        for content_id in TEST_VISITED_IDS
    }
    returned_visited = [
        str(result.content_id)
        for result in results
        if str(result.content_id) in visited_ids
    ]
    if returned_visited:
        warnings.append(
            "visited_ids가 검색 결과에 포함됨: "
            + ", ".join(returned_visited)
        )

    if len(results) < 3:
        warnings.append(
            f"후보가 {len(results)}건뿐임: ContextData 권장 최소 3건 미만"
        )

    return warnings


def print_results(query: str) -> None:
    results = search_places(
        user_query=query,
        region=TEST_REGION,
        content_types=TEST_CONTENT_TYPES,
        preferences=TEST_PREFERENCES,
        visited_ids=TEST_VISITED_IDS,
        top_k=TEST_TOP_K,
    )

    if not results:
        print("검색 결과가 없습니다.")
        return

    type_counts = Counter(
        str(result.content_type_id)
        for result in results
    )
    print(
        "\n[테스트 설정] "
        f"region={TEST_REGION}, "
        f"content_types={TEST_CONTENT_TYPES}, "
        f"top_k={TEST_TOP_K}"
    )
    print(f"[유형 분포] {dict(type_counts)}")

    for rank, result in enumerate(results, start=1):
        print(
            f"{rank}. {result.name} "
            f"(ID={result.content_id}, "
            f"type={result.content_type_id}, "
            f"최종={result.final_score:.4f})"
        )
        print(f"   주소: {result.address}")
        print(
            "   점수: "
            f"semantic={result.semantic_score:.3f}, "
            f"keyword={result.keyword_score:.3f}, "
            f"preference={result.preference_score:.3f}, "
            f"iconic={result.iconic_score:.3f}"
        )

    warnings = validate_results(results)
    if warnings:
        print("\n[검증 경고]")
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("\n[검증 통과] 지역·유형·ID 기본 계약을 만족했습니다.")


def main() -> None:
    print("MVP1 경주 5시간 자차 Hybrid Search 테스트")
    print("질문 없이 Enter를 누르면 기본 시나리오를 사용합니다.")
    print("종료하려면 q를 입력하세요.")

    while True:
        query = input("\n검색 질문 입력 > ").strip()
        if query.lower() in {"q", "quit", "exit"}:
            break

        query = query or DEFAULT_QUERY
        print(f"[실행 질문] {query}")

        try:
            print_results(query)
        except Exception as exc:
            print(f"[검색 실패] {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
