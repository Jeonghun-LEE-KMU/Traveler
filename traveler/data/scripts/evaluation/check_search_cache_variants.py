from __future__ import annotations

import logging
import time

from traveler.rag.retriever import search_places

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

TEST_CASES = [
    {
        "label": "busan_ocean_walk",
        "query": "부산에서 바다 보면서 산책하기 좋은 곳 추천해줘",
        # 부산 해운대 근처
        "lat": 35.1587,
        "lon": 129.1604,
        "prefs": {
            "companion": "",
            "mood": "바다,산책,자연,여유",
            "travel_mode": "",
        },
    },
    {
        "label": "gangneung_coffee_ocean",
        "query": "강릉에서 커피 마시고 바다 볼 수 있는 곳 추천해줘",
        # 강릉 안목해변 근처
        "lat": 37.7713,
        "lon": 128.9487,
        "prefs": {
            "companion": "",
            "mood": "카페,커피,바다,여유",
            "travel_mode": "",
        },
    },
    {
        "label": "gapyeong_drive_healing",
        "query": "가평으로 드라이브 가서 조용하게 쉬기 좋은 곳 알려줘",
        # 가평 아침고요수목원 근처
        "lat": 37.7430,
        "lon": 127.3527,
        "prefs": {
            "companion": "",
            "mood": "조용,힐링,자연,뷰",
            "travel_mode": "근교",
        },
    },
]

# 같은 의도인데 문장만 다른 경우 테스트
INTENT_VARIANTS = [
    {
        "label": "busan_ocean_original",
        "query": "부산에서 바다 보면서 산책하기 좋은 곳 추천해줘",
    },
    {
        "label": "busan_ocean_variant_1",
        "query": "부산 바닷가 산책 코스 추천해줘",
    },
    {
        "label": "busan_ocean_variant_2",
        "query": "부산에서 해변 근처로 여유롭게 걸을 만한 곳 알려줘",
    },
    {
        "label": "busan_ocean_variant_3",
        "query": "부산 바다 보이는 산책 명소 가고 싶어",
    },
]


def run_search(label: str, query: str, lat: float, lon: float, prefs: dict) -> float:
    start = time.perf_counter()

    results = search_places(
        user_query=query,
        lat=lat,
        lon=lon,
        visited=set(),
        prefs=prefs,
        top_k=10,
        radius_km=30,
    )

    elapsed = time.perf_counter() - start
    top_names = [result.place_name for result in results[:3]]

    print(f"{label}: elapsed={elapsed:.3f}s | top3={top_names}")
    return elapsed


def test_multiple_queries() -> None:
    print("\n[TEST 1] 다양한 유저 쿼리 cache miss/hit 확인")

    for case in TEST_CASES:
        print(f"\n--- {case['label']} ---")

        run_search(
            label="1st call",
            query=case["query"],
            lat=case["lat"],
            lon=case["lon"],
            prefs=case["prefs"],
        )

        run_search(
            label="2nd call",
            query=case["query"],
            lat=case["lat"],
            lon=case["lon"],
            prefs=case["prefs"],
        )


def test_same_intent_different_phrasing() -> None:
    print("\n[TEST 2] 같은 의도지만 다른 문장 구성 cache 동작 확인")

    prefs = {
        "companion": "",
        "mood": "바다,산책,자연,여유",
        "travel_mode": "",
    }

    for variant in INTENT_VARIANTS:
        print(f"\n--- {variant['label']} ---")

        run_search(
            label="1st call",
            query=variant["query"],
            lat=35.1587,
            lon=129.1604,
            prefs=prefs,
        )

        run_search(
            label="2nd call",
            query=variant["query"],
            lat=35.1587,
            lon=129.1604,
            prefs=prefs,
        )


def main() -> None:
    test_multiple_queries()
    test_same_intent_different_phrasing()


if __name__ == "__main__":
    main()