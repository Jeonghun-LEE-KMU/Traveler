import logging
import time

from traveler.rag.retriever import search_places


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def timed_call(label: str) -> None:
    query = "경주에서 역사 유적지 중심으로 여행하고 싶어"
    prefs = {
        "companion": "",
        "mood": "역사,전통,문화",
        "travel_mode": "",
    }

    start = time.perf_counter()
    results = search_places(
        user_query=query,
        lat=35.8562,
        lon=129.2247,
        visited=set(),
        prefs=prefs,
        top_k=10,
        radius_km=30,
    )
    elapsed = time.perf_counter() - start

    print(f"[{label}] elapsed={elapsed:.3f}s")
    for idx, result in enumerate(results, start=1):
        print(f"{idx}. {result.place_name} / {result.category} / {result.distance_min}분")


def main() -> None:
    timed_call("1st retriever call / cache miss")
    timed_call("2nd retriever call / cache hit")


if __name__ == "__main__":
    main()