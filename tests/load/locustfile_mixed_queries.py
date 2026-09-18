from __future__ import annotations

from locust import HttpUser, between, task


class MixedQueryUser(HttpUser):
    """
    B. Mixed query 부하테스트.

    여러 지역/의도의 query를 섞어 호출해
    LLM 호출 없이 검색 캐시 miss/hit 혼합 상태의 응답 시간을 확인한다.
    """

    wait_time = between(1, 2)

    @task
    def busan_ocean_walk(self) -> None:
        self.client.get(
            "/api/dev/search-cache",
            params={
                "user_message": "부산에서 바다 보면서 산책하기 좋은 곳 추천해줘",
                "lat": 35.1587,
                "lon": 129.1604,
                "mode": "nearby",
            },
            name="B mixed queries /api/dev/search-cache busan_ocean_walk",
            timeout=30,
        )

    @task
    def gangneung_coffee_ocean(self) -> None:
        self.client.get(
            "/api/dev/search-cache",
            params={
                "user_message": "강릉에서 커피 마시고 바다 볼 수 있는 곳 추천해줘",
                "lat": 37.7713,
                "lon": 128.9487,
                "mode": "nearby",
            },
            name="B mixed queries /api/dev/search-cache gangneung_coffee_ocean",
            timeout=30,
        )

    @task
    def gapyeong_drive_healing(self) -> None:
        self.client.get(
            "/api/dev/search-cache",
            params={
                "user_message": "가평으로 드라이브 가서 조용하게 쉬기 좋은 곳 알려줘",
                "lat": 37.7430,
                "lon": 127.3527,
                "mode": "nearby",
            },
            name="B mixed queries /api/dev/search-cache gapyeong_drive_healing",
            timeout=30,
        )