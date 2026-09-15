from __future__ import annotations

from locust import HttpUser, between, task


class WarmCacheUser(HttpUser):
    """
    A. Warm cache 부하테스트.

    동일 session_id + 동일 query + 동일 좌표를 반복 호출해
    Redis cache hit 상태의 응답 시간을 확인한다.
    """

    wait_time = between(1, 2)

    @task
    def gyeongju_history_nearby(self) -> None:
        self.client.get(
            "/api/chat",
            params={
                "user_message": "경주에서 역사 유적지 중심으로 여행하고 싶어",
                "lat": 35.8562,
                "lon": 129.2247,
                "session_id": "locust-cache-test",
                "mode": "nearby",
            },
            name="A warm cache /api/chat gyeongju_history",
            timeout=30,
        )