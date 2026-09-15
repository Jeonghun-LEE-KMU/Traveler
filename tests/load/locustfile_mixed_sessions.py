from __future__ import annotations

from locust import HttpUser, between, task


class MixedSessionUser(HttpUser):
    """
    B. Mixed sessions 부하테스트.

    사용자별 session_id를 다르게 두고 같은 query를 반복 호출한다.
    여러 세션이 동시에 들어와도 Redis/session 처리와 검색 캐시가 안정적인지 확인한다.
    """

    wait_time = between(1, 2)

    def on_start(self) -> None:
        self.session_id = f"locust-user-{id(self)}"

    @task
    def gyeongju_history_nearby(self) -> None:
        self.client.get(
            "/api/chat",
            params={
                "user_message": "경주에서 역사 유적지 중심으로 여행하고 싶어",
                "lat": 35.8562,
                "lon": 129.2247,
                "session_id": self.session_id,
                "mode": "nearby",
            },
            name="B mixed sessions /api/chat gyeongju_history",
            timeout=30,
        )