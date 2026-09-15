from __future__ import annotations

import random

from locust import HttpUser, between, task


SCENARIOS = [
    {
        "user_message": "경주에서 역사 유적지 중심으로 여행하고 싶어",
        "lat": 35.8562,
        "lon": 129.2247,
        "mode": "nearby",
    },
    {
        "user_message": "경주에서 연인이랑 감성 있는 야경 데이트 코스 추천해줘",
        "lat": 35.8562,
        "lon": 129.2247,
        "mode": "nearby",
    },
    {
        "user_message": "서울에서 비 오는 날 실내로 갈 만한 문화시설 추천해줘",
        "lat": 37.5665,
        "lon": 126.9780,
        "mode": "nearby",
    },
    {
        "user_message": "서울에서 아이랑 실내에서 체험할 수 있는 곳 위주로 추천해줘",
        "lat": 37.5665,
        "lon": 126.9780,
        "mode": "nearby",
    },
    {
        "user_message": "가평으로 드라이브 가서 조용하게 쉬고 풍경 볼 수 있는 곳 알려줘",
        "lat": 37.8315,
        "lon": 127.5099,
        "mode": "nearby",
    },
    {
        "user_message": "양평에서 강가나 물가 근처로 산책하고 쉬기 좋은 곳 찾고 싶어",
        "lat": 37.4912,
        "lon": 127.4876,
        "mode": "nearby",
    },
    {
        "user_message": "남양주에서 전망 좋고 감성 있는 카페나 산책 장소 추천해줘",
        "lat": 37.6360,
        "lon": 127.2165,
        "mode": "nearby",
    },
    {
        "user_message": "파주에서 역사나 평화 관련해서 의미 있게 볼 수 있는 곳 추천해줘",
        "lat": 37.7599,
        "lon": 126.7802,
        "mode": "nearby",
    },
]


class MixedQueryUser(HttpUser):
    """
    C. Mixed queries 부하테스트.

    사용자별 session_id와 query를 모두 다르게 섞어
    실제 서비스에 가까운 다양한 요청 부하를 확인한다.
    """

    wait_time = between(1, 2)

    def on_start(self) -> None:
        self.session_id = f"locust-user-{id(self)}"

    @task
    def mixed_query_chat(self) -> None:
        scenario = random.choice(SCENARIOS)

        self.client.get(
            "/api/chat",
            params={
                "user_message": scenario["user_message"],
                "lat": scenario["lat"],
                "lon": scenario["lon"],
                "session_id": self.session_id,
                "mode": scenario["mode"],
            },
            name="C mixed queries /api/chat",
            timeout=30,
        )