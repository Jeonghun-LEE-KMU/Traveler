FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성 (psycopg2-binary 빌드용)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 의존성 먼저 (캐시 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 복사
COPY . .

# places.db migration 실행 (SQLite 잔존 파일 있을 경우 대비)
# RDS 사용 시 이 단계는 무시해도 됨

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
