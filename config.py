import os
from dotenv import load_dotenv

# .env 파일에 숨겨둔 키들을 메모리에 로드
load_dotenv()

# 변수에 키를 담음
KORSERVICE_API_KEY = os.getenv("KORSERVICE_API_KEY")
KMA_API_KEY = os.getenv("KMA_API_KEY")
KORWITHSERVICE_API_KEY = os.getenv("KORWITHSERVICE_API_KEY")
TARRITETARSERVICE_API_KEY= os.getenv("TARRITETARSERVICE_API_KEY")
MDCLTURSMSERVICE_API_KEY = os.getenv("MDCLTURSMSERVICE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Sprint 2 추가
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Sprint 3 예정 (Pinecone 연결 시 사용)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")