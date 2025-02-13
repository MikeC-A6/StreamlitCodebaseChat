import os
from dataclasses import dataclass

@dataclass
class Settings:
    OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]
    PINECONE_API_KEY: str = os.environ["PINECONE_API_KEY"]
    PINECONE_INDEX: str = os.environ["PINECONE_INDEX"]
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSION: int = 1536

settings = Settings()
