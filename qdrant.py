from qdrant_client import QdrantClient
from qdrant_client.http import models

client = QdrantClient("http://10.240.68.50:6333/")

client.set_payload(
    collection_name="CE-GPT",
    payload={
        "data_type": "curriculum"
    },
    # This filter finds every point where filename matches your string
    points=models.Filter(
        must=[
            models.FieldCondition(
                key="filename",
                match=models.MatchValue(value="curriculum-structure.md"),
            )
        ]
    ),
)