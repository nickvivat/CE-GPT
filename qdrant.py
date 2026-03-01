from qdrant_client import QdrantClient
from qdrant_client.http import models

client = QdrantClient("https://d534f10f-52e5-4422-997c-d3fa7a80cc01.eu-west-1-0.aws.cloud.qdrant.io", api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.VZUXujF6FLGIU8Xs5acrhkvqkpc1gSY-lG_4JjRy4V8")

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