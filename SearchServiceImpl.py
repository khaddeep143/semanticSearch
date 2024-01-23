import json

from flask import Flask
from llama_index.embeddings import HuggingFaceEmbedding
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from waitress import serve

app = Flask(__name__)

CLUSTER_ENDPOINT = "http://localhost:19530"  # Set your cluster endpoint
TOKEN = ""  # Set your token
COLLECTION_NAME = "test_collection"  # Set your collection name
DATASET_PATH = "DataRepository/high-performance-rag/medium_articles_2020_dpr.json"  # Set your dataset path


def vec_embeddings(text):
    embed_model = HuggingFaceEmbedding(model_name="e5-small-v2-fine-tuned")
    return embed_model.get_text_embedding(text)


if __name__ == '__main__':
    connections.connect(alias='default', uri="http://localhost:19530")

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="title_vector", dtype=DataType.FLOAT_VECTOR, dim=384),
    FieldSchema(name="link", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="reading_time", dtype=DataType.INT64),
    FieldSchema(name="publication", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="claps", dtype=DataType.INT64),
    FieldSchema(name="responses", dtype=DataType.INT64)
]
schema = CollectionSchema(fields, description="Schema of Medium articles", enable_dynamic_field=False)
collection = Collection(name=COLLECTION_NAME, schema=schema)

search_params = {"metric_type": "L2"}

results = collection.search(
    data=[vec_embeddings("Quick")],
    anns_field="title_vector",
    param=search_params,
    output_fields=["id", "title", "link"],
    limit=5
)

distances = results[0].distances

print(distances)

entities = [x.entity.to_dict()["entity"] for x in results[0]]
print(entities)

serve(app, host='0.0.0.0', port=250)
