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
    connections.connect(alias='default', uri=CLUSTER_ENDPOINT)

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
collection = Collection(name=COLLECTION_NAME, description="Medium articles", schema=schema)
index_params = {
    "index_type": "AUTOINDEX",
    "metric_type": "L2",
    "params": {}
}

collection.create_index(field_name="title_vector", index_params=index_params, index_name='title_vector_index')
collection.load()
progress = utility.loading_progress(COLLECTION_NAME)

print(progress)

data = [{'id': 0, 'title': 'Test', 'title_vector': vec_embeddings("A Quick Brown Fox Jumps"), 'link': "test.xyz",
         'reading_time': 13,
         'publication': 'Kathmandu', 'claps': 100, 'responses': 1},
        {'id': 1, 'title': 'Test1', 'title_vector': vec_embeddings("A Quick Brown Fox Jumps"), 'link': "test.xyz",
         'reading_time': 13,
         'publication': 'Kathmandu', 'claps': 100, 'responses': 1}
        ]

results = collection.upsert(data)
collection.flush()

print(f"Data upserted successfully! Upstarted rows: {results.upsert_count}")

serve(app, host='0.0.0.0', port=250)
