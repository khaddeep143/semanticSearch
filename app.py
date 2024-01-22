import os
from flask import Flask, request
from llama_index.embeddings import HuggingFaceEmbedding
from pymilvus import MilvusClient

from waitress import serve

os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = Flask(__name__)

COLLECTION_NAME = "medium_articles_2020"


def vec_embeddings(text):
    embed_model = HuggingFaceEmbedding(model_name="e5-small-v2-fine-tuned")
    return embed_model.get_text_embedding(text)


# @app.route('/embedding', methods=['POST'])
def getEmbeddings():
    text = request.args.get("text")
    return vec_embeddings(text)


# @app.route('/create_collection', methods=['POST'])
def create_collection():
    client = MilvusClient(uri="http://localhost:19530")
    client.create_collection(collection_name=COLLECTION_NAME, dimension=384)


# @app.route('/describe_collection', methods=['POST'])
def describe_collection():
    client = MilvusClient(uri="http://localhost:19530")
    res = client.describe_collection(collection_name=COLLECTION_NAME)
    return res


# @app.route('/insert', methods=['POST'])
def insert():
    client = MilvusClient(uri="http://localhost:19530")
    res = client.insert(
        collection_name=COLLECTION_NAME,
        data={
            'id': 4,
            'title': 'The Reported Mortality Rate of Coronavirus Is Not Important',
            'link': '<https://medium.com/swlh/the-reported-mortality-rate-of-coronavirus-is-not-important-369989c8d912>',
            'reading_time': 13,
            'publication': 'The Startup',
            'claps': 1100,
            'responses': 18,
            'vector': vec_embeddings("world")
        })
    print(res)


if __name__ == '__main__':
    # create_collection()
    insert()
serve(app, host='0.0.0.0', port=250)
