import os

import json
from flask import Flask
from llama_index.llms import OpenAI
from waitress import serve
from llama_index.finetuning import generate_qa_embedding_pairs, SentenceTransformersFinetuneEngine
from llama_index.finetuning import EmbeddingQAFinetuneDataset
from llama_index.node_parser import SentenceSplitter

from llama_index.schema import Document

app = Flask(__name__)

os.environ["OPENAI_API_KEY"] = ""

TRAIN_FILES_INPUT_LOCATION = "DataRepository/high-performance-rag/doc-1.json"

TRAIN_DATASET_OUTPUT_LOCATION = "DataRepository/high-performance-rag/train_dataset_new.json"
VAL_DATASET_OUTPUT_LOCATION = "DataRepository/high-performance-rag/val_dataset_new.json"

HUGGING_FACE_MODEL = "intfloat/e5-small-v2"

FINE_TUNED_MODEL_LOCATION = "e5-small-v2-fine-tuned"


@app.route('/load_corpus', methods=['POST'])
def load_corpus_from_json(paragraphs_list):
    documents_list = []
    for para in paragraphs_list:
        documents_list.append(Document(text=para))

    node_parser = SentenceSplitter(chunk_size=50, chunk_overlap=0)
    nodes = node_parser.get_nodes_from_documents(documents_list, show_progress=True)
    print(f"Parsed {len(nodes)} nodes")

    return nodes


@app.route('/training_dataset', methods=['POST'])
def get_training_dataset_json():
    return EmbeddingQAFinetuneDataset.from_json(TRAIN_DATASET_OUTPUT_LOCATION)


@app.route('/validation_dataset', methods=['POST'])
def get_validation_dataset_json():
    return EmbeddingQAFinetuneDataset.from_json(VAL_DATASET_OUTPUT_LOCATION)


def finetune(train_dataset, val_dataset, no_epochs):
    finetune_engine = SentenceTransformersFinetuneEngine(
        train_dataset,  # Dataset to be trained on
        model_id=HUGGING_FACE_MODEL,  # HuggingFace reference to base embeddings model
        model_output_path=FINE_TUNED_MODEL_LOCATION,  # Output directory for fine-tuned embeddings model
        val_dataset=val_dataset,  # Dataset to validate on
        epochs=no_epochs  # Number of Epochs to train for
    )
    finetune_engine.finetune()

    fine_tuned_embedding_model = finetune_engine.get_finetuned_model()
    return fine_tuned_embedding_model.to_json()


def get_dataSet(nodes, json_output_file):
    llm = OpenAI(temperature=0.0, model="gpt-3.5-turbo")

    train_dataset = generate_qa_embedding_pairs(nodes, llm)
    train_dataset.save_json(json_output_file)
    return train_dataset


if __name__ == '__main__':
    json_file = open(TRAIN_FILES_INPUT_LOCATION)
    reader = json.load(json_file)

    paragraphs_train = [paragraph['content'] for paragraph in reader['paragraphs']]
    train_nodes = load_corpus_from_json(paragraphs_train)
    temp_train_dataset = get_dataSet(train_nodes, TRAIN_DATASET_OUTPUT_LOCATION)

    paragraphs_val = [paragraph['content'] for paragraph in reader['paragraphs']]
    val_nodes = load_corpus_from_json(paragraphs_val)
    temp_val_dataset = get_dataSet(val_nodes, VAL_DATASET_OUTPUT_LOCATION)

    finetune(get_training_dataset_json(), get_validation_dataset_json(), 10)
    serve(app, host='0.0.0.0', port=500)
