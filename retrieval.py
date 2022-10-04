import os

from sentence_transformers import SentenceTransformer, CrossEncoder, util
import torch


model_bi_encoder = "msmarco-distilbert-base-tas-b"
model_cross_encoder = "cross-encoder/ms-marco-MiniLM-L-12-v2"

bi_encoder = SentenceTransformer(model_bi_encoder)
bi_encoder.max_seq_length = 512

cross_encoder = CrossEncoder(model_cross_encoder)

top_k = 20


def get_corpus(passages):
    if "corpus.pt" not in os.listdir(os.getcwd()):
        corpus_embeddings = bi_encoder.encode(passages, convert_to_tensor=True, show_progress_bar=True)
        torch.save(corpus_embeddings, "corpus.pt")
    else:
        corpus_embeddings = torch.load("corpus.pt")
    return corpus_embeddings


def search(query, passages):
    corpus_embeddings = get_corpus(passages)
    question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
    hits = hits[0]
    cross_inp = [[query, passages[hit['corpus_id']]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inp)

    # Sort results by the cross-encoder scores
    for idx in range(len(cross_scores)):
        hits[idx]['cross-score'] = cross_scores[idx]

    hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
    for hit in hits[0:5]:
        print("{}".format(passages[hit['corpus_id']].replace("\n", " ")))
