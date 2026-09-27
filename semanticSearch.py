import os
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import normalize_embeddings


def loadLocalModel(model_path='./rubert_finetuned'):
    # загружаем дообученную модель
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(model_path, device=device)
    model.max_seq_length = 512
    return model


# "первая башня"
# FAISS индекс создаем чтобы не сравнивать эмбеддинг с каждым из 200тыс других
def buildFaissIndex(model, items_df, batch_size=128):
    embeddings_path = 'semantic_embeddings.npy'
    index_path = 'semantic.index'

    if os.path.exists(embeddings_path) and os.path.exists(index_path):
        embeddings = np.load(embeddings_path, mmap_mode='r')
        index = faiss.read_index(index_path)
        if index.ntotal != len(items_df):
            raise ValueError(f'Количество embeddings ({index.ntotal}) не совпадает с количеством items ({len(items_df)})')
        return index

    texts = items_df['clean_text'].tolist()

    # получаем эмбеддинги
    embeddings = model.encode(texts,
                              batch_size=batch_size,
                              show_progress_bar=True,
                              convert_to_numpy=True,
                              normalize_embeddings=True)
    embeddings = embeddings.astype(np.float32, copy=False)
    np.save(embeddings_path, embeddings)

    dimension = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 300
    index.hnsw.efSearch = 150
    index.add(embeddings)
    faiss.write_index(index, index_path)
    return index


# "вторая башня"
# тут работаем с векторами запросов, искать для них кандидатов будем среди уже готового на прошлом шаге индекса
def getFaissCandidates(model, index, items_df, queries_df, top_k=100, batch_size=128):
    item_ids = items_df['item_id'].values
    queries_texts = queries_df['clean_text'].tolist()

    # генерируем и нормализуем эмбеддинги запросов
    query_embeddings = model.encode(queries_texts,
                                    batch_size=batch_size,
                                    show_progress_bar=True,
                                    convert_to_numpy=True,
                                    normalize_embeddings=True)
    query_embeddings = query_embeddings.astype(np.float32, copy=False)

    distances, indices = index.search(query_embeddings, top_k)

    candidates = {}
    for i, query_id in enumerate(queries_df['query_id']):
        candidates[query_id] = item_ids[indices[i]].tolist()

    return candidates