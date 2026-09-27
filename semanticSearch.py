import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer


def loadLocalModel(model_path='./rubert_finetuned'):
    # загружаем дообученную модель
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(model_path, device=device)
    return model


# "первая башня"
# этот индекс создаем чтобы не сравнивать эмбеддинг с каждым из 200тыс других
def buildFaissIndex(model, items_df, batch_size=256):
    texts = items_df['clean_text'].tolist()

    # получаем эмбеддинги
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)

    # проводим l2-нормализацию: получаем: скалярное произведение ЭКВИВАЛЕНТНО косинусному расстоянию
    faiss.normalize_L2(embeddings)

    # индекс поиска по скалярному произведению
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


# "вторая башня"
# тут работаем с векторами запросов, искать для них кандидатов будем среди уже готового на прошлом шаге индекса
def getFaissCandidates(model, index, items_df, queries_df, top_k=100, batch_size=256):
    item_ids = items_df['item_id'].values
    queries_texts = queries_df['clean_text'].tolist()

    # генерируем и нормализуем эмбеддинги запросов
    query_embeddings = model.encode(queries_texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    faiss.normalize_L2(query_embeddings)

    # ищем топ-k ближайших соседей
    distances, indices = index.search(query_embeddings, top_k)

    candidates = {}
    for i, query_id in enumerate(queries_df['query_id']):
        candidates[query_id] = item_ids[indices[i]].tolist()

    return candidates