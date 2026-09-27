import numpy as np
from rank_bm25 import BM25Okapi
from preprocessing import tokenizeText


def buildModelIndex(items_df):
    tokenized_df = items_df['clean_text'].apply(tokenizeText).tolist()
    return BM25Okapi(tokenized_df)


def getModelCandidates(bm25, items_df, queries_df, top_k=100):
    item_ids = items_df['item_id'].values
    candidates = {}

    for _, row in queries_df.iterrows():
        query_tokens = tokenizeText(row['clean_text'])
        scores = bm25.get_scores(query_tokens)

        # получаем индексы топ-k кандидатов с наибольшим скором (ключ - query_id, значения - item_id)
        top_indices = np.argsort(scores)[::-1][:top_k]
        candidates[row['query_id']] = item_ids[top_indices].tolist()

    return candidates