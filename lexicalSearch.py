import numpy as np
from rank_bm25 import BM25Okapi
from preprocessing import tokenizeText
from tqdm import tqdm


def buildModelIndex(items_df):
    tokenized_df = items_df['clean_text'].apply(tokenizeText).tolist()
    return BM25Okapi(tokenized_df)


def getModelCandidates(bm25, items_df, queries_df, top_k=100):
    item_ids = items_df['item_id'].values
    candidates = {}

    for _, row in tqdm(queries_df.iterrows(), total=len(queries_df), desc="bm25 поиск"):
        query_tokens = tokenizeText(row['clean_text'])
        scores = bm25.get_scores(query_tokens)

        # сортировка (находим только первые 100)
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        candidates[row['query_id']] = item_ids[top_indices].tolist()

    return candidates