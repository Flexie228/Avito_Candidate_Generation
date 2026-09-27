import math
from collections import defaultdict
import numpy as np
from tqdm import tqdm

from preprocessing import tokenizeText


class FastBM25:
    def __init__(self, corpus, k=1.5, b=0.75):
        self.k, self.b = k, b
        self.corpus_size = len(corpus)
        self.avgdl = sum(len(doc) for doc in corpus) / self.corpus_size     # средняя длина текста

        self.doc_len = np.zeros(self.corpus_size)                           # {документ: его длина}

        # словарь: слово -> список (индекс_документа, частота) (Inverse Document Frequency)
        self.inverted_index = defaultdict(list)
        self.idf = {}

        term_df = defaultdict(int)

        # строим индекс один раз при инициализации
        for idx, doc in enumerate(corpus):
            self.doc_len[idx] = len(doc)
            term_freqs = defaultdict(int)
            for term in doc:
                term_freqs[term] += 1

            for term, freq in term_freqs.items():
                self.inverted_index[term].append((idx, freq))
                term_df[term] += 1

        # считаем IDF для каждого уникального слова
        for term, df in term_df.items():
            self.idf[term] = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1)

    def get_scores(self, query):
        # сначала у всех документов по нулям
        scores = np.zeros(self.corpus_size)

        for term in query:
            if term not in self.inverted_index:
                continue

            term_idf = self.idf[term]
            # считаем балл только для документов, где есть искомое слово (аналогично, как bm25)
            for doc_idx, freq in self.inverted_index[term]:
                numerator = freq * (self.k + 1)
                denominator = freq + self.k * (1 - self.b + self.b * self.doc_len[doc_idx] / self.avgdl)
                scores[doc_idx] += term_idf * (numerator / denominator)

        return scores


def buildModelIndex(items_df):
    tokenized_df = items_df['clean_text'].apply(tokenizeText).tolist()
    return FastBM25(tokenized_df)


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