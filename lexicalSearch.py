import math
from collections import defaultdict
import numpy as np
from tqdm import tqdm

from preprocessing import tokenizeText


class FastBM25:
    def __init__(self, corpus_texts, k=1.5, b=0.75):
        self.k, self.b = k, b
        self.corpus_size = len(corpus_texts)
        self.doc_len = np.zeros(self.corpus_size, dtype=np.float32)

        # Используем два параллельных списка вместо тяжелых кортежей
        self.inverted_index_docs = defaultdict(list)
        self.inverted_index_freqs = defaultdict(list)
        term_df = defaultdict(int)

        total_len = 0

        # Токенизируем тексты на лету, чтобы не держать в памяти весь корпус слов
        for idx, text in tqdm(enumerate(corpus_texts), total=self.corpus_size, desc="Сборка индекса BM25"):
            doc = tokenizeText(text)
            length = len(doc)
            self.doc_len[idx] = length
            total_len += length

            term_freqs = defaultdict(int)
            for term in doc:
                term_freqs[term] += 1

            for term, freq in term_freqs.items():
                self.inverted_index_docs[term].append(idx)
                self.inverted_index_freqs[term].append(freq)
                term_df[term] += 1

        self.avgdl = total_len / self.corpus_size
        self.idf = {}

        for term, df in term_df.items():
            self.idf[term] = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1)

        for term in list(self.inverted_index_docs.keys()):
            self.inverted_index_docs[term] = np.array(self.inverted_index_docs[term], dtype=np.int32)
            self.inverted_index_freqs[term] = np.array(self.inverted_index_freqs[term], dtype=np.int32)

    def get_scores(self, query):
        scores = np.zeros(self.corpus_size, dtype=np.float32)

        for term in query:
            if term not in self.inverted_index_docs:
                continue

            term_idf = self.idf[term]
            docs = self.inverted_index_docs[term]
            freqs = self.inverted_index_freqs[term]

            # Полностью векторизованное вычисление NumPy (работает без питоновских циклов)
            numerators = freqs * (self.k + 1.0)
            denominators = freqs + self.k * (1 - self.b + self.b * self.doc_len[docs] / self.avgdl)
            scores[docs] += term_idf * (numerators / denominators)

        return scores


def buildModelIndex(items_df):
    return FastBM25(items_df['clean_text'].values)


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