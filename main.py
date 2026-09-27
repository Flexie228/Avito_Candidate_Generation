import sys
import logging
from logging import exception
import pickle
import pandas as pd


from environment import checkEnvironment, setSeed
from preprocessing import prepareItems, prepareQueries
from lexicalSearch import buildModelIndex, getModelCandidates
from semanticSearch import loadLocalModel, buildFaissIndex, getFaissCandidates
from fusion import answersScores
from genAnswer import saveAnswer


checkEnvironment()
setSeed(56)

try:
    logging.info('читаем данные')
    items_df = pd.read_parquet('benchmark_items.parquet')
    queries_df = pd.read_parquet('benchmark_queries.parquet')

    logging.info('препроцессинг текстов')
    items_df = prepareItems(items_df)
    queries_df = prepareQueries(queries_df)

    logging.info('запускаем лексическую ветку (BM25)')
    bm25_index = buildModelIndex(items_df)
    bm25_candidates = getModelCandidates(bm25_index, items_df, queries_df, top_k=100)

    logging.info('запускаем семантическую ветку (RuBERT)')
    model = loadLocalModel('./rubert_finetuned')
    faiss_index = buildFaissIndex(model, items_df)
    faiss_candidates = getFaissCandidates(model, faiss_index, items_df, queries_df, top_k=100)

    logging.info('ансамблируем')
    res = answersScores(bm25_candidates, faiss_candidates)

    logging.info('сохраняем в файл answer.csv')
    saveAnswer(res, 'answer.csv')

except Exception as e:
    logging.error(f'произошла ошибка при выполнении: {e}')
    sys.exit(1)