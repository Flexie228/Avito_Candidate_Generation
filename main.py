import sys
import logging
import pandas as pd
from tqdm import tqdm


from environment import checkEnvironment, setSeed, paths
from preprocessing import prepareItems, prepareQueries
from lexicalSearch import buildModelIndex, getModelCandidates
from semanticSearch import loadLocalModel, buildFaissIndex, getFaissCandidates
from fusion import answers_scores
from genAnswer import saveAnswer


checkEnvironment()
setSeed(56)

try:
    logging.info('чтение данных')
    items_df = pd.read_parquet(paths['benchmark_items'])
    queries_df = pd.read_parquet(paths['benchmark_queries'])

    logging.info('препроцессинг текстов')
    items_df, item_meta_dicts = prepareItems(items_df)
    queries_df = prepareQueries(queries_df)

    tqdm.write('INFO: работа лексической ветки (BM25)')
    bm25_index = buildModelIndex(items_df)
    bm25_candidates = getModelCandidates(bm25_index, items_df, queries_df, top_k=1000)

    tqdm.write('INFO: работа семантической ветки (RuBERT)')
    model = loadLocalModel(paths['rubert_finetuned'])
    faiss_index = buildFaissIndex(model, items_df)
    faiss_candidates = getFaissCandidates(model, faiss_index, items_df, queries_df, top_k=1000)

    logging.info('ансамблирование результатов')
    # передаем датафрейм запросов (чтобы брать оттуда параметры поиска) и словари объявлений
    res = answers_scores(bm25_candidates, faiss_candidates, queries_df, item_meta_dicts)

    logging.info('сохранение в файл answer.csv')
    saveAnswer(res, 'answer.csv')

except Exception as e:
    logging.error(f'произошла ошибка при выполнении: {e}')
    sys.exit(1)