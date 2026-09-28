import os
import sys
import logging


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', stream=sys.stdout)
logging.getLogger('faiss.loader').setLevel(logging.WARNING)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

paths = {
    'benchmark_items': 'data/benchmark_items.parquet',
    'benchmark_queries': 'data/benchmark_queries.parquet',
    'rubert_finetuned': './rubert_finetuned'
}

def checkEnvironment():
    for path in paths.values():
        if not os.path.exists(path):
            logging.error(f'не найден обязательный путь: {path}')
            sys.exit(1)


def setSeed(seed=56):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)