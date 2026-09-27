import os
import sys
import logging


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', stream=sys.stdout)


def checkEnvironment():
    required_paths = [
        'benchmark_items.parquet',
        'benchmark_queries.parquet',
        'rubert_finetuned'
    ]

    for path in required_paths:
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