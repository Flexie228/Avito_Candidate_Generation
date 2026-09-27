import pandas as pd
import re


def cleanText(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def tokenizeText(text):
    # по пробелам для bm25
    return text.split()


def prepareQueries(df):
    # склейка: текст запроса + фильтры
    res_text = df['search_query'].fillna('') + ' ' + df['search_infm_params_text'].fillna('')
    df['clean_text'] = res_text.apply(cleanText)
    return df


def prepareItems(df):
    # склейка: заголовок + описание + параметры объявления
    res_text = df['item_title_raw'].fillna('') + ' ' + df['item_description_raw'].fillna('') + ' ' + df['item_infm_params_text'].fillna('')
    df['clean_text'] = res_text.apply(cleanText)
    return df