import re

PUNCTUATION_PATTERN = re.compile(r'[^\w\s]')
SPACE_PATTERN = re.compile(r'\s+')

def cleanText(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = PUNCTUATION_PATTERN.sub(' ', text)
    return SPACE_PATTERN.sub(' ', text).strip()

def tokenizeText(text):
    return text.split()

def prepareQueries(df):
    # склеиваем текст (именно по склейкам будем начислять баллы)
    df['clean_title'] = df['search_query'].fillna('').apply(cleanText)
    df['clean_params'] = df['search_infm_params_text'].fillna('').apply(cleanText)
    df['clean_text'] = df['clean_title'] + " " + df['clean_params']

    # извлекаем метаданные поиска
    df['search_location_id'] = df['search_location_id'].fillna(-1).astype(int)
    df['search_category'] = df['search_category'].fillna(-1).astype(int)
    df['search_is_delivery_search'] = df['search_is_delivery_search'].fillna(0).astype(int)
    return df

def prepareItems(df):
    # склеиваем текст (именно по склейкам будем начислять баллы)
    df['clean_title'] = df['item_title_raw'].fillna('').apply(cleanText)
    df['clean_params'] = df['item_infm_params_text'].fillna('').apply(cleanText)
    df['clean_desc'] = df['item_description_raw'].fillna('').apply(cleanText)
    df['clean_text'] = df['clean_title'] + " " + df['clean_params'] + " " + df['clean_desc']

    # словари для реализации бизнес-логики при ансамблировании
    meta_dicts = {
        'location': df.set_index('item_id')['item_location_id'].fillna(-1).astype(int).to_dict(),
        'category': df.set_index('item_id')['item_category_id'].fillna(-1).astype(int).to_dict(),
        'phone_hidden': df.set_index('item_id')['item_is_phone_hidden'].fillna(0).astype(int).to_dict(),
        'msg_forbidden': df.set_index('item_id')['item_is_message_forbidden'].fillna(0).astype(int).to_dict()
    }
    return df, meta_dicts