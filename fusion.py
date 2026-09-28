from collections import defaultdict


def RRF_score(bm25_list, faiss_list, k_rrf=60, w_bm25=1.3, w_faiss=1.0):
    scores = defaultdict(float)
    # даем больший вес bm25, так как лексическая ветка показала более высокий recall при тестировании
    # 60 — стандартная сглаживающая константа для алгоритма rrf
    for rank, item_id in enumerate(bm25_list):
        scores[item_id] += w_bm25 / (k_rrf + rank + 1)
    for rank, item_id in enumerate(faiss_list):
        scores[item_id] += w_faiss / (k_rrf + rank + 1)
    return scores


def answers_scores(
    bm25_output,
    faiss_output,
    queries_df,
    item_meta_dicts,
    top_k=50,
    w_bm25=1.3,
    w_faiss=1.0,
    loc_match_boost=3.0,
    loc_mismatch_penalty=0.15,
    cat_match_boost=2.0,
    cat_mismatch_penalty=0.35,
    delivery_boost=1.7,
    hidden_contact_penalty=0.7
):
    # словари параметров объявлений
    item_loc_map = item_meta_dicts['location']
    item_cat_map = item_meta_dicts['category']
    item_phone_map = item_meta_dicts['phone_hidden']
    item_msg_map = item_meta_dicts['msg_forbidden']

    # индексируем параметры запросов в словари для быстрого доступа
    q_loc_map = queries_df.set_index('query_id')['search_location_id'].fillna(-1).astype(int).to_dict()
    q_cat_map = queries_df.set_index('query_id')['search_category'].fillna(-1).astype(int).to_dict()
    q_delivery_map = queries_df.set_index('query_id')['search_is_delivery_search'].fillna(0).astype(int).to_dict()

    result = {}

    for query_id in bm25_output.keys():
        b_list = bm25_output.get(query_id, [])
        f_list = faiss_output.get(query_id, [])
        scores = RRF_score(b_list, f_list, k_rrf=60, w_bm25=w_bm25, w_faiss=w_faiss)

        q_loc = q_loc_map.get(query_id, -1)
        q_cat = q_cat_map.get(query_id, -1)
        q_delivery = q_delivery_map.get(query_id, 0)

        for item_id in scores:
            multiplier = 1.0

            # если закрыты оба канала связи, контакт невозможен
            is_phone_hidden = item_phone_map.get(item_id, 0) == 1
            is_msg_forbidden = item_msg_map.get(item_id, 0) == 1
            if is_phone_hidden and is_msg_forbidden:
                multiplier *= 0.5
            # конверсия в объявлениях со скрытыми контактами 100% ниже, поэтому сдвигаем их вниз выдачи штрафом
            elif is_phone_hidden or is_msg_forbidden:
                multiplier *= hidden_contact_penalty

            # бустим совпадение категорий, штрафуем несовпадение
            if q_cat != -1:
                item_cat = item_cat_map.get(item_id, -2)
                # если пользователь указал категорию, то он точно хочет видеть именно ее, поэтому:
                if item_cat == q_cat:
                    multiplier *= cat_match_boost
                elif item_cat != -2:
                    multiplier *= cat_mismatch_penalty

            # аналогично для локации
            if q_loc != -1:
                item_loc = item_loc_map.get(item_id, -2)
                # смягчаем отсутствие локации, если пользователь ищет услуги с доставкой
                # если не указана доставка, то чужой город это "до свидания", поэтому местное объявление получает буст
                if item_loc == q_loc:
                    multiplier *= loc_match_boost
                elif q_delivery == 1:
                    multiplier *= delivery_boost
                elif item_loc != -2:
                    multiplier *= loc_mismatch_penalty

            scores[item_id] *= multiplier

        # итоговый срез топ-50
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        result[query_id] = [item[0] for item in ranked[:top_k]]

    return result