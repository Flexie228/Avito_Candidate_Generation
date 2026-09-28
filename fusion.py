from collections import defaultdict


def RRF_with_business_logic(bm25_list, faiss_list, query_meta, item_meta, top_k=50):
    scores = defaultdict(float)

    # Базовый RRF (сглаживание 60)
    for rank, item_bm25 in enumerate(bm25_list):
        scores[item_bm25] += 1.0 / ((rank + 1) + 60)

    for rank, item_faiss in enumerate(faiss_list):
        scores[item_faiss] += 1.0 / ((rank + 1) + 60)

    # Применение soft-фильтров
    for item_id in scores:
        # 1. Штраф за скрытые контакты[cite: 11]
        if item_meta['phone_hidden'].get(item_id, 0) == 1 or item_meta['msg_forbidden'].get(item_id, 0) == 1:
            scores[item_id] *= 0.85

        # 2. Бустинг категории[cite: 11]
        if query_meta['search_category'] == item_meta['category'].get(item_id, -2):
            scores[item_id] *= 1.2

        # 3. Бустинг локации (если не ищут с доставкой)[cite: 11]
        if query_meta['search_location_id'] == item_meta['location'].get(item_id, -2):
            scores[item_id] *= 1.5
        elif query_meta['search_is_delivery_search'] == 1:
            # Смягчаем отсутствие локации, если есть доставка
            scores[item_id] *= 1.1

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [item[0] for item in sorted_scores[:top_k]]


def answersScores(bm25_output, faiss_output, queries_df, item_meta_dicts):
    result = {}

    # Индексируем параметры запросов для быстрого доступа
    queries_dict = queries_df.set_index('query_id').to_dict('index')

    for query_id in bm25_output.keys():
        query_meta = queries_dict.get(query_id, {})
        result[query_id] = RRF_with_business_logic(
            bm25_output[query_id],
            faiss_output[query_id],
            query_meta,
            item_meta_dicts
        )

    return result