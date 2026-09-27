from collections import defaultdict


# константа сглаживания выбрана =60.
def RRF(bm25_list, faiss_list, top_k=50):
    scores = defaultdict(float)

    # ранк - просто индекс, который увеличивается ==> получаем меньший скор для убывающих мест
    for rank, (item_bm25, item_faiss) in enumerate(zip(bm25_list, faiss_list)):
        score = 1.0 / ((rank + 1) + 60)
        scores[item_bm25] += score
        scores[item_faiss] += score

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    #возвращаем топ-50 для каждого query_id
    return [item[0] for item in sorted_scores[:top_k]]


# получаем топ-50 кандидатов для каждого запроса
def answersScores(bm25_output, faiss_output):
    result = {}
    for query_id in bm25_output.keys():
        result[query_id] = RRF(bm25_output[query_id], faiss_output[query_id])

    return result