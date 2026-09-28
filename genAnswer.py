import pandas as pd


# формируем ответ и сохраняем его в файл
def saveAnswer(result, output_path='answer.csv'):
    query_ids = list(result.keys())
    predictions = list(result.values())

    answers = pd.DataFrame({'query_id': query_ids,
                            'answer': [' '.join(res_query) for res_query in predictions]})
    answers.to_csv(output_path, index=False)