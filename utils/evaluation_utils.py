def precision_score(y_real, y_predict, k):
    cnt = 0
    y_predict = y_predict[: k]
    y_real = set(y_real)
    for y in y_predict:
        if y in y_real:
            cnt += 1
    return cnt / len(y_predict)


def recall_score(y_real, y_predict, k):
    cnt = 0
    y_predict = set(y_predict[: k])
    for y in y_real:
        if y in y_predict:
            cnt += 1
    return cnt / len(y_real)


def recall_rate_score(y_real, y_predict, k):
    y_predict = set(y_predict[: k])
    for y in y_real:
        if y in y_predict:
            return 1
    return 0
