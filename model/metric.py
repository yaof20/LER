import numpy as np


def ndcg_at_k(input_ranking, k=3, pos_score=None, zero_division=0):
    """
    NDCG@k = DCG@k / iDCG@k
         DCG = sum[ (2^rel_i - 1) / log(i+1) ],
        iDCG = sum[ (2^rel_i - 1) / log(i+1) ]|ideal_rank
    """
    input_true_score = np.array([case['score'] for case in input_ranking])
    ideal_true_score = np.sort(input_true_score)[::-1]

    input_true_score = input_true_score[:k]
    ideal_true_score = ideal_true_score[:k]

    input_gain = 2 ** input_true_score - 1
    ideal_gain = 2 ** ideal_true_score - 1
    discount = np.log2(np.arange(len(input_true_score)) + 2)

    dcg = sum(input_gain / discount)
    idcg = sum(ideal_gain / discount)
    ndcg = dcg / idcg if idcg else zero_division

    return ndcg


def precision_at_k(input_ranking, k=3, pos_score=2):
    """
    P@k = Yk / k,    Yk = #Positives in pred ranking top k
    """

    yk = sum([case['score'] >= pos_score for case in input_ranking[:k]])
    precision = yk/k

    return precision


def recall_at_k(input_ranking, k=3, pos_score=2, zero_division=0):
    """
    R@k = Yk / Gk,   Gk = #Positives in true ranking top k,
    """
    true_ranking = sorted(input_ranking, key=lambda x: x['score'], reverse=True)

    yk = sum([case['score'] >= pos_score for case in input_ranking[:k]])
    gk = sum([case['score'] >= pos_score for case in true_ranking[:k]])

    recall = yk/gk if gk else zero_division

    return recall


def mean_average_precision(input_ranking, pos_score=2, zero_division=0):
    """
    MAP = sum(P@ki) / r, where,
        ki are the indices of positives in the pred ranking,
        r is the number of positives.
    """
    rel = [idx+1 for idx, case in enumerate(input_ranking) if case['score'] >= pos_score]

    ma_precision = sum([precision_at_k(input_ranking, k=idx) for idx in rel]) / len(rel) if rel else zero_division

    return ma_precision


def mrr(input_ranking, pos_score=2, zero_division=0):
    """
    MRR = P@k1, where k1 is the index of the first positive in the pred ranking.
    """
    k1 = None
    for idx, case in enumerate(input_ranking):
        if case['score'] >= pos_score:
            k1 = idx + 1
            break
    mrr_score = precision_at_k(input_ranking, k=k1, pos_score=pos_score) if k1 else zero_division
    return mrr_score
