from .metric import ndcg_at_k, precision_at_k, recall_at_k, mrr, mean_average_precision
from transformers import AutoTokenizer, BertTokenizerFast


backbone_plm_dict = {
    'bert': 'bert-base-chinese',
    'bert-tiny': 'ckiplab/bert-tiny-chinese',
    'albert': 'ckiplab/albert-base-chinese',
    'roberta': 'hfl/chinese-roberta-wwm-ext',
    'ernie': 'nghuyong/ernie-1.0',
    'mengzi': 'Langboat/mengzi-bert-base',
    'lawformer': 'thunlp/Lawformer',
    'sbert': 'sentence-transformers/all-MiniLM-L6-v2',
    'legal-simcse': './PLMs/legal-simcse'
}

special_tokenizer = {
    'bert-tiny': BertTokenizerFast,
    'albert': BertTokenizerFast
}


metric_dict = {
    "MRR": mrr,
    "NDCG@k": ndcg_at_k,
    "P@k": precision_at_k,
    "R@k": recall_at_k,
    "MAP": mean_average_precision,

}


def init_tokenizer(name, *args, **params):
    plm_path = backbone_plm_dict[name]
    if name in special_tokenizer:
        return special_tokenizer[name].from_pretrained(plm_path, cache_dir='./data/ckpt/')
    else:
        return AutoTokenizer.from_pretrained(plm_path, cache_dir='./data/ckpt/')


def init_metric(name):
    if name in metric_dict:
        return metric_dict[name]
    else:
        raise NotImplementedError
