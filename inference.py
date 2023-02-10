import jsonlines
import json
import torch

from tqdm import tqdm
from config_parser import create_config
from model.models import TFIDFSearcher, RobertaEncoder
from model.CrossCaseCL import CrossCaseCL
from transformers import BertTokenizerFast, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


if __name__ == "__main__":
    checkpoint = './output/model/harm_simcse_contra_attn_loss_roberta/3-10k.pkl'
    infer_args = dict(tfidf=True, roberta=False, ours=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_path = './data/new_short/test_example'
    if 'jsonl' in test_path:
        test_data = list(jsonlines.open(test_path))
    else:
        test_data = json.load(open(test_path, encoding='utf-8'))

    if infer_args['roberta']:
        roberta = RobertaEncoder('hfl/chinese-roberta-wwm-ext').to(device)

    if infer_args['tfidf']:
        tfidf = TFIDFSearcher()

    if infer_args['ours']:
        config = create_config('previous_configs/config/train/train.config')
        ours = CrossCaseCL(config).to(device)
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        ours.load_state_dict(checkpoint['model'], strict=False)

    tokenizer = BertTokenizerFast.from_pretrained('hfl/chinese-roberta-wwm-ext')

    outputs = dict(tfidf=[], roberta=[], ours=[])

    for data in tqdm(test_data, desc='Testing...'):
        doc_tmp = dict(tfidf=[], roberta=[], ours=[])
        fact_list = data['fact']
        evidence_list = data['evidence']

        test_names = []
        test_scores = []

        if infer_args['tfidf']:
            sim_score_tfidf = tfidf.search(fact_list, evidence_list).tolist()
            test_names.append('tfidf')
            test_scores.append(sim_score_tfidf)

        if infer_args['roberta']:
            roberta.eval()
            with torch.no_grad():
                sim_score_roberta = roberta.search(fact_list, evidence_list).tolist()
                test_names.append('roberta')
                test_scores.append(sim_score_roberta)

        if infer_args['ours']:
            ours.eval()
            with torch.no_grad():
                # ours
                inputs_f = tokenizer(fact_list, padding='longest', truncation=True, max_length=128, return_tensors='pt')
                inputs_e = tokenizer(evidence_list, padding='longest', truncation=True, max_length=128, return_tensors='pt')

                for ipt_key in inputs_f.keys():
                    inputs_f[ipt_key] = inputs_f[ipt_key].to(device)
                    inputs_e[ipt_key] = inputs_e[ipt_key].to(device)

                embedding_f = ours.encoder_f(**inputs_f)
                embedding_e = ours.encoder_f(**inputs_e)
                sim_score_ours = cosine_similarity(embedding_f.cpu(), embedding_e.cpu()).tolist()

                test_names.append('ours')
                test_scores.append(sim_score_ours)

        for name, scores in list(zip(test_names, test_scores)):
            for fid, score in enumerate(scores):
                fact = fact_list[fid]
                tmp = dict(query=fact, evidences=[])
                for eid, s in enumerate(score):
                    evidence = evidence_list[eid]
                    tmp['evidences'].append({'evidence': evidence, 'pred_score': s})

                tmp['evidences'] = sorted(tmp['evidences'], key=lambda x: x['pred_score'], reverse=True)
                doc_tmp[name].append(tmp)
            outputs[name].append(doc_tmp[name])

    for name in outputs:
        with open('./data/new_short/predictions/{}-pred.json'.format(name), 'w', encoding='utf-8') as f:
            json.dump(outputs[name], f, indent=4, ensure_ascii=False)
