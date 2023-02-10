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
    checkpoint = './output/August/harm-wo-test/shared/avg/roberta/simcse-10-contra-100-cos/1-1k.pkl'
    infer_args = dict(roberta=False, ours=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_path = '../data/test_example/1.txt'
    test_data = [json.load(open(test_path, encoding='utf-8'))]

    if infer_args['ours']:
        config = create_config(
            '../previous_configs/config-add/train/harm-wo-test/shared/avg/roberta/simcse-10-contra-100-cos.config')
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
                sim_score_ours = cosine_similarity(embedding_e.cpu(), embedding_f.cpu()).tolist()

                attention_scores = ours.attention(queries=embedding_e, keys=embedding_f, values=embedding_f)
                weights = attention_scores['sim_scores'].tolist()
                attention_scores = attention_scores['attn_scores'].tolist()

                result = []
                for f_id in range(len(sim_score_ours)):
                    fact = data['evidence'][f_id]
                    fact_tmp = dict(evidence=fact, cos_sim=[], attention=[], weight=[])
                    for e_id in range(len(sim_score_ours[f_id])):
                        evidence = data['fact'][e_id]
                        tmp1 = dict(text=evidence,
                                    attn_score=attention_scores[f_id][e_id])
                        fact_tmp['attention'].append(tmp1)
                        tmp2 = dict(text=evidence,
                                    attn_score=sim_score_ours[f_id][e_id])
                        fact_tmp['cos_sim'].append(tmp2)
                        tmp3 = dict(text=evidence,
                                    attn_score=weights[f_id][e_id])
                        fact_tmp['weight'].append(tmp3)

                    result.append(fact_tmp)

                with open('../data/test_sample_result-evidence-query.txt', 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=4, ensure_ascii=False)

