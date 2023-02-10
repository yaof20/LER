import jsonlines
import json
import torch

from tqdm import tqdm
from config_parser import create_config
from model.models import TFIDFSearcher, RobertaEncoder
from model.CrossCaseCL import CrossCaseCL
from transformers import BertTokenizerFast, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


def get_embedding_list(input_data, batch_size):
    our_vectors, roberta_vectors = [], []
    for batch in tqdm(generate_batch(input_data, batch_size)):
        inputs = tokenizer(batch, padding='longest', truncation=True, max_length=128, return_tensors='pt')
        for key in inputs:
            inputs[key] = inputs[key].to(device)
        with torch.no_grad():
            outputs1 = roberta(**inputs)['pooler_output'].tolist()
            outputs2 = ours.encoder_f(**inputs).tolist()

            roberta_vectors += outputs1
            our_vectors += outputs2

    return our_vectors, roberta_vectors


def generate_batch(input_data, batch_size):
    batches = []
    for i in range(len(input_data) // batch_size + 1):
        batches.append(input_data[i * batch_size:(i + 1) * batch_size])

    return batches


if __name__ == "__main__":
    process = False
    if process:
        data = json.load(open('../data/harm4label.json', encoding='utf-8'))
        checkpoint = './output/model/all_wo_test_loss_123/1-6k.pkl'

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # init ours
        config = create_config('../previous_configs/config/train/train.config')
        ours = CrossCaseCL(config).to(device)
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        ours.load_state_dict(checkpoint['model'], strict=False)

        # init baseline
        roberta = AutoModel.from_pretrained('hfl/chinese-roberta-wwm-ext').to(device)

        fact_text_list, fact_label_list = [], []
        evidence_text_list, evidence_label_list = [], []

        for cid, d in enumerate(data):
            for fid, f in enumerate(d['fact']):
                fact_text_list.append(f)
                fact_label_list.append('case{}-fact-{}'.format(cid, fid))
            for eid, e in enumerate(d['evidence']):
                for eid_sub, e_sub in enumerate(e):
                    evidence_text_list.append(e_sub)
                    evidence_label_list.append('case-{}-evidence-{}.{}'.format(cid, eid, eid_sub))

        tokenizer = BertTokenizerFast.from_pretrained('hfl/chinese-roberta-wwm-ext')

        fact_ours, fact_roberta = get_embedding_list(fact_text_list, batch_size=50)
        evidence_ours, evidence_roberta = get_embedding_list(evidence_text_list, batch_size=50)

        with open('./ours.json', 'w', encoding='utf-8') as f:
            json.dump(fact_ours+evidence_ours, f, ensure_ascii=False)

        with open('./roberta.json', 'w', encoding='utf-8') as f:
            json.dump(fact_roberta+evidence_roberta, f, ensure_ascii=False)

        with open('./labels.json', 'w', encoding='utf-8') as f:
            json.dump(fact_label_list+evidence_label_list, f, ensure_ascii=False)
    else:
        ours = json.load(open('../data/ours.json'))
        roberta = json.load(open('../data/roberta.json'))
        labels = json.load(open('../data/labels.json', encoding='utf-8'))

        with open('../data/ours.tsv', 'w', encoding='utf-8') as f:
            for o in ours:
                f.write('\t'.join(map(str, o))+'\n')

        with open('../data/roberta.tsv', 'w', encoding='utf-8') as f:
            for o in roberta:
                f.write('\t'.join(map(str, o))+'\n')

        with open('../data/labels.tsv', 'w', encoding='utf-8') as f:
            for l in labels:
                d_type = 'fact' if 'fact' in l else 'evidence'
                c_no = l.split('-')[0]
                f.write(d_type+'\n')





