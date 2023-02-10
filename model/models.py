import torch
import torch.nn as nn
import json
import math
import numpy as np
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel
from .utils import init_tokenizer, backbone_plm_dict


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], \
            "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state

        hidden_states = outputs.hidden_states

        if self.pooler_type == "cls":
            return outputs.pooler_output
        elif self.pooler_type == "cls_before_pooler":
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return (last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.pooler = Pooler(config.get('encoder', 'pooling'))
        self.model = AutoModel.from_pretrained(backbone_plm_dict[config.get('encoder', 'backbone')], cache_dir='./data/ckpt/')

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        outputs = self.model(input_ids, attention_mask, token_type_ids)
        output = self.pooler(attention_mask, outputs)
        return output


class AutoEncoder(nn.Module):
    def __init__(self, model_name, config):
        super(AutoEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(backbone_plm_dict[model_name], cache_dir='./data/ckpt/').cuda()
        self.pooler = Pooler(config.get('baseline', 'pooling'))
        self.tokenizer = init_tokenizer(model_name)

    def forward(self, input_text):
        inputs = self.tokenizer(input_text, padding="longest", truncation=True, max_length=128, return_tensors='pt')
        for key in inputs:
            inputs[key] = inputs[key].cuda()
        outputs = self.bert(**inputs)
        output = self.pooler(inputs['attention_mask'], outputs)
        return output

    def search(self, query, corpus):
        query_embedding = self.forward(query)
        corpus_embeddings = self.forward(corpus)
        cos_scores = cosine_similarity(query_embedding.cpu(), corpus_embeddings.cpu())
        return cos_scores


class TFIDFSearcher(object):
    def __init__(self):
        super(TFIDFSearcher, self).__init__()
        self.stopwords = self.load_stopwords_list()
        self.token_dict = json.load(open('./data/token_dict.json', encoding='utf-8'))

        self.tf_dict = defaultdict(dict)
        self.inverted_index = defaultdict(list)
        self.idf_dict = dict()

    def search(self, queries, evidences):
        self.tf_dict, self.inverted_index = self.build_model(evidences)
        self.idf_dict = dict()

        sim_scores = np.zeros((len(queries), len(evidences)))
        for q_id, query in enumerate(queries):
            query_words = [w for w in self.token_dict['fact'][query].split("|") if w not in self.stopwords]

            for e_id, evidence in enumerate(evidences):
                tfidf_score = 0.0

                for word in query_words:
                    if word not in self.tf_dict[e_id]:
                        continue

                    tf = self.tf_dict[e_id][word]

                    if word not in self.idf_dict:
                        # idf(wi) = log[ #Docs. / (#Doc_contain_wi + 1)]
                        idf = math.log(len(evidences) / (len(set(self.inverted_index[word])) + 1))
                        self.idf_dict[word] = idf
                    else:
                        idf = self.idf_dict[word]

                    tfidf_score += tf * idf
                sim_scores[q_id][e_id] = tfidf_score

        return sim_scores

    def build_model(self, evidences):
        """
        Construct Inverted Index, TF lookup tables.
        """
        tf_dict = defaultdict(dict)
        inverted_index_dict = defaultdict(list)
        for e_id, e in enumerate(evidences):
            words = [w for w in self.token_dict['evidence'][e].split("|") if w not in self.stopwords]
            word_count_dic = Counter(words)

            for word in words:
                if not word.strip():
                    continue
                inverted_index_dict[word].append(e_id)
                tf_dict[e_id][word] = word_count_dic[word] / len(words)    # term frequency

        return tf_dict, inverted_index_dict

    @staticmethod
    def load_stopwords_list(stopwords_path='./data/stopwords.txt'):
        stopwords = [line.strip() for line in open(stopwords_path, "r", encoding="utf-8").readlines()]
        return stopwords


class BM25Searcher(object):
    def __init__(self, k1=1.5, k2=1.0, b=0.75):
        super(BM25Searcher, self).__init__()
        self.stopwords = self.load_stopwords_list()
        self.token_dict = json.load(open('./data/token_dict.json', encoding='utf-8'))

        self.tf_dict = defaultdict(dict)
        self.inverted_index = defaultdict(list)
        self.avg_doc_len = 0
        self.eid2words = dict()
        self.w_dict = dict()

        # bm25 hyper-parameters
        self.k1 = k1
        self.k2 = k2
        self.b = b

    def search(self, queries, evidences):
        """
        BM25 Search.
            BM25 Score:
                query = {q1, q2, ..., qn}, where qi is a token/word,
                Documents = {d1, d2, ..., dm}, where dj is a sequence of tokens/words.
                Score(query, d) = sum( Wi * R(qi,d) )| i ranges from 1 to n.
                    Wi = log[ (N - df_i + 0.5) / (df_i + 0.5) ]
                    R(qi, d) = fi*(k1+1) / (fi+K)  *  qf_i*(k2+1) / (qf_i+k2)
                        K = k1 * (1 - b + b * doc_len / avg_doc_len)
        """
        self.inverted_index, self.tf_dict, self.avg_doc_len, self.eid2words = self.build_model(evidences)
        self.w_dict = dict()

        sim_scores = np.zeros((len(queries), len(evidences)))
        for q_id, query in enumerate(queries):
            query_words = [w for w in self.token_dict['fact'][query].split("|") if w not in self.stopwords]

            qf_dict = dict()
            for e_id, e in enumerate(evidences):
                bm25_score = 0.0

                doc_len = len(self.eid2words[e_id])
                K = self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)  # K

                for word in query_words:
                    if word not in self.tf_dict[e_id]:
                        continue

                    # f_i is the freq of qi in the current document
                    f_i = self.tf_dict[e_id][word]

                    # qf_i is the freq of qi in the current query
                    if word not in qf_dict:
                        qf_i = query_words.count(word)
                        qf_dict[word] = qf_i
                    else:
                        qf_i = qf_dict[word]

                    if word not in self.w_dict:
                        # Wi = log[ (N - df_i + 0.5) / (df_i + 0.5) ], where df_i is freq of doc. contains qi
                        N = len(evidences)
                        df_i = len(set(self.inverted_index[word]))
                        wi = math.log(1 + (N - df_i + 0.5) / (df_i + 0.5))

                        self.w_dict[word] = wi
                    else:
                        wi = self.w_dict[word]

                    ri = f_i * (self.k1 + 1) / (f_i + K) * qf_i * (self.k2 + 1) / (qf_i + self.k2)

                    bm25_score += wi * ri
                sim_scores[q_id][e_id] = bm25_score

        return sim_scores

    def build_model(self, evidences):
        """
        Construct Inverted Index, uid2words, TF and IDF lookup tables.
        """
        inverted_index_dict = defaultdict(list)
        tf_dict = defaultdict(dict)
        eid2words = dict()

        doc_len_sum = 0.0
        # build inverted index and tf_dict for words in each document.
        for e_id, e in enumerate(evidences):
            words = [t for t in self.token_dict['evidence'][e].split("|") if t not in self.stopwords]
            eid2words[e_id] = words
            word_count_dic = Counter(words)

            doc_len_sum += len(words)

            for word in words:
                if not word.strip():
                    continue
                inverted_index_dict[word].append(e_id)
                tf_dict[e_id][word] = word_count_dic[word] / len(words)    # term frequency

        avg_doc_len = doc_len_sum / len(evidences)

        return inverted_index_dict, tf_dict, avg_doc_len, eid2words

    @staticmethod
    def load_stopwords_list(stopwords_path='./data/stopwords.txt'):
        stopwords = [line.strip() for line in open(stopwords_path, "r", encoding="utf-8").readlines()]
        return stopwords


class BOE(object):
    def __init__(self, event_dict_file='./data/event_dict.json'):
        super(BOE, self).__init__()
        self.event_dict = json.load(open(event_dict_file, encoding='utf-8'))
        self.candidate_idf_vectors = {}

    def search(self, query, corpus):
        self.candidate_idf_vectors = self.gen_event_idf(query, corpus)
        sim_scores = np.zeros((len(query), len(corpus)))

        for qid, q in enumerate(query):
            q_event_ids = self.event_dict['fact'][q]['event_ids']
            q_vector = self.get_event_vector(q_event_ids, q, corpus)
            for cid, c in enumerate(corpus):
                c_event_ids = self.event_dict['evidence'][c]['event_ids']
                c_vector = self.get_event_vector(c_event_ids, q, corpus)
                sim_scores[qid][cid] = torch.cosine_similarity(q_vector, c_vector)

        return sim_scores

    def get_event_vector(self, event_ids, query, corpus, mode='count'):
        vector = torch.zeros(109)
        for idx in event_ids:
            if mode != 'binary':
                vector[idx] += 1
            else:
                vector[idx] = 1

        # normalized TF
        # vector /= input_ids.shape[0]
        # IDF
        vector *= torch.log(len(corpus) / (self.candidate_idf_vectors[query] + 1))  # comment this line to get vanilla bag-of-event

        return vector.view(1, -1)

    def gen_event_idf(self, queries, corpus):
        candidate_idf_vectors = {}
        for query in queries:
            idf_mat = torch.zeros(109)
            for candi in corpus:
                event_ids = self.event_dict['evidence'][candi]['event_ids']

                idf_single = torch.zeros(109)
                for eid in event_ids:
                    idf_single[eid] = 1

                idf_mat += idf_single

            candidate_idf_vectors[query] = idf_mat
        return candidate_idf_vectors

special_baseline = {
    "tfidf": TFIDFSearcher(),
    "bm25": BM25Searcher(),
    "boe": BOE(),
}


def init_baseline(name, *args, **params):
    if name in special_baseline:
        return special_baseline[name]
    else:
        return AutoEncoder(name, params['config'])

