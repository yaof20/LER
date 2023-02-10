import torch
import torch.nn as nn
import math
from torch import Tensor as T
from typing import List
from .models import Encoder
from .loss import AttentionLoss, ContrastiveLoss, SimcseLoss


class CrossCaseCL(nn.Module):
    def __init__(self, config):
        super(CrossCaseCL, self).__init__()
        self.config = config
        self.encoder_f, self.encoder_e = self.get_encoder(config)

        # configurations
        self.simcse_loss = SimcseLoss(config)
        self.attn_loss = AttentionLoss(config)
        self.contra_loss = ContrastiveLoss(config)

        self.use_simcse = config.getboolean('simcse_loss', 'use')

    def forward(self, inputs):
        # Get the embeddings for all fact sentences and evidence sentences
        sub_batch_size = self.config.getint('train', 'sub_batch_size')
        embeddings_f = self.get_embedding_list(inputs['inputs_facts'], sub_batch_size, mode='fact')
        embeddings_e = self.get_embedding_list(inputs['inputs_evidences'], sub_batch_size, mode='evidence')

        doc_list_f = self.get_document_level_features(features=embeddings_f, offset=inputs['offset']['fact'])
        doc_list_e = self.get_document_level_features(features=embeddings_e, offset=inputs['offset']['evidence'])

        loss_simcse, loss_contra = None, None

        # SimcseLoss
        if self.config.getboolean('simcse_loss', 'use') and not self.config.getboolean('contra_loss', 'rm_simcse'):
            doc_list_f_double = [d['doc_all'] for d in doc_list_f]
            doc_list_e_double = [d['doc_all'] for d in doc_list_e]
            loss_simcse = self.simcse_loss(query_list=doc_list_f_double, value_list=doc_list_e_double)

        # ContrastiveLoss
        if self.config.getboolean('contra_loss', 'use'):
            query_mode = self.config.get('contra_loss', 'query')
            if query_mode != 'bi-direction':
                loss_contra = 0.0
                pos_outputs = self.get_attn_pos_outputs(doc_fact_list=doc_list_f,
                                                        doc_evidence_list=doc_list_e,
                                                        query_mode=query_mode)

                loss_contra_tmp = self.contra_loss(query_list=pos_outputs['doc_list_q'],
                                                   output_list=pos_outputs['doc_list_a'],
                                                   value_list=pos_outputs['doc_list_v'],
                                                   scores_list=pos_outputs['doc_list_s'])
                loss_contra += loss_contra_tmp

                if self.config.getboolean('contra_loss', 'positive_query'):
                    # do the other directional simcse
                    query_mode = 'fact' if query_mode == 'evidence' else 'evidence'
                    self.config.set('contra_loss', 'positive_attention', 'False')   # do not use the other attention
                    pos_outputs = self.get_attn_pos_outputs(doc_fact_list=doc_list_f,
                                                            doc_evidence_list=doc_list_e,
                                                            query_mode=query_mode)

                    loss_contra_tmp = self.contra_loss(query_list=pos_outputs['doc_list_q'],
                                                       output_list=pos_outputs['doc_list_a'],
                                                       value_list=pos_outputs['doc_list_v'],
                                                       scores_list=pos_outputs['doc_list_s'])
                    loss_contra += loss_contra_tmp

            else:
                loss_contra = 0.0
                for mode in ['fact', 'evidence']:
                    pos_outputs = self.get_attn_pos_outputs(doc_fact_list=doc_list_f,
                                                            doc_evidence_list=doc_list_e,
                                                            query_mode=mode)

                    loss_contra_tmp = self.contra_loss(query_list=pos_outputs['doc_list_q'],
                                                       output_list=pos_outputs['doc_list_a'],
                                                       value_list=pos_outputs['doc_list_v'],
                                                       scores_list=pos_outputs['doc_list_s'])
                    loss_contra += loss_contra_tmp

        # Overall Loss
        overall = 0.0
        for loss in [loss_simcse, loss_contra]:
            if loss is not None:
                overall += loss

        loss = dict(overall=overall, simcse=loss_simcse, contra=loss_contra)

        return loss

    @staticmethod
    def get_encoder(config):
        encoder_f = Encoder(config)
        encoder_e = Encoder(config)

        if config.getboolean('encoder', 'shared'):
            encoder_e = None

        return encoder_f, encoder_e

    def get_embedding_list(self, inputs, sub_batch_size=4, mode='fact'):
        """
        Calculate the embeddings of a large list of sentences.
        """
        output_list = list()

        if mode == 'fact':
            model = self.encoder_f
        else:
            model = self.encoder_e if self.encoder_e else self.encoder_f

        for idx in range(0, len(inputs['input_ids']), sub_batch_size):
            outputs = model(input_ids=inputs['input_ids'][idx: idx+sub_batch_size],
                            attention_mask=inputs['attention_mask'][idx: idx+sub_batch_size])
            output_list.extend(outputs)

        embeddings = torch.stack(output_list, dim=0)

        return embeddings

    def get_document_level_features(self, features: T, offset: list) -> List[dict]:
        """
        Get the features for each document.
        """
        document_features = []
        for oft in offset:
            tmp = dict()
            tmp['doc_all'] = features[oft[0]: oft[1]]
            if self.use_simcse:
                tmp['doc_raw'] = features[oft[0]: oft[1]][0::2]   # raw features
                tmp['doc_drop'] = features[oft[0]: oft[1]][1::2]  # dropout features

            document_features.append(tmp)

        return document_features

    @staticmethod
    def attention(queries: T, keys: T, values: T) -> dict:
        """ Attention (dot product or cosine) """

        dot_product = torch.matmul(queries, keys.permute(1, 0))
        attention_scores = torch.softmax(dot_product, dim=-1)
        outputs = torch.matmul(attention_scores, values)

        sim_scores = dict(dot=attention_scores)
        # sim_scores = dict(dot=dot_product)

        return {'attn_outputs': outputs, 'attn_scores': attention_scores, 'sim_scores': sim_scores}

    def get_attn_pos_outputs(self, doc_fact_list, doc_evidence_list, query_mode='evidence'):
        # q: query, v: value, a: attention output, s: attention scores
        outputs = dict(doc_list_q=dict(single=[], double=[]),
                       doc_list_v=dict(single=[], double=[]),
                       doc_list_a=list(),
                       doc_list_s=list())

        contra_query_list = doc_fact_list if query_mode == 'fact' else doc_evidence_list
        contra_value_list = doc_evidence_list if query_mode == 'fact' else doc_fact_list

        for doc_id in range(len(doc_fact_list)):
            single_key = 'doc_raw' if self.use_simcse else 'doc_all'
            queries = contra_query_list[doc_id][single_key]
            values = contra_value_list[doc_id][single_key]

            outputs['doc_list_q']['single'].append(queries)
            outputs['doc_list_v']['single'].append(values)
            if self.use_simcse:
                outputs['doc_list_q']['double'].append(contra_query_list[doc_id]['doc_all'])
                outputs['doc_list_v']['double'].append(contra_value_list[doc_id]['doc_all'])

            attn_output = self.attention(queries=queries, keys=values, values=values)
            outputs['doc_list_a'].append(attn_output['attn_outputs'])
            outputs['doc_list_s'].append(attn_output['sim_scores'])

        return outputs
