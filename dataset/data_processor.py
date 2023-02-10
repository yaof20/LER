import copy
import os.path
import random

from torch.utils.data import Dataset
import jsonlines
from tqdm import tqdm
import torch
from collections import defaultdict
import json


class InputExample(object):
    """
    An example for a document, containing:
        a list of single facts:  [f1, f2, ..., fn]
        a list of single evidences doc: [[e1, e2], [e3, e4], [e5, e6]]
    """
    def __init__(self, evidences, facts):
        self.evidence_doc_list = evidences
        self.fact_sent_list = facts


class InputFeature(object):
    """
    A feature for a document, containing:
        for simcse:
            a list of doubled fact bert inputs: [f1, f1, f2, f2, ..., fn, fn]
            a list of doubled evidence bert inputs: [[e1, e1, e2, e2], [e3, e3, e4, e4], [e5, e5, e6, e6]]
        for non-simcse:
            a list of single fact bert inputs: [f1, f2, ..., fn]
            a list of single evidence bert inputs: [[e1, e2], [e3, e4], [e5, e6]]

    """
    def __init__(self, inputs_facts, inputs_evidences):
        self.inputs_facts = inputs_facts
        self.inputs_evidences = inputs_evidences


class DataProcessor(Dataset):
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.input_examples = None
        self.input_features = None

    def read_example(self, input_file):
        raise NotImplementedError

    def convert_examples_to_features(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.input_features)

    def __getitem__(self, index):
        features = self.input_features[index]

        return features


class CLProcessor(DataProcessor):
    def __init__(self, config, tokenizer, input_file):
        super(CLProcessor, self).__init__(config, tokenizer)
        self.config = config
        self.double_feature = config.getboolean('simcse_loss', 'use')
        self.input_examples = self.read_example(input_file)
        self.input_features = self.convert_examples_to_features()

    def read_example(self, input_file):
        examples = []
        for case in jsonlines.open(input_file):
            example = InputExample(evidences=case['evidence'], facts=case['fact'])
            examples.append(example)

        return examples

    def convert_examples_to_features(self):
        feature_num = 'double' if self.double_feature else 'single'
        token_type = self.config.get('encoder', 'backbone') + "_" + feature_num

        data_path = self.config.get('data', 'train_data')
        cache_path = data_path.replace('train/', 'train_cache/').replace('.jsonl', '-{}.jsonl'.format(token_type))

        features = []
        if os.path.exists(cache_path):
            inputs_list = jsonlines.open(cache_path)
            for inputs in tqdm(inputs_list, desc='Loading from {}'.format(cache_path)):
                feature = InputFeature(inputs_facts=inputs['fact'], inputs_evidences=inputs['evidence'])
                features.append(feature)
        else:
            with jsonlines.open(cache_path, 'w') as f:
                for example in tqdm(self.input_examples, desc='Converting examples to features'):
                    fact_sent_list = copy.deepcopy(example.fact_sent_list)
                    evidence_doc_list = copy.deepcopy(example.evidence_doc_list)

                    if self.double_feature:
                        # double the inputs for simcse here
                        fact_sent_list = sum([list(f) for f in zip(fact_sent_list, fact_sent_list)], [])
                        for e_id, evidence_doc in enumerate(evidence_doc_list):
                            evidence_doc_list[e_id] = sum([list(e) for e in zip(evidence_doc, evidence_doc)], [])

                    inputs_facts = self.tokenizer(fact_sent_list, padding='max_length', truncation=True, max_length=128)
                    inputs_evidences = defaultdict(list)
                    for evi_doc in evidence_doc_list:
                        evi_inputs = self.tokenizer(evi_doc, padding='max_length', truncation=True, max_length=128)
                        for key in evi_inputs:
                            inputs_evidences[key].append(evi_inputs[key])

                    if 'token_type_ids' in inputs_facts and 'token_type_ids' in inputs_evidences:
                        del inputs_facts['token_type_ids']
                        del inputs_evidences['token_type_ids']

                    jsonlines.Writer.write(f, dict(fact=dict(inputs_facts), evidence=dict(inputs_evidences)))
                    features.append(InputFeature(inputs_facts=inputs_facts, inputs_evidences=inputs_evidences))

            print('finished caching.')
        return features

    def collate_fn(self, batch):
        batch_copy = copy.deepcopy(batch)
        output_batch = dict()

        inputs_facts = {'input_ids': [], 'attention_mask': []}
        inputs_evidences = {'input_ids': [], 'attention_mask': []}
        offset = {'fact': [], 'evidence': [], 'evidence_inner': []}

        st1, st2 = 0, 0
        for i, b in enumerate(batch_copy):
            if isinstance(b.inputs_evidences['input_ids'][0], list):
                # sample strategy: random sample two records and concatenate together as training evidence list
                record_num = len(b.inputs_evidences['input_ids'])
                sample_num = self.config.getint('contra_loss', 'value_sample_num')
                indices = random.sample(range(record_num), min(sample_num, record_num))   # random sample 2 records
                tmp = defaultdict(list)
                for key in b.inputs_evidences:
                    for idx in indices:
                        tmp[key] += b.inputs_evidences[key][idx]
                evidence = tmp
            else:
                evidence = b.inputs_evidences

            ed1 = st1 + len(b.inputs_facts['input_ids'])
            ed2 = st2 + len(evidence['input_ids'])

            offset['fact'].append([st1, ed1])
            offset['evidence'].append([st2, ed2])

            st1, st2 = ed1, ed2

            for key in ['input_ids', 'attention_mask']:
                inputs_facts[key].extend(b.inputs_facts[key])
                inputs_evidences[key].extend(evidence[key])

        for key in ['input_ids', 'attention_mask']:
            inputs_facts[key] = torch.tensor(inputs_facts[key], dtype=torch.long)
            inputs_evidences[key] = torch.tensor(inputs_evidences[key], dtype=torch.long)

        output_batch['inputs_facts'] = inputs_facts
        output_batch['inputs_evidences'] = inputs_evidences
        output_batch['offset'] = offset

        return output_batch
