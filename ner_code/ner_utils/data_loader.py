# --- coding:utf-8 ---
# author: Cyberfish time:2021/7/19
from paddle.io import Dataset
import json
from paddlenlp.transformers import NeZhaTokenizer
from tqdm import tqdm
import random


class NerData():
    def __init__(self, filename, args, shuffle=False):
        with open(filename) as f:
            self.data = json.load(f)

        self.idx = list(range(len(self.data)))
        self.bs = args.batch_size
        self.max_len = args.max_len
        self.tokenizer = NeZhaTokenizer.from_pretrained('nezha-base-wwm-chinese')

        if shuffle:
            random.shuffle(self.idx)

    def data_iterator(self):
        batch_content, lens, batch_labels = [], [], []
        for i in range(len(self.data)):
            id = self.idx[i]
            batch_content.append([w for w in self.data[id]['content']])
            pad_labels = self.data[id]['ner_labels'] + ([0]*(self.max_len-len(self.data[id]['ner_labels'])))
            lens.append(len(batch_content[-1]))
            batch_labels.append(pad_labels)
            # assert (len(batch_content[-1]) == len(batch_labels[-1])), 'NER标签和字对不上'

            if len(batch_labels) == self.bs or i == len(self.data)-1:
                batch_input_ids = self.tokenizer.batch_encode(batch_content,
                                                              is_split_into_words=True,
                                                              max_seq_len=self.max_len,
                                                              pad_to_max_seq_len=True,
                                                              return_token_type_ids=False
                                                              )
                yield batch_content, batch_input_ids, lens, batch_labels
                batch_content, lens, batch_labels = [], [], []

    def get_batch_num(self):
        if len(self.data) % self.bs == 0:
            BATCH_NUM = len(self.data) // self.bs
        else:
            BATCH_NUM = len(self.data) // self.bs + 1
        return BATCH_NUM


