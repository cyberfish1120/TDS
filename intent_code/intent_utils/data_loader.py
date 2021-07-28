# --- coding:utf-8 ---
# author: Cyberfish time:2021/7/19
from paddle.io import Dataset
import json
from intent_utils.utils_fn import label_process
from paddlenlp.transformers import NeZhaTokenizer
from tqdm import tqdm
import random


class IntentData():
    def __init__(self, filename, batch_size, pos_neg=-1, shuffle=False):
        with open(filename) as f:
            self.data = json.load(f)

        self.bs = batch_size

        if pos_neg > 0:
            self.pos, self.neg = [], []
            for id in tqdm(range(len(self.data))):
                if self.data[id]['intent_y'] == 0:
                    self.neg.append(self.data[id])
                else:
                    self.pos.append(self.data[id])
            self.data = random.sample(self.neg, pos_neg*len(self.pos))
            self.data.extend(self.pos)

        self.idx = list(range(len(self.data)))
        if shuffle:
            random.shuffle(self.idx)

    def data_iterator(self):
        batch_input_ids, batch_token_type_ids, batch_labels = [], [], []
        for i in range(len(self.data)):
            id = self.idx[i]
            batch_input_ids.append(self.data[id]['input_ids'])
            batch_token_type_ids.append(self.data[id]['token_type_ids'])
            batch_labels.append(self.data[id]['intent_y'])

            if len(batch_labels) == self.bs or i == len(self.data)-1:
                yield batch_input_ids, batch_token_type_ids, batch_labels
                batch_input_ids, batch_token_type_ids, batch_labels = [], [], []

    def get_batch_num(self):
        if len(self.data) % self.bs == 0:
            BATCH_NUM = len(self.data) // self.bs
        else:
            BATCH_NUM = len(self.data) // self.bs + 1
        return BATCH_NUM


