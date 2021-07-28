# --- coding:utf-8 ---
# author: Cyberfish time:2021/7/20
import numpy as np
import random
import paddle


def set_seed(seed):
    seed = int(seed)
    np.random.seed(seed)
    random.seed(seed)
    paddle.seed(seed)


def label_process(filename):
    with open(filename) as f:
        slots = f.read().split('\n')

    ner_label2id, ner_id2label = {'O': 0}, {0: 'O'}
    bi_label = []
    ner_n = 0
    for label in slots:
        if label.split('-')[1] != '酒店设施':
            ner_label2id['B_' + label] = 2 * ner_n + 1
            ner_label2id['I_' + label] = 2 * ner_n + 2
            ner_id2label[2 * ner_n + 1] = 'B_' + label
            ner_id2label[2 * ner_n + 2] = 'I_' + label
            ner_n += 1
        else:
            bi_label.append(label)

    slots += ['greet-none', 'thank-none', 'bye-none']
    slots2id = {slot: i for i, slot in enumerate(slots)}
    id2slots = {i: slot for i, slot in enumerate(slots)}

    bi_label2id = {b_l: i for i, b_l in enumerate(bi_label)}
    id2bi_label = {i: b_l for i, b_l in enumerate(bi_label)}
    return ner_label2id, ner_id2label, bi_label2id, id2bi_label, slots2id, id2slots, slots