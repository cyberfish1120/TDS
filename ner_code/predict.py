# --- coding:utf-8 ---
# author: Cyberfish time:2021/7/22
import paddle
from paddle import set_device, get_device
from ner_code.ner_utils import model, utils_fn
from paddlenlp.transformers import NeZhaTokenizer

ner_label2id, ner_id2label, bi_label2id, id2bi_label, slots2id, id2slots, slots = utils_fn.label_process('data/slots.txt')


class NerPredict():
    def __init__(self):
        self.ner_model = model.NerModel()
        self.ner_model.to(set_device(get_device()))
        ner_state_dict = paddle.load('weight/ner_model.state')
        self.ner_model.set_state_dict(ner_state_dict)
        self.tokenizer = NeZhaTokenizer.from_pretrained('nezha-base-wwm-chinese')

    def ner_predict(self, content):
        content = [c for c in content]
        input_ids = paddle.to_tensor([self.tokenizer(content, is_split_into_words=True)['input_ids']])
        lens = paddle.to_tensor(len(content))
        _, pred = self.ner_model(input_ids, lens)

        entities = []
        entity = ''
        for content, label in zip(content, pred[0]):
            label = int(label)
            if label == 0:
                if entity:
                    entities.append(entity)
                    entity = ''
                else:
                    continue
            else:
                if label % 2 == 1:
                    if entity:
                        entities.append(entity)
                    entity = ner_id2label[label].split('_')[1] + '\t' + content
                else:
                    if entity:
                        entity += content
                    else:
                        continue

        return entities
