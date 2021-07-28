# --- coding:utf-8 ---
# author: Cyberfish time:2021/7/22
import paddle
from paddle import set_device, get_device
from biclass_code.bi_utils import model, utils_fn
from paddlenlp.transformers import NeZhaTokenizer

ner_label2id, ner_id2label, bi_label2id, id2bi_label, slots2id, id2slots, slots = utils_fn.label_process('data/slots.txt')


class BiPredict():
    def __init__(self):
        self.bi_model = model.BiModel()
        self.bi_model.to(set_device(get_device()))
        bi_state_dict = paddle.load('weight/bi_model.state')
        self.bi_model.set_state_dict(bi_state_dict)
        self.tokenizer = NeZhaTokenizer.from_pretrained('nezha-base-wwm-chinese')

    def bi_predict(self, content):
        contents = []
        for slot in bi_label2id:
            contents.append([content, '对' + slot + '有要求吗？'])

        embedding = self.tokenizer.batch_encode(contents,
                                                max_seq_len=128,
                                                pad_to_max_seq_len=True
                                                )
        input_ids, token_type_ids = paddle.to_tensor([i['input_ids'] for i in embedding]), \
                                    paddle.to_tensor([i['token_type_ids'] for i in embedding])

        bi_pred = self.bi_model(input_ids, token_type_ids)

        y_pred = paddle.argmax(bi_pred, 1).numpy()

        bis = []
        for i, label in enumerate(y_pred):
            if label == 1:
                bis.append(id2bi_label[i])

        return bis
