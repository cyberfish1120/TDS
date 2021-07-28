# --- coding:utf-8 ---
# author: Cyberfish time:2021/7/22
import paddle
from paddle import set_device, get_device
from intent_code.intent_utils import model, utils_fn
from paddlenlp.transformers import NeZhaTokenizer

ner_label2id, ner_id2label, bi_label2id, id2bi_label, slots2id, id2slots, slots = utils_fn.label_process('data/slots.txt')


class IntentPredict():
    def __init__(self):
        self.intent_model = model.IntentModel()
        self.intent_model.to(set_device(get_device()))
        intent_state_dict = paddle.load('weight/intent_model.state')
        self.intent_model.set_state_dict(intent_state_dict)
        self.tokenizer = NeZhaTokenizer.from_pretrained('nezha-base-wwm-chinese')

    def intent_predict(self, content):
        contents = []
        for slot in slots:
            contents.append([content, '有在问' + slot + '吗？'])

        embedding = self.tokenizer.batch_encode(contents,
                                                max_seq_len=128,
                                                pad_to_max_seq_len=True
                                                )
        input_ids, token_type_ids = paddle.to_tensor([i['input_ids'] for i in embedding]), \
                                    paddle.to_tensor([i['token_type_ids'] for i in embedding])

        intent_pred = self.intent_model(input_ids, token_type_ids)

        y_pred = paddle.argmax(intent_pred, 1).numpy()

        intents = []
        for i, label in enumerate(y_pred):
            if label == 1:
                intents.append(id2slots[i])

        return intents
