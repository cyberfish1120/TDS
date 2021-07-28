# --- coding:utf-8 ---
# author: Cyberfish time:2021/7/19
import paddle.nn as nn
from paddlenlp.transformers import NeZhaModel


class IntentModel(nn.Layer):
    def __init__(self):
        super().__init__()

        self.nezha = NeZhaModel.from_pretrained('nezha-base-wwm-chinese')
        self.dropout = nn.Dropout(0.1)
        self.mlp = nn.Linear(768, 2)

    def forward(self, input_ids, token_type_ids):
        cls_out = self.nezha(input_ids=input_ids, token_type_ids=token_type_ids)[1]
        output = self.dropout(cls_out)
        output = self.mlp(output)

        return output