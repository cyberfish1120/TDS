# --- coding:utf-8 ---
# author: Cyberfish time:2021/7/20
import paddle.nn as nn
import paddle
from paddlenlp.transformers import NeZhaModel
from paddlenlp.layers import LinearChainCrf, ViterbiDecoder


class NerModel(nn.Layer):
    def __init__(self):
        super().__init__()

        self.nezha = NeZhaModel.from_pretrained('nezha-base-wwm-chinese')
        # self.dropout = nn.Dropout(0.1)
        self.mlp = nn.Linear(768, 71+2)
        self.crf = LinearChainCrf(71)
        self.decoder = ViterbiDecoder(self.crf.transitions)

    def forward(self, input_ids, lens):
        seq_out = self.nezha(input_ids=input_ids)[0]
        output = self.mlp(seq_out)
        _, pred = self.decoder(output, lens)
        return output, pred
