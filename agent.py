# --- coding:utf-8 ---
# author: Cyberfish time:2021/7/23
from response_code.response_main import rule_response
from collections import defaultdict


class Agent:
    def __init__(self, intent_pre, bi_pre, ner_pre, database):
        self.intent_pre = intent_pre
        self.bi_pre = bi_pre
        self.ner_pre = ner_pre
        self.database = database

        self.DST = defaultdict(set)

    def response(self, content):
        """
        :param content:
        :return:
        """
        intents = self.intent_pre.intent_predict(content)
        bi_entities = self.bi_pre.bi_predict(content)
        ner_entities = self.ner_pre.ner_predict(content)

        if len(bi_entities) != 0:
            self.DST['酒店-酒店设施'] = [bi_entity.split('-')[-1] for bi_entity in bi_entities]
        for entity in ner_entities:
            domain_slot, value = entity.split('\t')

            self.DST[domain_slot] = [value]
            # else:
            #     self.DST[domain_slot].add(value)

        answer, self.DST = rule_response(content, intents, self.database, self.DST)

        # print('意图识别：', intents)
        # print('酒店设施识别：', bi_entities)
        # print('NER识别：', ner_entities)
        # print('DST：', self.DST)

        print(answer)


