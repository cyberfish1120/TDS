# --- coding:utf-8 ---
# author: Cyberfish time:2021/7/23
from ner_code.predict import NerPredict
from intent_code.predict import IntentPredict
from biclass_code.predict import BiPredict
from database_code.database_main import DataBase
from agent import Agent
from collections import defaultdict

intent_predict = IntentPredict()
bi_predict = BiPredict()
ner_predict = NerPredict()

database = DataBase()

agent = Agent(intent_predict, bi_predict, ner_predict, database)

debug_data = []

with open('test_contents.txt') as f:
    f = f.read().split('\n\n\n')

for part in f[:-1]:
    line = part.split('\n')
    debug_data.append([l.split('\t')[1] for l in line[1:] if l.split('\t')[0] == '用户：'])

start_id = 5

for i, contents in enumerate(debug_data[start_id:]):
    id = i + start_id
    for content in contents:
        agent.response(content)
    agent.DST = defaultdict(set)
    print('已重置用户对话状态')



