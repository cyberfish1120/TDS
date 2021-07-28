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

while True:
    content = input('请输入内容： ')
    if content == 'clear':
        agent.DST = defaultdict(set)
        print('已重置用户对话状态')
    else:
        agent.response(content)

    # agent.response('好的，麻烦你帮我查一下桔子水晶酒店(北京安贞店)电话呗。')
    # agent.response('营业时间是什么时间？')

    # agent.response('关于酒店没有其他问题了。我想去八达岭长城游玩，麻烦你告诉我这个景点的门票和电话。')
    # agent.response('我如果坐出租车，从北京京泰龙国际大酒店到北京杜莎夫人蜡像馆，能查到车型和车牌信息吗？')
    # agent.response('好的，没有其他问题了，谢谢。')
    # agent.response('收到，非常感谢！')



