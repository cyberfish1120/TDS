'''
Author: your name
Date: 2021-07-23 08:45:46
LastEditTime: 2021-07-23 10:50:28
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /PyCode/project_demo/remotegit/wx-duihua/model/api.py
'''
from flask import Flask
from flask.globals import request
from flask.json import jsonify
from flask_cors import CORS

from ner_code.predict import NerPredict
from intent_code.predict import IntentPredict
from biclass_code.predict import BiPredict
from database_code.database_main import DataBase
from response_code.response_main import rule_response
from collections import defaultdict

app = Flask(__name__)
CORS(app, supports_credentials=True)
#app.debug=True
'''
测试接口
'''
@app.route("/test",methods=['GET'])
def test():
    return "ok"


'''
定义对话系统web接口
'''
@app.route("/sendmsg",methods=['POST'])
def sendMsg2Robot():
    req = request.form.get('content')
    data1, data2, data3, data4, resp = FitAndPredict(req)
    return jsonify({
        "data":{
            "data1": data1,
            "data2": data2,
            "data3": data3,
            "data4": data4,
            "resp": resp
        }
    })

'''
模型入口
'''
intent_pre = IntentPredict()
bi_pre = BiPredict()
ner_pre = NerPredict()

database = DataBase()
DST = defaultdict(set)


def FitAndPredict(content):
    global DST
    print(content)
    if content == 'clear':
        DST = defaultdict(set)
        return '', '', '', '', '用户对话状态已重置，可以开启新一轮对话！'
    else:
        intents = intent_pre.intent_predict(content)
        bi_entities = bi_pre.bi_predict(content)
        ner_entities = ner_pre.ner_predict(content)

        if len(bi_entities) != 0:
            DST['酒店-酒店设施'] = [bi_entity.split('-')[-1] for bi_entity in bi_entities]
        for entity in ner_entities:
            domain_slot, value = entity.split('\t')

            DST[domain_slot] = [value]

        answer, DST = rule_response(content, intents, database, DST)


        intents = '\n'.join(intents)
        bi_entities = '\n'.join(bi_entities)
        ner_entities = '\n'.join(ner_entities)

        str_DST = ''
        for key, value in DST.items():
            if isinstance(value, str):
                str_DST += key + '：' + value + '\n'
            else:
                str_DST += key+'：'+','.join(value)+'\n'
        if intents == '':
            intents == '无意图识别结果'
        if bi_entities == '':
            bi_entities = '无酒店设施要求'
        if ner_entities == '':
            ner_entities = '无NER识别结果'
        return intents, bi_entities, ner_entities, str_DST, answer


if __name__ == '__main__':
    webhost = "0.0.0.0"
    webport = 5000
    app.run(host=webhost, port=webport)
