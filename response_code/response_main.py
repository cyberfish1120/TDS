# --- coding:utf-8 ---
# author: Cyberfish time:2021/7/23
def rule_response(content, intents, database, DST):
    greet, thank, bye = '', '', ''
    responds = set()
    if len(intents) == 0:
        return '实在是不好意思，我没听懂您想问啥，抱歉！', DST
    for intent in intents:
        if intent == 'greet-none':
            greet = '你好！'
        elif intent == 'thank-none':
            thank = '不用客气！'
        elif intent == 'bye-none':
            bye = '再见!'
        else:
            responds.add(database.database_return(content, intent, DST))

    if responds:
        sequence = greet + thank + ';'.join(responds) + bye + '。'
        if sequence[-2] == '？':
            sequence = sequence[:-1]
    else:
        sequence = greet + thank + bye
    return sequence, DST
