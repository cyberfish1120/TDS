# --- coding:utf-8 ---
# author: Cyberfish time:2021/7/22
import json
import random


def load_json(filename):
    data = {}
    with open(filename) as f:
        raw_data = json.load(f)
    for line in raw_data:
        data[line[0]] = line[1]
    return data


class DataBase:
    def __init__(self):
        attraction_db = load_json('./data/database/attraction_db.json')
        hotel_db = load_json('./data/database/hotel_db.json')
        restaurant_db = load_json('./data/database/restaurant_db.json')
        metro_db = load_json('./data/database/metro_db.json')
        taxi_db = load_json('./data/database/taxi_db.json')
        self.candidate_n = 3
        self.num2chinese = {1: '一', 2: '两', 3: '三', 4: '四', 5: '五', }

        self.database = {'景点': attraction_db,
                         '酒店': hotel_db,
                         '餐馆': restaurant_db,
                         '地铁': metro_db,
                         '出租': taxi_db,
                         '所有景点名': list(attraction_db.keys()),
                         '所有酒店名': list(hotel_db.keys()),
                         '所有餐馆名': list(restaurant_db.keys())
                         }

    def _grade_precess(self, query_domain, slot, value, candidate,new_candidates):
        parts = value.split('分')
        v = float(parts[0])
        if parts[1] == '':
            if self.database[query_domain][candidate][slot] == v:
                new_candidates.append(candidate)
        else:
            if parts[1] == '以上' and self.database[query_domain][candidate][slot] > v:
                new_candidates.append(candidate)
            elif parts[1] == '以下' and self.database[query_domain][candidate][slot] < v:
                new_candidates.append(candidate)

    def _price_precess(self, query_domain, slot, value, candidate,new_candidates):
        if value == '免费':
            if self.database[query_domain][candidate][slot] == 0:
                new_candidates.append(candidate)
        else:
            parts = value.split('元')
            if parts[1] == '':
                v1, v2 = parts[0].split('-')
                v1, v2 = int(v1), int(v2)
                if v1 <= self.database[query_domain][candidate][slot] <= v2:
                    new_candidates.append(candidate)
            else:
                v = int(parts[0])
                if parts[1] == '以上' and self.database[query_domain][candidate][slot] > v:
                    new_candidates.append(candidate)
                elif parts[1] == '以下' and self.database[query_domain][candidate][slot] < v:
                    new_candidates.append(candidate)

    def database_return(self, content, intent, DST):
        try:
            query_domain, query_slot = intent.split('-')
        except ValueError:
            query_domain, query_slot, bi_slot = intent.split('-')

        requests = []
        if query_domain == '景点':
            if query_slot == '名称':
                """统计查询#景点名称#的条件"""
                for domain_slot, value in DST.items():
                    domain, slot = domain_slot.split('-')
                    if domain == query_domain:
                        if slot == '名称':
                            continue
                        else:
                            requests.append([slot, value])
                """根据统计的条件查询#景点名称#"""
                candidates = self.database['所有' + query_domain + '名']
                for slot, value in requests:
                    new_candidates = []
                    for candidate in candidates:
                        if self.database[query_domain][candidate][slot] is None:
                            continue
                        elif slot == '评分':
                            self._grade_precess(query_domain, slot, list(value)[0], candidate, new_candidates)
                        elif slot == '门票':
                            self._price_precess(query_domain, slot, list(value)[0], candidate, new_candidates)
                        elif slot[:2] == '周边':
                            if set(value) & set(self.database[query_domain][candidate][slot]) == set(value):
                                new_candidates.append(candidate)
                        else:
                            if list(value)[0] in self.database[query_domain][candidate][slot]:
                                new_candidates.append(candidate)
                    candidates = new_candidates
            else:
                name_n = len(DST['景点-名称'])
                if name_n == 0:
                    return '我不太清楚目前想找的景点名，您能和我再说一下想要去哪里吗？'
                elif name_n == 1:
                    name = DST['景点-名称'][0]
                    try:
                        if self.database[query_domain][name][query_slot] is None:
                            return '不好意思，我们目前没有关于' + name + '的' + query_slot + '信息'
                    except KeyError:
                        return '不好意思，我们并没有' + name + '的相关记录'
                    else:
                        DST[intent] = self.database[query_domain][name][query_slot]
                        if isinstance(DST[intent], str):
                            return query_slot + '：' + DST[intent]
                        deng = ''
                        if len(DST[intent]) > self.candidate_n:
                            deng = '等'
                        return query_slot + '有' + '，'.join(DST[intent][:self.candidate_n]) + deng
                else:
                    return ','.join(DST['景点-名称'])+'您更想去哪里玩呢？'

        elif query_domain == '餐馆':
            if query_slot == '名称':
                """统计查询#餐馆名称#的条件"""
                for domain_slot, value in DST.items():
                    domain, slot = domain_slot.split('-')
                    if domain == query_domain:
                        if slot == '名称':
                            continue
                        else:
                            requests.append([slot, value])
                """根据统计的条件查询#餐馆名称#"""
                candidates = self.database['所有' + query_domain + '名']
                for slot, value in requests:
                    new_candidates = []
                    for candidate in candidates:
                        if self.database[query_domain][candidate][slot] is None:
                            continue
                        elif slot == '评分':
                            self._grade_precess(query_domain, slot, list(value)[0], candidate, new_candidates)
                        elif slot == '人均消费':
                            self._price_precess(query_domain, slot, list(value)[0], candidate, new_candidates)
                        elif slot == '推荐菜' or slot[:2] == '周边':
                            if set(value) & set(self.database[query_domain][candidate][slot]) == set(value):
                                new_candidates.append(candidate)
                        else:
                            if list(value)[0] in self.database[query_domain][candidate][slot]:
                                new_candidates.append(candidate)
                    candidates = new_candidates
            else:
                name_n = len(DST['餐馆-名称'])
                if name_n == 0:
                    return '我不太清楚目前想找的餐馆名，您能和我再说一下您想去哪家餐馆吃饭吗？'
                elif name_n == 1:
                    name = DST['餐馆-名称'][0]
                    try:
                        if self.database[query_domain][name][query_slot] is None:
                            return '不好意思，我们目前没有关于' + name + '的' + query_slot + '信息'
                    except KeyError:
                        return '不好意思，我们并没有' + name + '的相关记录'
                    else:
                        DST[intent] = self.database[query_domain][name][query_slot]
                        if isinstance(DST[intent], str):
                            return query_slot + '：' + DST[intent]
                        deng = ''
                        if len(DST[intent]) > self.candidate_n:
                            deng = '等'
                        return query_slot + '有' + '，'.join(DST[intent][:self.candidate_n]) + deng
                else:
                    return ','.join(DST['餐馆-名称'])+'您更想去哪家吃呢？'

        elif query_domain == '酒店':
            if query_slot == '名称':
                """统计查询#酒店名称#的条件"""
                for domain_slot, value in DST.items():
                    domain, slot = domain_slot.split('-')
                    if domain == query_domain:
                        if slot == '名称':
                            continue
                        else:
                            requests.append([slot, value])
                """根据统计的条件查询#餐馆名称#"""
                candidates = self.database['所有' + query_domain + '名']
                for slot, value in requests:
                    new_candidates = []
                    for candidate in candidates:
                        if self.database[query_domain][candidate][slot] is None:
                            continue
                        elif slot == '评分':
                            self._grade_precess(query_domain, slot, list(value)[0], candidate, new_candidates)
                        elif slot == '价格':
                            self._price_precess(query_domain, slot, list(value)[0], candidate, new_candidates)
                        elif slot == '酒店设施' or slot[:2] == '周边':
                            if set(value) & set(self.database[query_domain][candidate][slot]) == set(value):
                                new_candidates.append(candidate)
                        else:
                            if list(value)[0] in self.database[query_domain][candidate][slot]:
                                new_candidates.append(candidate)
                    candidates = new_candidates
            else:
                name_n = len(DST['酒店-名称'])
                if name_n == 0:
                    return '我不太清楚目前想找的酒店名，您能和我再说一下您想去哪家酒店住宿吗？'
                elif name_n == 1:
                    name = DST['酒店-名称'][0]
                    try:
                        if self.database[query_domain][name][query_slot] is None:
                            return '不好意思，我们目前没有关于'+name+'的'+query_slot+'信息'
                    except KeyError:
                        return '不好意思，我们并没有'+name+'的相关记录'
                    else:
                        if query_slot == '酒店设施':
                            if bi_slot in self.database[query_domain][name]['酒店设施']:
                                return '有'+bi_slot
                            else:
                                return '没有' + bi_slot
                        else:
                            DST[intent] = self.database[query_domain][name][query_slot]
                            if isinstance(DST[intent], str):
                                return query_slot+'：'+DST[intent]
                        deng = ''
                        if len(DST[intent]) > self.candidate_n:
                            deng = '等'
                        return query_slot+'有'+'，'.join(DST[intent][:self.candidate_n])+deng
                else:
                    return ','.join(DST['酒店-名称'])+'您更想去哪家酒店呢？'

        elif query_domain == '地铁':
            try:
                candidates = self.database[query_domain][DST[query_domain + '-' + query_slot[:3]]]['地铁']
            except KeyError:
                return '我不太清楚'+query_slot[:3]+'是哪，您能再跟我说一下吗？'
            if candidates is None:
                return DST['地铁'+'-'+query_slot[:3]]+'附近没有地铁站'
            else:
                return DST['地铁'+'-'+query_slot[:3]]+'从'+candidates+'出来'

        else:
            return query_slot+'：'+self.database[query_domain]['出租 ($出发地 - $目的地)'][query_slot]

        if len(candidates) == 0:
            requests = [slot+'：'+value if isinstance(value, str) else slot+'：'+','.join(value) for slot, value in requests]
            return '不好意思，没有查到符合条件的'+query_domain+'，您可以放宽查询条件在试下。您目前的查询要求为：'+';'.join(requests)
        if '一' in content:
            candidates = random.sample(candidates, 1)
            DST[intent] = candidates
        else:
            candidates = candidates[:self.candidate_n]
        measure_word = {'景点': '个', '餐馆': '家', '酒店': '家', }
        return '这边推荐'+','.join(candidates)+'这'+self.num2chinese[len(candidates)]+measure_word[query_domain]+query_domain
