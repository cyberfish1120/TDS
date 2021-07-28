# --- coding:utf-8 ---
# author: Cyberfish time:2021/7/20
from ner_utils import config, utils_fn, data_loader, model
from paddle import set_device, get_device
import paddle
import paddle.nn as nn
import time
import numpy
from tqdm import tqdm, trange
from paddlenlp.layers import LinearChainCrfLoss

gpu_id = '1'
localtime = time.localtime(time.time())
ner_label2id, ner_id2label, bi_label2id, id2bi_label, slots2id, id2slots, slots = utils_fn.label_process('../data/slots.txt')


def train():
    ner_model.train()
    one_epoch = trange(ner_train_data.get_batch_num())
    batch_data = ner_train_data.data_iterator()
    for _ in one_epoch:
        _, batch_input_ids, lens, batch_labels = next(batch_data)
        batch_input_ids = paddle.to_tensor([b['input_ids'] for b in batch_input_ids])
        lens = paddle.to_tensor(lens)
        batch_labels = paddle.to_tensor(batch_labels)

        output, pred = ner_model(batch_input_ids, lens)

        ner_pred_loss = ner_loss_fn(output, lens, batch_labels)

        one_epoch.set_postfix(ner_loss=float(paddle.mean(ner_pred_loss)))

        ner_pred_loss.backward()
        opt.step()
        opt.clear_grad()


def output_entity(contents, labels):
    entities = []
    entity = ''
    for content, label in zip(contents, labels):
        if label == 0:
            if entity:
                entities.append(entity)
                entity = ''
            else:
                continue
        else:
            if label % 2 == 1:
                if entity:
                    entities.append(entity)
                entity = ner_id2label[label].split('-')[1]+'\t'+content
            else:
                if entity:
                    entity += content
                else:
                    continue

    return entities


def eval():
    ner_model.eval()
    G, P, S = 1e-10, 1e-10, 1e-10

    one_epoch = trange(ner_test_data.get_batch_num())
    batch_data = ner_test_data.data_iterator()

    for _ in one_epoch:
        batch_content, batch_input_ids, lens, batch_labels = next(batch_data)
        batch_input_ids = paddle.to_tensor([b['input_ids'] for b in batch_input_ids])
        lens = paddle.to_tensor(lens)
        batch_labels = paddle.to_tensor(batch_labels)

        _, y_preds = ner_model(batch_input_ids, lens)

        _len = [int(x) for x in lens]
        y_preds = numpy.array(y_preds)
        y_trues = numpy.array(batch_labels)

        for idx, end in enumerate(_len):
            y_pred = output_entity(batch_content[idx], y_preds[idx][:end])
            y_true = output_entity(batch_content[idx], y_trues[idx][:end])
            y_pred, y_true = set(y_pred), set(y_true)
            G += len(y_true)
            P += len(y_pred)
            S += len(y_true & y_pred)

    f1, precision, recall = 2 * S / (G + P), S / P, S / G
    return f1, precision, recall


if __name__ == '__main__':
    args = config.arg_parse()
    utils_fn.set_seed(args.seed)

    ner_train_data = data_loader.NerData(args.train_file, args, shuffle=True)
    ner_test_data = data_loader.NerData(args.test_file, args)

    args.device = set_device(get_device())
    ner_model = model.NerModel()
    ner_model.to(args.device)

    ner_loss_fn = LinearChainCrfLoss(ner_model.crf)

    opt = paddle.optimizer.AdamW(parameters=ner_model.parameters(), learning_rate=args.lr)

    with open('../result/best_ner_f1.txt') as f:
        best_f1 = float(f.read())

    for epoch in range(1, args.epoch + 1):
        print(f'Epoch: {epoch}/{args.epoch}')
        train()

        f1, precision, recall = eval()
        print(f'f1: {f1}, pre: {precision}, rec: {recall}')
        with open(f'../result/Ner_{localtime.tm_mday}_{localtime.tm_hour}_{localtime.tm_min}.txt', 'a') as f:
            f.write(f'f1: {f1}, pre: {precision}, rec: {recall} \n')

        if f1 > best_f1:
            best_f1 = f1
            with open('../result/best_ner_f1.txt', 'w') as f:
                f.write(str(best_f1))
            paddle.save(ner_model.state_dict(), '../weight/ner_model.state')
            paddle.save(opt.state_dict(), '../weight/ner_opt.state')
