# --- coding:utf-8 ---
# author: Cyberfish time:2021/7/19
import paddle.nn as nn
from intent_utils import data_loader, config, utils_fn, model
from tqdm import tqdm, trange
from paddle.device import get_device, set_device
import paddle
import numpy
from paddle.io import DataLoader
import time

gpu_id = '0'
localtime = time.localtime(time.time())


def train():
    intent_model.train()
    one_epoch = trange(intent_train_data.get_batch_num())
    batch_data = intent_train_data.data_iterator()
    for _ in one_epoch:
        batch_input_ids, batch_token_type_ids, batch_labels = next(batch_data)
        batch_input_ids = paddle.to_tensor(batch_input_ids)
        batch_token_type_ids = paddle.to_tensor(batch_token_type_ids)
        batch_labels = paddle.to_tensor(batch_labels)

        intent_pred = intent_model(batch_input_ids, batch_token_type_ids)

        intent_pred_loss = intent_loss_fn(intent_pred, batch_labels)

        one_epoch.set_postfix(intent_loss=float(intent_pred_loss))

        intent_pred_loss.backward()
        opt.step()
        opt.clear_grad()


def eval():
    intent_model.eval()
    G, P, S = 1e-10, 1e-10, 1e-10

    one_epoch = trange(intent_test_data.get_batch_num())
    batch_data = intent_test_data.data_iterator()

    for _ in one_epoch:
        batch_input_ids, batch_token_type_ids, batch_labels = next(batch_data)
        batch_input_ids = paddle.to_tensor(batch_input_ids)
        batch_token_type_ids = paddle.to_tensor(batch_token_type_ids)
        batch_labels = paddle.to_tensor(batch_labels)

        intent_pred = intent_model(batch_input_ids, batch_token_type_ids)

        y_pred = paddle.argmax(intent_pred, 1).numpy()
        y_true = numpy.array(batch_labels)
        G += (y_true == 1).sum()
        P += (y_pred == 1).sum()
        S += (y_pred & y_true).sum()

    f1, precision, recall = 2 * S / (G + P), S / P, S / G
    return f1, precision, recall


if __name__ == '__main__':
    args = config.arg_parse()
    utils_fn.set_seed(args.seed)

    intent_train_data = data_loader.IntentData(args.test_file, batch_size=args.batch_size, pos_neg=-1, shuffle=True)
    intent_test_data = data_loader.IntentData(args.train_file, batch_size=args.batch_size//2, pos_neg=-1)

    args.device = set_device(get_device())
    intent_model = model.IntentModel()
    intent_model.to(args.device)

    intent_loss_fn = nn.CrossEntropyLoss()

    opt = paddle.optimizer.AdamW(parameters=intent_model.parameters(), learning_rate=args.lr)

    with open('../result/best_intent_f1.txt') as f:
        best_f1 = float(f.read())

    for epoch in range(1, args.epoch + 1):
        print(f'Epoch: {epoch}/{args.epoch}')
        train()

        f1, precision, recall = eval()
        print(f'f1: {f1}, pre: {precision}, rec: {recall}')
        with open(f'../result/Intent_{localtime.tm_mday}_{localtime.tm_hour}_{localtime.tm_min}.txt', 'a') as f:
            f.write(f'f1: {f1}, pre: {precision}, rec: {recall} \n')

        if f1 > best_f1:
            best_f1 = f1
            with open('../result/best_intent_f1.txt', 'w') as f:
                f.write(str(best_f1))
            paddle.save(intent_model.state_dict(), '../weight/intent_model.state')
            paddle.save(opt.state_dict(), '../weight/intent_opt.state')


