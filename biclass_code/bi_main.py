# --- coding:utf-8 ---
# author: Cyberfish time:2021/7/20
from bi_utils import config, utils_fn, data_loader, model
from paddle import set_device, get_device
import paddle
import paddle.nn as nn
import time
import numpy
from tqdm import tqdm, trange

gpu_id = '1'
localtime = time.localtime(time.time())


def train():
    bi_model.train()
    one_epoch = trange(bi_train_data.get_batch_num())
    batch_data = bi_train_data.data_iterator()
    for _ in one_epoch:
        batch_input_ids, batch_token_type_ids, batch_labels = next(batch_data)
        batch_input_ids = paddle.to_tensor(batch_input_ids)
        batch_token_type_ids = paddle.to_tensor(batch_token_type_ids)
        batch_labels = paddle.to_tensor(batch_labels)

        bi_pred = bi_model(batch_input_ids, batch_token_type_ids)

        bi_pred_loss = bi_loss_fn(bi_pred, batch_labels)

        one_epoch.set_postfix(bi_loss=float(bi_pred_loss))

        bi_pred_loss.backward()
        opt.step()
        opt.clear_grad()


def eval():
    bi_model.eval()
    G, P, S = 1e-10, 1e-10, 1e-10

    one_epoch = trange(bi_test_data.get_batch_num())
    batch_data = bi_test_data.data_iterator()

    for _ in one_epoch:
        batch_input_ids, batch_token_type_ids, batch_labels = next(batch_data)
        batch_input_ids = paddle.to_tensor(batch_input_ids)
        batch_token_type_ids = paddle.to_tensor(batch_token_type_ids)
        batch_labels = paddle.to_tensor(batch_labels)

        bi_pred = bi_model(batch_input_ids, batch_token_type_ids)

        y_pred = paddle.argmax(bi_pred, 1).numpy()
        y_true = numpy.array(batch_labels)
        G += (y_true == 1).sum()
        P += (y_pred == 1).sum()
        S += (y_pred & y_true).sum()

    f1, precision, recall = 2 * S / (G + P), S / P, S / G
    return f1, precision, recall


if __name__ == '__main__':
    args = config.arg_parse()
    utils_fn.set_seed(args.seed)

    bi_train_data = data_loader.BiData(args.train_file, batch_size=args.batch_size, pos_neg=10, shuffle=True)
    bi_test_data = data_loader.BiData(args.test_file, batch_size=args.batch_size//2, pos_neg=100)

    args.device = set_device(get_device())
    bi_model = model.BiModel()
    bi_model.to(args.device)

    # bi_loss_fn = nn.CrossEntropyLoss(weight=paddle.to_tensor([1.0, 30.0]))
    bi_loss_fn = nn.CrossEntropyLoss()

    opt = paddle.optimizer.AdamW(parameters=bi_model.parameters(), learning_rate=args.lr)

    with open('../result/best_bi_f1.txt') as f:
        best_f1 = float(f.read())

    for epoch in range(1, args.epoch + 1):
        print(f'Epoch: {epoch}/{args.epoch}')
        train()

        f1, precision, recall = eval()
        print(f'f1: {f1}, pre: {precision}, rec: {recall}')
        with open(f'../result/Bi_{localtime.tm_mday}_{localtime.tm_hour}_{localtime.tm_min}.txt', 'a') as f:
            f.write(f'f1: {f1}, pre: {precision}, rec: {recall} \n')

        if f1 > best_f1:
            best_f1 = f1
            with open('../result/best_bi_f1.txt', 'w') as f:
                f.write(str(best_f1))
            paddle.save(bi_model.state_dict(), '../weight/bi_model.state')
            paddle.save(opt.state_dict(), '../weight/bi_opt.state')