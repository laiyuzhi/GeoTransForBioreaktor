from dataset_bioreaktorMulti import Bioreaktor_Detection
from torch.utils.data import DataLoader, Dataset
from Multi_Model_new import WideResNet
from Multi_Model_noInputLayer import WideResNet as WideResNetNoInput
import torch
import torch.optim as optim
from torch import nn
import visdom
import sys
sys.path.append('/mnt/projects_sdc/lai/GeoTransForBioreaktor/geoTrans')

from utils import Config as cfg
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
import numpy as np
from visdom import Visdom


def main():
    batchsz = cfg.BATCH_SIZE
    lr = cfg.LEARN_RATE
    epochs = cfg.EPOCHS
    num_trans = cfg.NUM_TRANS
    root = "/mnt/data_sdb/datasets/BioreaktorAnomalieDaten/processed/MultiModelAll/DataDistanz400"
    train_db = Bioreaktor_Detection(root, 64, mode='Train')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True,
                            num_workers=0)
    testnormal_db = Bioreaktor_Detection(root, 64, mode='Vali')
    testnormal_loader = DataLoader(testnormal_db, batch_size=batchsz, shuffle=False, num_workers=0)
    testanormal_db = Bioreaktor_Detection(root, 64, mode='Test')
    testanormal_loader = DataLoader(testanormal_db, batch_size=batchsz, shuffle=False, num_workers=0)
    x1, x2, label = iter(testnormal_loader).next()
    print('x1:', x1.shape, 'x2:', x2, 'label:', label.shape)
    x1, x2, label = iter(testanormal_loader).next()
    print('x1:', x1.shape, 'x2:', x2, 'label:', label.shape)
    device = torch.device('cuda')
    criterion = nn.CrossEntropyLoss().to(device)
    criterion_pred = nn.MSELoss().to(device)
    # viz = visdom.Visdom()
    torch.manual_seed(1234)
    model = WideResNet(10, num_trans, 6).to(device)
    # model.load_state_dict(torch.load('/mnt/projects_sdc/lai/GeoTransForBioreaktor/ModelMultiAllRes5_106_0.mdl'))
    # model.eval()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 100], gamma=0.2)
    print(model)

    viz = Visdom()
    viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
    viz.line([[0.0, 0.0]], [0.], win='test_acc', opts=dict(title='normal acc.&anormal acc.',
                                                   legend=['normal acc.', 'anormal acc.']))


    best_epoch, best_acc = 0, 0
    worst_epoch, worst_acc = 0, 1
    global_step = 0
    for epoch in range(int(np.ceil(epochs / num_trans))):
    # for epoch in range(1):
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for batchidx, (x1, x2, label) in pbar:
            # pbar.set_description("Epoch: s%" % str(epoch))
            # [b, 3, 64, 64]
            x1 = x1.to(device)
            x2 = x2.float().view(-1, cfg.INPUT_MULTI).to(device)
            label = label.to(device)

            logits_end, logits_pred, logits_early = model(x1, x2)
            loss1 = criterion(logits_end, label)
            loss2 = criterion(logits_early, label)
            loss3 = criterion(logits_pred, label)
            loss = 0.8 * loss1  + 0.1 * loss2 + 0.1 * loss3
            pred = logits_end.argmax(dim=1)
            correct = torch.eq(pred, label).float().sum().item()

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            viz.line([loss.item()], [global_step], win='train_loss', update='append')

            pbar.set_description(f'Epoch [{epoch}/{int(np.ceil(epochs/num_trans))}]')
            pbar.set_postfix({'loss early=': loss2.item(), 'loss pred=': loss3.item(), 'loss hybrid=': loss1.item(), 'acc=': correct / x1.size(0)})

        print(epoch, 'loss:', loss.item())
        path = 'ModelMultiAll3lossRes_106_' + str(epoch) + '.mdl'
        torch.save(model.state_dict(), path)

        # validation
        if (epoch != 0 and epoch % cfg.VAL_EACH == 0) or epoch == 0:
            total_correct = 0
            total_num = 0
            total_mse = 0
            model.eval()
            with torch.no_grad():

                pbar = tqdm(enumerate(testnormal_loader), total=len(testnormal_loader))
                for batchidx, (x1, x2, label) in pbar:

                    x1, x2, label = x1.to(device), x2.to(device), label.to(device)

                    # [b, 72]
                    logits_end, logits_pred, logits_early = model(x1, x2)

                    # [b]
                    pred = logits_end.argmax(dim=1)
                    # [b] vs [b] => scalar tensor
                    correct = torch.eq(pred, label).float().sum().item()
                    # mse = criterion_pred(logits_pred, x2).float().item()
                    total_correct += correct
                    # total_mse += mse
                    total_num += x1.size(0)



                normalacc = total_correct / total_num

                print(epoch, 'Vali acc:', normalacc)


        ##test
        if (epoch != 0 and epoch % cfg.VAL_EACH == 0) or epoch == 0:
            total_correct_end = 0
            total_correct_early = 0
            total_num = 0
            total_mse = 0
            model.eval()
            with torch.no_grad():
                pbar = tqdm(enumerate(testanormal_loader), total=len(testanormal_loader))
                for batchidx, (x1, x2, label) in pbar:
                    x1, x2, label = x1.float().to(device), x2.to(device), label.float().to(device)

                    # [b, 72]
                    logits_end, logits_pred, logits_early = model(x1, x2)
                    # [b]
                    pred_end = logits_end.argmax(dim=1)
                    pred_early = logits_early.argmax(dim=1)

                    # [b] vs [b] => scalar tensor
                    # mse = criterion_pred(logits_pred, x2).float().item()
                    # total_mse += mse
                    correct_end = torch.eq(pred_end, label).float().sum().item()
                    total_correct_end += correct_end
                    total_num += x1.size(0)
                    correct_early = torch.eq(pred_early, label).float().sum().item()
                    total_correct_early += correct_early
                anormalacc_end = total_correct_end / total_num
                anormalacc_early = total_correct_early / total_num
                print(epoch, 'anormal_end acc:', anormalacc_end, 'anormal:early acc:', anormalacc_early)

        viz.line([[normalacc, anormalacc_end]], [global_step], win='test_acc', update='append')


if __name__ == '__main__':
    main()
