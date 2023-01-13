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
    root = '/mnt/data_sdb/datasets/BioreaktorAnomalieDaten/processed/MultiModelAll'
    train_db = Bioreaktor_Detection(root, 64, mode='Train')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True,
                            num_workers=0)
    testnormal_db = Bioreaktor_Detection(root, 64, mode='Vali')
    testnormal_loader = DataLoader(testnormal_db, batch_size=batchsz, shuffle=True,
                                num_workers=0)
    testanormal_db = Bioreaktor_Detection(root, 64, mode='Test')
    testanormal_loader = DataLoader(testanormal_db, batch_size=batchsz, shuffle=True,
                                num_workers=0)

    x1, x2, label = iter(testanormal_loader).next()
    print('x2:', x2, 'label:', label.shape)

    device = torch.device('cuda')
    criterion = nn.CrossEntropyLoss().to(device)
    # viz = visdom.Visdom()
    torch.manual_seed(1234)
    model = WideResNet(10, num_trans, 6).to(device)
    # if os.path.exists('/mnt/projects_sdc/lai/GeoTransForBioreaktor/ModelMultiAllNewRes100.mdl'):
    #     model.load_state_dict(torch.load('/mnt/projects_sdc/lai/GeoTransForBioreaktor/ModelMultiAllNewRes100.mdl'))
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
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))

        for batchidx, (x1, x2, label) in pbar:
            # pbar.set_description("Epoch: s%" % str(epoch))
            # [b, 3, 64, 64]
            x1 = x1.to(device)
            x2 = x2.float().view(-1, cfg.INPUT_MULTI).to(device)
            label = label.to(device)

            logits = model(x1, x2)
            loss = criterion(logits, label)

            pred = logits.argmax(dim=1)
            correct = torch.eq(pred, label).float().sum().item()

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            viz.line([loss.item()], [global_step], win='train_loss', update='append')

            pbar.set_description(f'Epoch [{epoch}/{int(np.ceil(epochs/num_trans))}]')
            pbar.set_postfix({'loss=': loss.item(), 'acc=': correct / x1.size(0)})

        print(epoch, 'loss:', loss.item())
        path = 'ModelMultiAllNewRes10' + str(epoch) + '.mdl'
        torch.save(model.state_dict(), path)

        # validation
        if (epoch != 0 and epoch % cfg.VAL_EACH == 0) or epoch == 0:
            total_correct = 0
            total_num = 0
            model.eval()
            with torch.no_grad():
                pbar = tqdm(enumerate(testnormal_loader), total=len(testnormal_loader))
                for batchidx, (x1, x2, label) in pbar:
                    x1, x2, label = x1.to(device), x2.float().view(-1, cfg.INPUT_MULTI).to(device), label.to(device)

                    # [b, 72]
                    logits = model(x1, x2)
                    # [b]
                    pred = logits.argmax(dim=1)
                    # [b] vs [b] => scalar tensor
                    correct = torch.eq(pred, label).float().sum().item()
                    total_correct += correct
                    total_num += x1.size(0)

                normalacc = total_correct / total_num
                print(epoch, 'Vali acc:', normalacc)


        ##test
        if (epoch != 0 and epoch % cfg.VAL_EACH == 0) or epoch == 0:
            total_correct = 0
            total_num = 0
            model.eval()
            with torch.no_grad():
                pbar = tqdm(enumerate(testanormal_loader), total=len(testanormal_loader))
                for batchidx, (x1, x2, label) in pbar:
                    x1, x2, label = x1.float().to(device), x2.float().view(-1, cfg.INPUT_MULTI).to(device), label.float().to(device)

                    # [b, 72]
                    logits = model(x1, x2)
                    # [b]
                    pred = logits.argmax(dim=1)
                    # [b] vs [b] => scalar tensor
                    correct = torch.eq(pred, label).float().sum().item()
                    total_correct += correct
                    total_num += x1.size(0)

                anormalacc = total_correct / total_num
                print(epoch, 'anormal acc:', anormalacc)

        viz.line([[normalacc, anormalacc]], [global_step], win='test_acc', update='append')


if __name__ == '__main__':
    main()
