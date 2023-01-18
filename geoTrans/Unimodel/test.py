from dataset_bioreaktorSpeed import Bioreaktor_Detection
from dataset_bioreaktor import Bioreaktor_Detection as Luft
from torch.utils.data import DataLoader, Dataset
from Model import WideResNet
import torch
import torch.optim as optim
from torch import nn
import visdom
from utils import Config as cfg
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
import numpy as np

def testSpeed():
    root = '/mnt/data_sdb/datasets/BioreaktorAnomalieDaten/processed/unimodelSpeedData2'


    batchsz = cfg.BATCH_SIZE
    num_trans = cfg.NUM_TRANS
    testbig_db = Bioreaktor_Detection(root, 64, mode='testbig')
    testsmall_db = Bioreaktor_Detection(root, 64, mode='testsmall')
    vali_db = Bioreaktor_Detection(root, 64, mode='vali')
    vali_loader = DataLoader(vali_db, batch_size=128, num_workers=0)
    testbig_loader = DataLoader(testbig_db, batch_size=batchsz, num_workers=0)
    testsmall_loader = DataLoader(testsmall_db, batch_size=batchsz, num_workers=0)
    x, label = iter(vali_loader).next()
    print('x:', x.shape, 'label:', label.shape)

    device = torch.device('cuda')
    # viz = visdom.Visdom()
    torch.manual_seed(1234)
    model = WideResNet(16, num_trans, 8).to(device)
    model.load_state_dict(torch.load('/mnt/projects_sdc/lai/GeoTransForBioreaktor/geoTrans/mdl/modelspeedfordata2.mdl'))

    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 100], gamma=0.2)
    print(model)

    best_epoch, best_acc = 0, 0



        ##test

    total_correct = 0
    total_num = 0
    model.eval()
#     with torch.no_grad():
#         pbar = tqdm(enumerate(testbig_loader), total=len(testbig_loader))
#         for batchidx, (x, label) in pbar:
#             x, label = x.to(device), label.to(device)

#             # [b, 72]
#             logits = model(x)
#             # [b]
#             pred = logits.argmax(dim=1)
#             # [b] vs [b] => scalar tensor
#             correct = torch.eq(pred, label).float().sum().item()
#             total_correct += correct
#             total_num += x.size(0)

#         acc = total_correct / total_num
#         print('anormalbig acc:', acc)
# # scheduler.step()

    total_correct = 0
    total_num = 0
    model.eval()
    with torch.no_grad():
        pbar = tqdm(enumerate(testsmall_loader), total=len(testsmall_loader))
        for batchidx, (x, label) in pbar:
            x, label = x.to(device), label.to(device)

            # [b, 72]
            logits = model(x)
            # [b]
            pred = logits.argmax(dim=1)
            # [b] vs [b] => scalar tensor
            correct = torch.eq(pred, label).float().sum().item()
            total_correct += correct
            total_num += x.size(0)

        acc = total_correct / total_num
        print('anormalsmall acc:', acc)
    # scheduler.step()

    total_correct = 0
    total_num = 0
    model.eval()
    with torch.no_grad():
        pbar = tqdm(enumerate(vali_loader), total=len(vali_loader))
        for batchidx, (x, label) in pbar:
            x, label = x.to(device), label.to(device)

            # [b, 72]
            logits = model(x)
            # [b]
            pred = logits.argmax(dim=1)
            # [b] vs [b] => scalar tensor
            correct = torch.eq(pred, label).float().sum().item()
            total_correct += correct
            total_num += x.size(0)

        acc = total_correct / total_num
        print('testnoraml acc:', acc)
    # scheduler.step()




        # temp = temp.to(device)
        # with torch.no_grad():
        #     out, mu, logvar = model(temp)

        # viz.images(train_db.denormalize(temp), nrow=8, win='batch', opts=dict(title='x'))
        # viz.images(train_db.denormalize(out), nrow=8, win='x_hat', opts=dict(title='x_hat'))

    # torch.save(model.state_dict(), 'best.mdl')

def testLuft():
    root = '/mnt/data_sdb/datasets/BioreaktorAnomalieDaten/processed/unimodelLuft_data'


    batchsz = cfg.BATCH_SIZE
    num_trans = cfg.NUM_TRANS
    test_db = Luft(root, 64, mode='test')
    vali_db = Luft(root, 64, mode='vali')
    vali_loader = DataLoader(vali_db, batch_size=64, num_workers=0)
    test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=0)
    x, label = iter(vali_loader).next()
    print('x:', x.shape, 'label:', label.shape)

    device = torch.device('cuda')
    # viz = visdom.Visdom()
    torch.manual_seed(1234)
    model = WideResNet(16, num_trans, 8).to(device)
    model.load_state_dict(torch.load('bestLuft.mdl'))

    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 100], gamma=0.2)
    print(model)

    best_epoch, best_acc = 0, 0



        ##test

    total_correct = 0
    total_num = 0
    model.eval()
    with torch.no_grad():
        pbar = tqdm(enumerate(test_loader), total=len(test_loader))
        for batchidx, (x, label) in pbar:
            x, label = x.to(device), label.to(device)

            # [b, 72]
            logits = model(x)
            # [b]
            pred = logits.argmax(dim=1)
            # [b] vs [b] => scalar tensor
            correct = torch.eq(pred, label).float().sum().item()
            total_correct += correct
            total_num += x.size(0)

        acc = total_correct / total_num
        print('testanormal acc:', acc)
# # scheduler.step()

    total_correct = 0
    total_num = 0
    model.eval()
    with torch.no_grad():
        pbar = tqdm(enumerate(vali_loader), total=len(vali_loader))
        for batchidx, (x, label) in pbar:
            x, label = x.to(device), label.to(device)

            # [b, 72]
            logits = model(x)
            # [b]
            pred = logits.argmax(dim=1)
            # [b] vs [b] => scalar tensor
            correct = torch.eq(pred, label).float().sum().item()
            total_correct += correct
            total_num += x.size(0)

        acc = total_correct / total_num
        print('testnoraml acc:', acc)
    # scheduler.step()




        # temp = temp.to(device)
        # with torch.no_grad():
        #     out, mu, logvar = model(temp)

        # viz.images(train_db.denormalize(temp), nrow=8, win='batch', opts=dict(title='x'))
        # viz.images(train_db.denormalize(out), nrow=8, win='x_hat', opts=dict(title='x_hat'))

    # torch.save(model.state_dict(), 'best.mdl')


if __name__ == '__main__':
    testSpeed()
