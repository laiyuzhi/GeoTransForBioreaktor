from dataset_bioreaktorMulti_NoGeo import Bioreaktor_Detection
from torch.utils.data import DataLoader, Dataset
from Multi_Model_noGeo import WideResNet
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
from sklearn.metrics import confusion_matrix, roc_curve, auc
def test(test_loader, device, model):
    total_correct_end =0
    total_num = 0
    model.eval()
    with torch.no_grad():
        pbar = tqdm(enumerate(test_loader), total=len(test_loader))
        for batchidx, (x1, x2, label) in pbar:
            x1, x2, label = x1.to(device), x2.float().view(-1, cfg.INPUT_MULTI).to(device), label.view(-1,1).to(device)

            # [b, 72]
            logits_end, logits_early= model(x1, x2)
            # [b]
            pred = torch.sigmoid(logits_end)
            correct_end = torch.where(pred>0.5, torch.ones_like(logits_end).to(device), torch.zeros_like(logits_end).to(device))
            correct = torch.eq(correct_end, label).float().sum().item()
            total_correct_end += correct
            total_num += x1.size(0)
            # [b] vs [b] => scalar tensor
            # cat to calcute confusion matrix
            if batchidx == 0:
                total_pred = pred
                total_label = label
            else:
                total_pred = torch.cat((total_pred, pred), 0)
                total_label = torch.cat((total_label, label), 0)

        TPFNTNFP_label = torch.Tensor.cpu(total_label)
        TPFNTNFP_prob = torch.Tensor.cpu(total_pred)
        anormalacc_end = total_correct_end / total_num
    return TPFNTNFP_label, TPFNTNFP_prob, anormalacc_end


def main(P_loss1=0.8):

    batchsz = cfg.BATCH_SIZE
    lr = cfg.LEARN_RATE
    epochs = cfg.EPOCHS
    num_out = 1
    root = "/mnt/data_sdb/datasets/BioreaktorAnomalieDaten/processed/MultiModelAll"
    train_db = Bioreaktor_Detection(root, 64, mode='Train')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True,
                            num_workers=0)
    test_db = Bioreaktor_Detection(root, 64, mode='Vali')
    test_loader = DataLoader(test_db, batch_size=batchsz, shuffle=False, num_workers=0)

    # x1, x2, label = iter(test_loader).next()
    # print('x1:', x1.shape, 'x2:', x2, 'label:', label.shape)
    # x1, x2, label = iter(test_loader).next()
    # print('x1:', x1.shape, 'x2:', x2, 'label:', label.shape)
    device = torch.device('cuda')
    criterion = nn.BCEWithLogitsLoss().to(device)
    torch.manual_seed(1234)
    model = WideResNet(10, num_out, 6, 10, 0, 0).to(device)
    path = '/mnt/projects_sdc/lai/GeoTransForBioreaktor/geoTrans/mdlMulti/NoGeoLossRatio/ModelMultiRes10_106_NoGeo_100_loss' + str(P_loss1) + '.mdl'
    print(path)
    model.load_state_dict(torch.load(path))
    # model.eval()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 15, 0.5)
    # print(model)

    viz = Visdom()
    viz.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))
    viz.line([[0.0]], [0.], win='test_acc', opts=dict(title='test acc.',
                                                   legend=['test acc.']))


    best_epoch, best_acc = 0, 0
    best_auc = 0
    global_step = 0
    for epoch in range(int(np.ceil(epochs / num_out))):
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for batchidx, (x1, x2, label) in pbar:
            # pbar.set_description("Epoch: s%" % str(epoch))
            # [b, 3, 64, 64]
            x1 = x1.to(device)
            x2 = x2.float().to(device)
            label = label.view(-1, 1).float().to(device)
            logits_end, logits_early = model(x1, x2)
            loss1 = criterion(logits_end, label)
            loss2 = criterion(logits_early, label)
            loss = P_loss1 * loss1  + (1-P_loss1) * loss2
            pred = torch.where(logits_end>0.0, torch.ones_like(logits_end).to(device), torch.zeros_like(logits_end).to(device))
            correct = torch.eq(pred, label).float().sum().item()

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            viz.line([loss.item()], [global_step], win='train_loss', update='append')

            pbar.set_description(f'Epoch [{epoch}/{int(np.ceil(epochs/num_out))}]')
            pbar.set_postfix({'loss early=': loss2.item(), 'loss hybrid=': loss1.item(), 'acc=': correct / x1.size(0)})

        print(epoch, 'loss:', loss.item())
        # path = 'ModelMultiRes5_106_NoGeo' + str(epoch) + '.mdl'
        # torch.save(model.state_dict(), path)

        ##test

        if (epoch != 0 and epoch % cfg.VAL_EACH == 0) or epoch == 0:

            TPFNTNFP_label, TPFNTNFP_prob, acc = test(test_loader, device, model)
            fpr, tpr, threshold = roc_curve(TPFNTNFP_label, TPFNTNFP_prob)
            roc_auc = auc(fpr, tpr)
            if acc > best_acc:
                best_acc = acc
                best_auc = roc_auc
                path = '/mnt/projects_sdc/lai/GeoTransForBioreaktor/geoTrans/mdlMulti/NoGeoLossRatio1/ModelMultiRes10_106_NoGeo_100_loss' + str(P_loss1) + '.mdl'
                # torch.save(model.state_dict(), path)

            print(epoch, 'anormal_end acc:', acc)

        viz.line([[acc]], [global_step], win='test_acc', update='append')
        # viz.line([loss.item()+0.1], [global_step], win='train_loss', name='test_acc',update='append', opts=dict(legend=['test acc.']))
        print(scheduler.get_lr()[0])
        scheduler.step()
    return best_acc, roc_auc

def find_p():

    best_acc = []
    x_label = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # x_label = ['0', '0.1']
    for i in x_label:
        acc,auc = main(i)
        best_acc.append(acc)

    bar = plt.bar(range(len(best_acc)), best_acc,tick_label=x_label)


    plt.bar_label(bar, label_type='edge')
    plt.xlabel('ratio for early loss')
    plt.ylabel('ACC')
    plt.title("Accuary under different Loss Ratio")
    plt.xticks(ticks=range(len(best_acc)) ,labels=x_label)
    plt.savefig('/mnt/projects_sdc/lai/GeoTransForBioreaktor/geoTrans/mdlMulti/NoGeoLossRatio/ration_loss_1.png')







if __name__ == '__main__':
    find_p()
