from sklearn.metrics import confusion_matrix, roc_curve, auc
from dataset_bioreaktorSpeed import Bioreaktor_Detection as Speed
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
import matplotlib.pyplot as plt




def load_data(dataname):
    if dataname == 'Luft':
        root = 'F:\\data_lai\\preprocess\\unimodelLuft_data'
        batchsz = cfg.BATCH_SIZE
        num_trans = cfg.NUM_TRANS
        test_db = Luft(root, 64, mode='test')
        vali_db = Luft(root, 64, mode='vali')
        vali_loader = DataLoader(vali_db, batch_size=batchsz, num_workers=0)
        test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=0)
        x, label = iter(vali_loader).next()
        print('x:', x.shape, 'label:', label.shape)

        device = torch.device('cuda')
        # viz = visdom.Visdom()
        model = WideResNet(16, num_trans, 8).to(device)
        model.load_state_dict(torch.load('E:\\Program Files\\Abschlussarbeit\\GeoTransForBioreaktor\\geoTrans\\mdl\\ModelLuftBest.mdl'))

        # Eva for normal data
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
                # cat to calcute confusion matrix
                if batchidx == 0:
                    total_pred = pred
                    total_label = label
                else:
                    total_pred = torch.cat((total_pred, pred), 0)
                    total_label = torch.cat((total_label, label), 0)
            total_pred = total_pred.view((-1, 72))
            total_label = total_label.view((-1, 72))
            TPFN = torch.eq(total_pred, total_label).float().sum(1)
            TPFN_prob = TPFN / cfg.NUM_TRANS

        # Eva for anormal data
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
                # cat to calcute confusion matrix
                if batchidx == 0:
                    total_pred = pred
                    total_label = label
                else:
                    total_pred = torch.cat((total_pred, pred), 0)
                    total_label = torch.cat((total_label, label), 0)
            total_pred = total_pred.view((-1, 72))
            total_label = total_label.view((-1, 72))
            TNFP = torch.eq(total_pred, total_label).float().sum(1)
            TNFP_prob = TNFP / cfg.NUM_TRANS

        TPFNTNFP_prob = torch.cat((TPFN_prob, TNFP_prob), 0)
        TPFNTNFP_label = torch.cat((torch.ones_like(TPFN), torch.zeros_like(TNFP)), 0)

        TPFNTNFP_label = torch.Tensor.cpu(TPFNTNFP_label)
        TPFNTNFP_prob = torch.Tensor.cpu(TPFNTNFP_prob)
    return TPFNTNFP_label, TPFNTNFP_prob

def cf_matrix(prob, label, threshold):
    pred = torch.where(prob >= threshold, 1, 0)
    print(pred)
    pred = torch.Tensor.cpu(pred)
    label = torch.Tensor.cpu(label)

    return confusion_matrix(label, pred)

def draw_roc(label, prob):
    label = label
    prob = prob
    fpr, tpr, threshold = roc_curve(label, prob)
    print(threshold)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC Curve(Anormalies: Air)")
    plt.legend(loc="lower right")
    plt.savefig("ROC Curve(Anormalies: Air).png")
    plt.show()
    maxindex = (tpr-fpr).tolist().index(max(tpr-fpr))
    best_threshold = threshold[maxindex]
    return best_threshold

label, prob = load_data('Luft')
print(label, prob)
threshold = draw_roc(label, prob)
print(threshold)
print(cf_matrix(prob, label, threshold))

