#from dataset_bioreaktor import Bioreaktor_Detection
from dataset_bioreaktorSpeed import Bioreaktor_Detection
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

def main():
    root = '/mnt/data_sdb/datasets/BioreaktorAnomalieDaten/processed/unimodelSpeedData'
    

    batchsz = cfg.BATCH_SIZE
    lr = cfg.LEARN_RATE
    epochs = cfg.EPOCHS
    num_trans = cfg.NUM_TRANS
    train_db = Bioreaktor_Detection(root, 64, mode='train')
    testbig_db = Bioreaktor_Detection(root, 64, mode='testbig')
    testsmall_db = Bioreaktor_Detection(root, 64, mode='testsmall')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True,
                            num_workers=0)
    testbig_loader = DataLoader(testbig_db, batch_size=batchsz, num_workers=0)
    testsmall_loader = DataLoader(testsmall_db, batch_size=batchsz, num_workers=0)
    x, label = iter(train_loader).next()
    print('x:', x.shape, 'label:', label.shape)

    device = torch.device('cuda')
    criterion = nn.CrossEntropyLoss().to(device)
    # viz = visdom.Visdom()
    torch.manual_seed(1234)
    model = WideResNet(16, num_trans, 8).to(device)
    if os.path.exists('best.mdl'):
        model.load_state_dict(torch.load('best.mdl'))
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 100], gamma=0.2)
    print(model)
    
    best_epoch, best_acc = 0, 0
    for epoch in range(int(np.ceil(epochs/num_trans))):
        model.train()  
        train_loss_one = 0
        train_num = 0   
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for batchidx, (x, label) in pbar:
            # pbar.set_description("Epoch: s%" % str(epoch))
            # [b, 3, 64, 64]
            x = x.to(device)
            label = label.to(device)

            logits = model(x)
            loss = criterion(logits, label)

            pred = logits.argmax(dim=1)
            correct = torch.eq(pred, label).float().sum().item()

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(f'Epoch [{epoch}/{int(np.ceil(epochs/num_trans))}]')
            pbar.set_postfix({'loss=': loss.item(), 'acc=': correct/x.size(0)})
        
        print(epoch, 'loss:', loss.item())
        path = 'modelspeed' + str(epoch) + '.mdl'
        torch.save(model.state_dict(), path)


        ##test  
        if epoch != 0 and epoch % cfg.VAL_EACH == 0:
            total_correct = 0
            total_num = 0
            model.eval()
            with torch.no_grad():
                pbar = tqdm(enumerate(testbig_loader), total=len(testbig_loader))
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
                print(epoch, 'anormalbig acc:', acc)
        # scheduler.step()
        if epoch != 0 and epoch % cfg.VAL_EACH == 0:
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
                print(epoch, 'anormalsmall acc:', acc)
        # scheduler.step()



       
        # temp = temp.to(device)
        # with torch.no_grad():
        #     out, mu, logvar = model(temp)
            
        # viz.images(train_db.denormalize(temp), nrow=8, win='batch', opts=dict(title='x'))
        # viz.images(train_db.denormalize(out), nrow=8, win='x_hat', opts=dict(title='x_hat'))

    # torch.save(model.state_dict(), 'best.mdl')      

if __name__ == '__main__':
    main()