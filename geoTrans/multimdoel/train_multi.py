from dataset_bioreaktorMultiSpeedLuft import Bioreaktor_Detection as MultiSpeedLuft
from dataset_bioreaktorMultiSpeed import Bioreaktor_Detection as MultiSpeed
from torch.utils.data import DataLoader, Dataset
from Multi_Model import WideResNet
from Multi_Model_noInputLayer import WideResNet as WideResNetNoInput
import torch
import torch.optim as optim
from torch import nn
import visdom
from utils import Config as cfg
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
import numpy as np
from visdom import Visdom


def main():
    root = '/mnt/data_sdb/datasets/BioreaktorAnomalieDaten/processed/MultimodelSpeed'
    batchsz = cfg.BATCH_SIZE
    lr = cfg.LEARN_RATE
    epochs = cfg.EPOCHS
    num_trans = cfg.NUM_TRANS

    train_db = MultiSpeed(root, 64, mode='Train')
    vali_db = MultiSpeed(root, 64, mode='Vali')
    # test_db = MultiSpeed(root, 64, mode='Test')
    prozess_db = MultiSpeed(root, 64, mode='Prozess')
    train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True,
                            num_workers=0)
    vali_loader = DataLoader(vali_db, batch_size=batchsz, num_workers=0)
    # test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=0)
    prozess_loader = DataLoader(prozess_db, batch_size=batchsz, num_workers=0)
    x1, x2, label = iter(train_loader).next()
    print('x1:', x1.shape, 'x2:', x2.shape, 'label:', label.shape)

    device = torch.device('cuda')
    criterion = nn.CrossEntropyLoss().to(device)
    # viz = visdom.Visdom()
    torch.manual_seed(1234)
    model = WideResNet(10, num_trans, 4).to(device)
    if os.path.exists('/mnt/projects_sdc/lai/GeoTransForBioreaktor/ModelMultiSpeedRes53.mdl'):
        model.load_state_dict(torch.load('/mnt/projects_sdc/lai/GeoTransForBioreaktor/ModelMultiSpeedRes53.mdl'))
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
        path = 'ModelMultiSpeedRes5' + str(epoch) + '.mdl'
        torch.save(model.state_dict(), path)

        # validation
        if (epoch != 0 and epoch % cfg.VAL_EACH == 0) or epoch == 0:
            total_correct = 0
            total_num = 0
            model.eval()
            with torch.no_grad():
                pbar = tqdm(enumerate(vali_loader), total=len(vali_loader))
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
                print(epoch, 'normal acc:', normalacc)


        ##test
        # if (epoch != 0 and epoch % cfg.VAL_EACH == 0) or epoch == 0:
        #     total_correct = 0
        #     total_num = 0
        #     model.eval()
        #     with torch.no_grad():
        #         pbar = tqdm(enumerate(test_loader), total=len(test_loader))
        #         for batchidx, (x1, x2, label) in pbar:
        #             x1, x2, label = x1.to(device), x2.float().view(-1, cfg.INPUT_MULTI).to(device), label.to(device)

        #             # [b, 72]
        #             logits = model(x1, x2)
        #             # [b]
        #             pred = logits.argmax(dim=1)
        #             # [b] vs [b] => scalar tensor
        #             correct = torch.eq(pred, label).float().sum().item()
        #             total_correct += correct
        #             total_num += x1.size(0)

        #         acc = total_correct / total_num
        #         print(epoch, 'Zustandanomalie acc:', acc)
        #         if acc < worst_acc:
        #             best_epoch = epoch
        #             worst_acc = acc

        if (epoch != 0 and epoch % cfg.VAL_EACH == 0) or epoch == 0:
            total_correct = 0
            total_num = 0
            model.eval()
            with torch.no_grad():
                pbar = tqdm(enumerate(prozess_loader), total=len(prozess_loader))
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

                anormalacc_end = total_correct / total_num
                print(epoch, 'Parameteranomalie acc:', anormalacc_end)


        # scheduler.step()
        viz.line([[normalacc, anormalacc_end]], [global_step], win='test_acc', update='append')




        # temp = temp.to(device)
        # with torch.no_grad():
        #     out, mu, logvar = model(temp)

        # viz.images(train_db.denormalize(temp), nrow=8, win='batch', opts=dict(title='x'))
        # viz.images(train_db.denormalize(out), nrow=8, win='x_hat', opts=dict(title='x_hat'))

    # torch.save(model.state_dict(), 'best.mdl')

if __name__ == '__main__':
    main()
