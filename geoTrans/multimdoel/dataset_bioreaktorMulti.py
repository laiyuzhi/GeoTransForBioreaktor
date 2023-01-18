import torch
import os
import csv
import glob
from PIL import Image
from matplotlib import pyplot as plt
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import visdom
import itertools
import torch.nn.functional as f
import sys
sys.path.append('/mnt/projects_sdc/lai/GeoTransForBioreaktor/geoTrans')
from utils import Config as cfg
import cv2
import re
import json
import math


class Bioreaktor_Detection(Dataset):

    def __init__(self, root, resize, mode, translation_x=0.25, translation_y=0.25):
        '''
        Train: Train
        Test1: Trainparameter
        Test2: Not Trained Parameter
        '''
        super(Bioreaktor_Detection, self).__init__()
        self.mode = mode
        self.root = root
        self.resize = resize
        self.max_tx = translation_x
        self.max_ty = translation_y

        #  72
        self.transformation_dic = {}
        for i, transform in zip(range(cfg.NUM_TRANS), itertools.product((False, True),
                                                           (0, -self.max_tx, self.max_tx),
                                                           (0, -self.max_ty, self.max_ty),
                                                           range(4))):
            self.transformation_dic[i] = transform

        # save path
        self.images = []
        self.labels = []

        # traindatalen(self.images_train) 1200
        if mode == 'Test':
            self. images_train, self.train_multi_inputs, self.labels_train = self.load_csv('vali.csv')
            self.images = self.images_train[:cfg.NUM_TRANS * 90]
            self.labels = self.labels_train[:cfg.NUM_TRANS * 90]
            self.multi_inputs = self.train_multi_inputs[:cfg.NUM_TRANS * 90]
            # self.multi_inputs = (np.sum([np.random.randn(len(self.multi_inputs)).tolist(), self.multi_inputs], axis=0) - 400).tolist()

        # testdata anormallen(self.images_testanormal)
        if mode == 'Train':
            # a = self.images_c[10000:10100]
            self. images_testanormal, self.testanormal_multi_inputs, self.labels_testanormal = self.load_csv('train.csv')
            self.images = self.images_testanormal[:cfg.NUM_TRANS * 900]
            self.labels = self.labels_testanormal[:cfg.NUM_TRANS * 900]
            self.multi_inputs = self.testanormal_multi_inputs[:cfg.NUM_TRANS * 900]
            # self.multi_inputs = (np.sum([np.random.randn(len(self.multi_inputs)).tolist(), self.multi_inputs], axis=0) - 400).tolist()
        ## vali data normal nicht trainiertlen(self.images_testnormal)
        if mode == 'Vali':
            self.images_testnormal, self.testnormal_multi_inputs, self.labels_testnormal = self.load_csv('vali.csv')
            self.images = self.images_testnormal[:cfg.NUM_TRANS * 90]
            self.labels = self.labels_testnormal[:cfg.NUM_TRANS * 90]
            self.multi_inputs = self.testnormal_multi_inputs[:cfg.NUM_TRANS * 90]
            # self.multi_inputs = (np.sum([np.random.randn(len(self.multi_inputs)).tolist(), self.multi_inputs], axis=0) - 400).tolist()

    def load_csv(self, filename):

        if not (os.path.exists(os.path.join(self.root, filename))):
            json_list =[]
            png_list = []
            if self.mode == 'Train':
                json_path = os.path.join(self.root, "Train")
            elif self.mode == 'Vali':
                json_path = os.path.join(self.root, "Vali")
            elif self.mode == 'Test':
                json_path = os.path.join(self.root, "Vali")
                # json_path = os.path.join('/mnt/data_sdb/datasets/BioreaktorAnomalieDaten/processed/MultiModelAll/Train/Test/Test2')
            # elif self.mode == 'Test2':
            #     json_path = os.path.join(self.root, "Test2")

            for json_name in os.listdir(json_path):
                if json_name.endswith('.json'):
                    json_list.append(json_name)
            # 24946 ['E:\\Program Files\\praktikum\\UdemyTF_Template-main\\Chapter9_AdvancedDL
            # \\Chapter9_1_CustomDataset\\Cat\\0.jpg',
            print(len(json_list), json_list[0])
            random.shuffle(json_list)

            with open(os.path.join(self.root, filename), mode='w', newline='') as f1:
                writer_1 = csv.writer(f1)
                for name, i in itertools.product(json_list, range(cfg.NUM_TRANS)): # 'cat\\1.jpg'
                        png = name.split('.json')[0] + '_camera_frame' + ".png"
                        png_path = os.path.join(json_path, png)
                        label = i
                        with open(os.path.join(json_path, name), 'r') as f2:
                            temp = json.load(f2)
                            multi_input = [int(temp["stirrer_rotational_speed"]["data"]["opcua_value"]["value"]),int(round(temp["gas_flow_rate"]["data"]["opcua_value"]["value"]/10)*10) ]
                        # 'speed200', 200, 0
                        writer_1.writerow([png_path, label] + multi_input)
                print('writen into csv file:', filename)
        # read from csv file
        images, labels, Speed, Volumestrom = [], [], [], []
        multi_inputs = []
        with open(os.path.join(self.root, filename)) as f1:
                reader = csv.reader(f1)
                for row in reader:
                    # 'cat\\1.jpg', 0
                    img, label, Speed, Volumestrom = row
                    label = int(label)
                    multi_input = [int(Speed), int(Volumestrom)]

                    images.append(img)
                    multi_inputs.append(multi_input)
                    labels.append(label)

        assert len(images) == len(labels)
        return images, multi_inputs, labels

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):
        # idx~[0~len(images)]
        # self.images, self.multi_inputs, self.labels
        # img: 'pokemon\\bulbasaur\\00000000.png'
        # multi_inputs: 200
        # label: 0
        M = torch.zeros((2, 3))
        M[0, 0] = 1
        M[1, 1] = 1
        img, multi_inputs, transformlabel = self.images[idx], self.multi_inputs[idx], self.labels[idx]
        tf = transforms.Compose([
            lambda x:Image.open(x).convert('L'), # string path= > image data
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.2686,],
                                 std=[0.0940,])
        ])
    # transforms.Normalize(mean=[0.485,],
                                #  std=[0.229,])
        img = tf(img)
        if self.transformation_dic[transformlabel][0]:
            img = torch.flip(img, dims=[2])
        if self.transformation_dic[transformlabel][1] != 0 or self.transformation_dic[transformlabel][2] != 0:
            M[0, 2] = self.transformation_dic[transformlabel][1]
            M[1, 2] = self.transformation_dic[transformlabel][2]
            M = torch.unsqueeze(M, 0)
            grid = f.affine_grid(M, torch.unsqueeze(img, 0).size())
            img = f.grid_sample(input=torch.unsqueeze(img, 0), grid=grid, padding_mode='reflection', align_corners=True)
            img = torch.squeeze(img, 0)
        if self.transformation_dic[transformlabel][3] != 0:
            img = torch.rot90(img, k=self.transformation_dic[transformlabel][3], dims=(1, 2))

        transformlabel = torch.tensor(transformlabel)
        # if self.mode == 'Test2':
        #     multi_inputs[0] = (multi_inputs[0] - 500 + random.gauss(multi_inputs[0], 10)) / 288.6751
        #     multi_inputs[1] = (multi_inputs[1] - 49.25 + random.gauss(multi_inputs[1], 1)) / 28.43
        #     multi_inputs = torch.tensor(multi_inputs)
        multi_inputs = torch.tensor(multi_inputs).float()
        if self.mode == 'Test':
            if multi_inputs[0] <= 250:
                multi_inputs[0] = torch.Tensor([random.uniform(multi_inputs[0]+300, 900)])
            elif 250 <= multi_inputs[0] <= 650:
                list0 = [random.uniform(multi_inputs[0]+300, 900), random.uniform(0, multi_inputs[0]-300)]
                multi_inputs[0] = torch.Tensor([np.random.choice(list0)])
            elif multi_inputs[0] >= 650:
                multi_inputs[0] = torch.Tensor([random.uniform(0, multi_inputs[0]-300)])

            if multi_inputs[1] <= 25:
                multi_inputs[1] = torch.Tensor([random.uniform(multi_inputs[1]+30, 90)])
            elif 25 <= multi_inputs[1] <= 65:
                list1 = [random.uniform(multi_inputs[1]+30, 90), random.uniform(0, multi_inputs[1]-30)]
                multi_inputs[1] = torch.Tensor([np.random.choice(list1)])
            elif multi_inputs[1] >= 65:
                multi_inputs[1] = torch.Tensor([random.uniform(0, multi_inputs[1])])
            multi_inputs[0] = multi_inputs[0].float()
            multi_inputs[1] = multi_inputs[1].float()

        elif self.mode == 'Train' or self.mode == 'Vali':
            # multi_inputs[0] = (multi_inputs[0] - 450) / 259.8076  259.8079, 25.98
            # multi_inputs[1] = (multi_inputs[1] - 45) / 25.98
            multi_inputs[0] = (multi_inputs[0])
            multi_inputs[1] = (multi_inputs[1])


        multi_inputs = torch.div(torch.add(multi_inputs, torch.Tensor([-450, -45])) , torch.Tensor([259.8076, 25.98]))
        return img, multi_inputs, transformlabel

    def denormalize(self, x_hat):

        mean = [0.2686,]
        std = [0.0940,]
        device = torch.device('cuda')
        # x_hat = (x-mean)/std
        # x = x_hat*std = mean
        # x: [c, h, w]
        # mean: [3] => [3, 1, 1]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        # print(mean.shape, std.shape)
        x = x_hat * std + mean

        return x


# root = "/mnt/data_sdb/datasets/BioreaktorAnomalieDaten/processed/MultiModelAll/DataDistanz400"
# train_db = Bioreaktor_Detection(root, 64, mode='Train')
# train_loader = DataLoader(train_db, batch_size=64, shuffle=False,
#                         num_workers=16)

# def get_mean_std(loader):
#     # Var[x] = E[X**2]-E[X]**2
#     channels_sum,channels_squared_sum,num_batches = 0,0,0
#     for data, _, _ in loader:
#         channels_sum += torch.mean(data, dim=[0,2,3])
#         channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
#         num_batches += 1

#     print(num_batches)
#     print(channels_sum)
#     mean = channels_sum/num_batches
#     std = (channels_squared_sum/num_batches - mean**2) **0.5

#     return mean,std

# mean,std = get_mean_std(train_loader)

# print(mean)
# print(std)
# # print(1)
# testnormal_db = Bioreaktor_Detection(root, 64, mode='Vali')
# testnormal_loader = DataLoader(testnormal_db, batch_size=64, shuffle=False,
#                             num_workers=16)
# print(2)
# testanormal_db = Bioreaktor_Detection(root, 64, mode='Test')
# testanormal_loader = DataLoader(testanormal_db, batch_size=64, shuffle=False,
#                             num_workers=16)
# print(3)
# # prozessanormal_db = Bioreaktor_Detection(root, 64, mode='Prozess')
# # prozess_loader = DataLoader(prozessanormal_db, batch_size=64, shuffle=False,
# #                             num_workers=0)
# x1, x2, label = iter(testanormal_loader).next()
# print('x2:', x2, 'label:', label)
# # # # # for i in range(72):
#     x, y, z = next(Iter)
#     print(len(y), z)
#     # if i > 6:
    #     plt.imshow(train_db.denormalize(x).permute(1, 2, 0))
    #     plt.show()
# def main():r
#     # viz = visdom.Visdom()
#     root = "E:\\Program Files\\praktikum\\UdemyTF_Template-main\\Chapter9_AdvancedDL\\Chapter9_1_CustomDataset"
#     data = Cat_Dog_Detection(root, 64, 'train')
#     Iter = iter(data)
#     for i in range(5):
#         x, y = next(Iter)
#         print(y)
#         plt.imshow(data.denormalize(x).permute(1, 2, 0))
#         plt.show()
#     # viz.image(data.denormalize(x), win='sample_x', opts=dict(title='sample_x'))

#     # loader = DataLoader(data, batch_size=32, shuffle=True, num_workers=0)
#     # for x, y in enumerate(loader):
#     #     print(x)
#     #     print(y)
#     # i = 0
#     # viz.image(
#     #     np.random.rand(3, 512, 256), #随机生成一张图
#     #     opts = dict(title = 'Random!', caption = 'how random'),
#     # )
#     # for x in loader:
#     #         viz.images(data.denormalize(x), nrow=8, win='batch', opts=dict(title='batch'))
#     #         # viz.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))
#     #         print(x.shape)
#     #         time.sleep(10)
#     # cats, labels_c, dogs, labels_d = data.load_csv('cat.csv', 'dog.csv')
#     # print(cats[0], labels_c[0], dogs[0], labels_d[0])

# if __name__ =='__main__':
#     main()
#     torch.nn.functional.grid_sample()
