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


class Bioreaktor_Detection(Dataset):

    def __init__(self, root, resize, mode, translation_x=0.25, translation_y=0.25):
        super(Bioreaktor_Detection, self).__init__()
        self.mode = mode
        if mode == 'Train':
            self.root = os.path.join(root, 'Train')
        if mode == 'Test' or mode == 'Vali' or mode == 'Prozess':
            self.root = os.path.join(root, 'Test')
        self.resize = resize
        self.max_tx = translation_x
        self.max_ty = translation_y

        self.name2label = {} # "speed200...":200 ... 800
        for name in sorted(os.listdir(os.path.join(self.root))):
            if not os.path.isdir(os.path.join(self.root, name)):
                continue

            self.name2label[name] = int(re.findall(r"\d+",name)[0])
        #  72
        self.transformation_dic = {}
        for i, transform in zip(range(cfg.NUM_TRANS), itertools.product((False, True),
                                                           (0, -self.max_tx, self.max_tx),
                                                           (0, -self.max_ty, self.max_ty),
                                                           range(4))):
            self.transformation_dic[i] = transform

        # Rotation 4
        # for i in range(cfg.NUM_TRANS):
        #     self.transformation_dic[i] = [False, 0, 0, i]
        #  Transaltion Flip 18
        # for i, transform in zip(range(cfg.NUM_TRANS), itertools.product((False, True),
        #                                                                 (0, -self.max_tx, self.max_tx),
        #                                                                 (0, -self.max_ty, self.max_ty))):
        #     # self.transformation_dic[i] += [False]
        #     list_temp = [0]
        #     list_temp.insert(0, transform[0])
        #     list_temp.insert(1, transform[1])
        #     list_temp.insert(2, transform[2])
        #     self.transformation_dic[i] = list_temp


        # save path
        self.images = []
        self.labels = []

        # traindatalen(self.images_train) 1200
        if mode == 'Test':
            self. images_train, self.train_multi_inputs, self.labels_train = self.load_csv('testanormal.csv')
            self.images = self.images_train[:cfg.NUM_TRANS * 100]
            self.labels = self.labels_train[:cfg.NUM_TRANS * 100]
            self.multi_inputs = self.train_multi_inputs[:cfg.NUM_TRANS * 100]
            # self.multi_inputs = (np.sum([np.random.randn(len(self.multi_inputs)).tolist(), self.multi_inputs], axis=0) - 400).tolist()
        # testdata anormallen(self.images_testanormal)
        if mode == 'Train':
            # a = self.images_c[10000:10100]
            self. images_testanormal, self.testanormal_multi_inputs, self.labels_testanormal = self.load_csv('train.csv')
            self.images = self.images_testanormal[:cfg.NUM_TRANS * 1000]
            self.labels = self.labels_testanormal[:cfg.NUM_TRANS * 1000]
            self.multi_inputs = self.testanormal_multi_inputs[:cfg.NUM_TRANS * 1000]
            # self.multi_inputs = (np.sum([np.random.randn(len(self.multi_inputs)).tolist(), self.multi_inputs], axis=0) - 400).tolist()
        ## vali data normal nicht trainiertlen(self.images_testnormal)
        if mode == 'Vali':
            self.images_testnormal, self.testnormal_multi_inputs, self.labels_testnormal = self.load_csv('testnormal.csv')
            self.images = self.images_testnormal[:cfg.NUM_TRANS * 100]
            self.labels = self.labels_testnormal[:cfg.NUM_TRANS * 100]
            self.multi_inputs = self.testnormal_multi_inputs[:cfg.NUM_TRANS * 100]
            # self.multi_inputs = (np.sum([np.random.randn(len(self.multi_inputs)).tolist(), self.multi_inputs], axis=0) - 400).tolist()
        ## Prozessanomalie
        if mode == 'Prozess':
            self.images_testnormal, self.testnormal_multi_inputs, self.labels_testnormal = self.load_csv('testnormal.csv')
            self.images = self.images_testnormal[:cfg.NUM_TRANS * 1000]
            self.labels = self.labels_testnormal[:cfg.NUM_TRANS * 1000]
            self.multi_inputs = self.testnormal_multi_inputs[:cfg.NUM_TRANS * 1000]
            # list1 = [200, 400, 600, 800]
            # self.multi_inputs = [np.random.choice(list(set(list1) ^ set([self.multi_inputs[i]]))) for i in range(len(self.multi_inputs))]
            # self.multi_inputs = (np.sum([np.random.randn(len(self.multi_inputs)).tolist(), self.multi_inputs], axis=0) / 400).tolist()


    def load_csv(self, filename):

        if not (os.path.exists(os.path.join(self.root, filename))):
            data_list = []
            for name in self.name2label.keys():
                # 'cat\\1.jpg
                # print(os.path.join(self.root, name))
                if self.mode == 'Test' and (name == 'Speed200' or name == 'Speed800'):
                    data_list += glob.glob(os.path.join(self.root, name, '*.png'))
                    data_list += glob.glob(os.path.join(self.root, name, '*.jpg'))
                    data_list += glob.glob(os.path.join(self.root, name, '*.jpeg'))
                elif (self.mode == 'Vali' or self.mode == 'Prozess') and (name == 'Speed400' or name == 'Speed600'):
                    data_list += glob.glob(os.path.join(self.root, name, '*.png'))
                    data_list += glob.glob(os.path.join(self.root, name, '*.jpg'))
                    data_list += glob.glob(os.path.join(self.root, name, '*.jpeg'))
                elif self.mode == 'Train' and (name == 'Speed400' or name == 'Speed600'):
                    data_list += glob.glob(os.path.join(self.root, name, '*.png'))
                    data_list += glob.glob(os.path.join(self.root, name, '*.jpg'))
                    data_list += glob.glob(os.path.join(self.root, name, '*.jpeg'))


            # 24946 ['E:\\Program Files\\praktikum\\UdemyTF_Template-main\\Chapter9_AdvancedDL
            # \\Chapter9_1_CustomDataset\\Cat\\0.jpg',

            print(len(data_list), data_list[0])

            random.shuffle(data_list)


            with open(os.path.join(self.root, filename), mode='w', newline='') as f1:
                writer_1 = csv.writer(f1)
                for img, i in itertools.product(data_list, range(cfg.NUM_TRANS)): # 'cat\\1.jpg'
                    name = img.split(os.sep)[-2]
                    label = i
                    multi_input = self.name2label[name]
                    # 'speed200', 200, 0
                    writer_1.writerow([img, multi_input, label])
                print('writen into csv file:', filename)
        # read from csv file
        images, multi_inputs, labels = [], [], []
        with open(os.path.join(self.root, filename)) as f1:
                reader = csv.reader(f1)
                for row in reader:
                    # 'cat\\1.jpg', 0
                    img, multi_input, label = row
                    label = int(label)
                    multi_input = int(multi_input)

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
            transforms.Normalize(mean=[0.485,],
                                 std=[0.229,])
        ])

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
        list1 = [200., 400., 600., 800.]
        if self.mode == 'Prozess':
            multi_inputs = np.random.choice(list(set(list1) ^ set([multi_inputs])))
            multi_inputs = (multi_inputs - 500 + random.gauss(0,1)) / 258.1989
            multi_inputs = torch.tensor(multi_inputs)
        else:
            multi_inputs = (multi_inputs - 500 + random.gauss(0,1)) / 258.1989
            multi_inputs = torch.tensor(multi_inputs)
        return img, multi_inputs, transformlabel

    def denormalize(self, x_hat):

        mean = [0.485,]
        std = [0.229,]
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


# root = "/mnt/data_sdb/datasets/BioreaktorAnomalieDaten/processed/MultimodelSpeed"
# train_db = Bioreaktor_Detection(root, 64, mode='Train')
# train_loader = DataLoader(train_db, batch_size=64, shuffle=False,
#                             num_workers=0)
# testnormal_db = Bioreaktor_Detection(root, 64, mode='Vali')
# testnormal_loader = DataLoader(train_db, batch_size=64, shuffle=True,
#                             num_workers=0)
# testanormal_db = Bioreaktor_Detection(root, 64, mode='Test')
# prozessanormal_db = Bioreaktor_Detection(root, 64, mode='Prozess')
# prozess_loader = DataLoader(prozessanormal_db, batch_size=64, shuffle=False,
#                             num_workers=0)
# x1, x2, label = iter(train_loader).next()
# print('x2:', x2, 'label:', label)
# for i in range(72):
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
