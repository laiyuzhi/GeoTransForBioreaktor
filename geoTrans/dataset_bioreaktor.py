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
from utils import Config as cfg
import cv2

class Bioreaktor_Detection(Dataset):

    def __init__(self, root, resize, mode, translation_x=0.25, translation_y=0.25):
        super(Bioreaktor_Detection, self).__init__()

        self.root = root
        self.resize = resize
        self.max_tx = translation_x
        self.max_ty = translation_y

        self.name2label = {} # "sq...":0 ... 1
        for name in sorted(os.listdir(os.path.join(root))):
            if not os.path.isdir(os.path.join(root, name)):
                continue
    
            self.name2label[name] = len(self.name2label.keys())

        self.transformation_dic = {}
        #  72
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
        self.images_train, self.labels_train, self.images_testanormal, self.labels_testanormal, self.images_testnormal, self.labels_testnormal = self.load_csv('train.csv', 'test_anormal.csv', 'test_normal.csv') 
        
        # traindata 
        if mode == 'train':
            self.images = self.images_train[:cfg.NUM_TRANS*len(self.images_train)]
            self.labels = self.labels_train[:cfg.NUM_TRANS*len(self.images_train)]
        # testdata anormal
        if mode == 'test':
            # a = self.images_c[10000:10100]
            self.images = self.images_testanormal[:cfg.NUM_TRANS*len(self.images_testanormal)]
            self.labels = self.labels_testanormal[:cfg.NUM_TRANS*len(self.images_testanormal)]
        ## vali data normal nicht trainiert    
        if mode == 'vali':
            self.images = self.images_testnormal[:cfg.NUM_TRANS*len(self.images_testnormal)]
            self.labels = self.labels_testnormal[:cfg.NUM_TRANS*len(self.images_testnormal)]
            # self.images = self.images_c[:int(0.4*len(self.images_c))]
            # self.labels = self.labels_c[:int(0.4*len(self.labels_c))]
                                  
    def load_csv(self, filename_train, filename_testanormal, filename_testnormal):

        if not (os.path.exists(os.path.join(self.root, filename_train)) and os.path.exists(os.path.join(self.root, filename_testanormal)) and os.path.exists(os.path.join(self.root, filename_testnormal))):
            train = []
            testnormal = []
            testanormal = []

            for name in self.name2label.keys():
                # 'cat\\1.jpg
                # print(os.path.join(self.root, name))
                if name == 'train_normal':
                    train += glob.glob(os.path.join(self.root, name, '*.png'))
                    train += glob.glob(os.path.join(self.root, name, '*.jpg'))
                    train += glob.glob(os.path.join(self.root, name, '*.jpeg'))
                elif name == 'test_normal':
                    testnormal += glob.glob(os.path.join(self.root, name, '*.png'))
                    testnormal += glob.glob(os.path.join(self.root, name, '*.jpg'))
                    testnormal += glob.glob(os.path.join(self.root, name, '*.jpeg'))
                elif name == 'test_anormal':
                    testanormal += glob.glob(os.path.join(self.root, name, '*.png'))
                    testanormal += glob.glob(os.path.join(self.root, name, '*.jpg'))
                    testanormal += glob.glob(os.path.join(self.root, name, '*.jpeg'))
                

            # 24946 ['E:\\Program Files\\praktikum\\UdemyTF_Template-main\\Chapter9_AdvancedDL
            # \\Chapter9_1_CustomDataset\\Cat\\0.jpg',
            
            print(len(train), train[0])
            print(len(testnormal), testnormal[0])
            print(len(testanormal), testanormal[0])

            random.shuffle(train)
            random.shuffle(testnormal)
            random.shuffle(testanormal)
            with open(os.path.join(self.root, filename_train), mode='w', newline='') as f1:
                writer_1 = csv.writer(f1)
                for img, i in itertools.product(train, range(cfg.NUM_TRANS)): # 'cat\\1.jpg'
                    name = img.split(os.sep)[-2]
                    label = i
                    # 'cat', 0
                    writer_1.writerow([img, label])
                print('writen into csv file:', filename_train)

            with open(os.path.join(self.root, filename_testanormal), mode='w', newline='') as f2:
                writer_2 = csv.writer(f2)
                for img, i in itertools.product(testanormal, range(cfg.NUM_TRANS)): # 'cat\\1.jpg'
                    name = img.split(os.sep)[-2]
                    
                    label = i
                    # 'dog', 1
                    writer_2.writerow([img, label])
                print('writen into csv file:', filename_testanormal)
            with open(os.path.join(self.root, filename_testnormal), mode='w', newline='') as f3:
                writer_3 = csv.writer(f3)
                for img, i in itertools.product(testnormal, range(cfg.NUM_TRANS)): # 'cat\\1.jpg'
                    name = img.split(os.sep)[-2]
                    
                    label = i
                    # 'dog', 1
                    writer_3.writerow([img, label])
                print('writen into csv file:', filename_testnormal)

        # read from csv file
        images_train, labels_train = [], []
        images_testnormal, labels_testnormal = [], []
        images_testanormal, labels_testanormal = [], []
        with open(os.path.join(self.root, filename_train)) as f1:
            with open(os.path.join(self.root, filename_testanormal)) as f2:
                with open(os.path.join(self.root, filename_testnormal)) as f3:
                    reader_1 = csv.reader(f1)
                    for row in reader_1:
                        # 'cat\\1.jpg', 0
                        img, label = row
                        label = int(label)

                        images_train.append(img)
                        labels_train.append(label)

                    reader_2 = csv.reader(f2)
                    for row in reader_2:
                        # 'dog\\1.jpg', 1
                        img, label = row
                        
                        label = int(label)

                        images_testanormal.append(img)
                        labels_testanormal.append(label)
                    
                    reader_3 = csv.reader(f3)
                    for row in reader_3:
                        # 'dog\\1.jpg', 1
                        img, label = row
                        
                        label = int(label)

                        images_testnormal.append(img)
                        labels_testnormal.append(label)

        assert len(images_train) == len(labels_train)
        assert len(images_testanormal) == len(labels_testanormal)
        assert len(images_testnormal) == len(labels_testnormal)
        return images_train, labels_train, images_testanormal, labels_testanormal, images_testnormal, labels_testnormal

    def __len__(self):

        return len(self.images)
    
    def __getitem__(self, idx):
        # idx~[0~len(images)]
        # self.images, self.labels
        # img: 'pokemon\\bulbasaur\\00000000.png'
        # label: 0
        M = torch.zeros((2, 3))
        M[0, 0] = 1
        M[1, 1] = 1
        img, transformlabel = self.images[idx], self.labels[idx]
        # img = Image.open("".join(img)).convert('L')
        # img = np.array(img)
        # # img = np.expand_dims(img, -1) # (H, w, 1)
        # img = Image.fromarray(img)
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
        return img, transformlabel

        # tf = transforms.Compose([
        #     # lambda x:Image.open(x).convert('RGB'), # string path= > image data
        #     # transforms.Resize((int(self.resize*1.25), int(self.resize*1.25))),
        #     # transforms.RandomRotation(15),
        #     # transforms.CenterCrop(self.resize),
        #     transforms.ToTensor()
        # ])
    
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


# oot = "F:\\data_lai\\unimodel_data"        
# train_db = Bioreaktor_Detection(root, 64, mode='train')
# Iter = iter(train_db)
# for i in range(72):
#     x, y = next(Iter)
#     print(y)
#     if i > 6:
#         plt.imshow(train_db.denormalize(x).permute(1, 2, 0))
#         plt.show()
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