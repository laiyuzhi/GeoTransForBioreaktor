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

class Cat_Dog_Detection(Dataset):

    def __init__(self, root, resize, mode, translation_x=0.25, translation_y=0.25):
        super(Cat_Dog_Detection, self).__init__()

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
        # for i, transform in zip(range(cfg.NUM_TRANS), itertools.product((False, True),
        #                                                    (0, -self.max_tx, self.max_tx),
        #                                                    (0, -self.max_ty, self.max_ty),
        #                                                    range(4))):
        # for i in range(cfg.NUM_TRANS):    
        #     self.transformation_dic[i] = [False, 0, 0, i]
        for i, transform in zip(range(cfg.NUM_TRANS), itertools.product((False, True),
                                                                        (0, -self.max_tx, self.max_tx),
                                                                        (0, -self.max_ty, self.max_ty))):
            # self.transformation_dic[i] += [False]
            list_temp = [0]
            list_temp.insert(0, transform[0])
            list_temp.insert(1, transform[1])
            list_temp.insert(2, transform[2])
            self.transformation_dic[i] = list_temp
            

        # save path
        self.images = []
        self.labels = []
        self.images_c , self.labels_c, self.images_d, self.labels_d = self.load_csv('cat.csv', 'dog.csv') 

        # traindata only cat 5000
        if mode == 'train':
            self.images = self.images_c[:cfg.NUM_TRANS*10000]
            self.labels = self.labels_c[:cfg.NUM_TRANS*10000]
            # self.images = self.images_c[:int(0.4*len(self.images_c))]
            # self.labels = self.labels_c[:int(0.4*len(self.labels_c))]
        # testdata 100 dog 100 cat 
        if mode == 'test':
            # a = self.images_c[10000:10100]
            b = self.images_d[cfg.NUM_TRANS*10000:cfg.NUM_TRANS*10000+1000]
            #self.images = np.hstack((a, b))
            self.images = b
            
            # c = self.labels_c[10000:10100]
            d = self.labels_d[cfg.NUM_TRANS*10000:cfg.NUM_TRANS*10000+1000]
            #self.labels = np.hstack((c, d))  
            self.labels = d
            
        if mode == 'vali':
            self.images = self.images_c[cfg.NUM_TRANS*10000:cfg.NUM_TRANS*10000+1000]
            self.labels = self.labels_c[cfg.NUM_TRANS*10000:cfg.NUM_TRANS*10000+1000]
            # self.images = self.images_c[:int(0.4*len(self.images_c))]
            # self.labels = self.labels_c[:int(0.4*len(self.labels_c))]
                                  
    def load_csv(self, filename_c, filename_d):

        if not (os.path.exists(os.path.join(self.root, filename_c)) and os.path.exists(os.path.join(self.root, filename_d))):
            cats = []
            dogs = []

            for name in self.name2label.keys():
                # 'cat\\1.jpg
                # print(os.path.join(self.root, name))
                if name == 'Cat':
                    cats += glob.glob(os.path.join(self.root, name, '*.png'))
                    cats += glob.glob(os.path.join(self.root, name, '*.jpg'))
                    cats += glob.glob(os.path.join(self.root, name, '*.jpeg'))
                else:
                    dogs += glob.glob(os.path.join(self.root, name, '*.png'))
                    dogs += glob.glob(os.path.join(self.root, name, '*.jpg'))
                    dogs += glob.glob(os.path.join(self.root, name, '*.jpeg'))
                

            # 24946 ['E:\\Program Files\\praktikum\\UdemyTF_Template-main\\Chapter9_AdvancedDL
            # \\Chapter9_1_CustomDataset\\Cat\\0.jpg',
            
            print(len(cats), cats[0])
            print(len(dogs), dogs[0])

            random.shuffle(cats)
            random.shuffle(dogs)
            with open(os.path.join(self.root, filename_c), mode='w', newline='') as f1:
                with open(os.path.join(self.root, filename_d), mode='w', newline='') as f2:
                    writer_1 = csv.writer(f1)
                    for img, i in itertools.product(cats, range(cfg.NUM_TRANS)): # 'cat\\1.jpg'
                        name = img.split(os.sep)[-2]
                        label = i
                        # 'cat', 0
                        writer_1.writerow([img, label])
                    print('writen into csv file:', filename_c)

                    writer_2 = csv.writer(f2)
                    for img, i in itertools.product(dogs, range(cfg.NUM_TRANS)): # 'cat\\1.jpg'
                        name = img.split(os.sep)[-2]
                        
                        label = i
                        # 'dog', 1
                        writer_2.writerow([img, label])
                    print('writen into csv file:', filename_d)

        # read from csv file
        images_c, labels_c = [], []
        images_d, labels_d = [], []
        with open(os.path.join(self.root, filename_c)) as f1:
            with open(os.path.join(self.root, filename_d)) as f2:
                reader_1 = csv.reader(f1)
                for row in reader_1:
                    # 'cat\\1.jpg', 0
                    img, label = row
                    label = int(label)

                    images_c.append(img)
                    labels_c.append(label)

                reader_2 = csv.reader(f2)
                for row in reader_2:
                    # 'dog\\1.jpg', 1
                    img, label = row
                    
                    label = int(label)

                    images_d.append(img)
                    labels_d.append(label)

        assert len(images_c) == len(labels_c)
        assert len(images_d) == len(labels_d)

        return images_c, labels_c, images_d, labels_d

    def __len__(self):

        return len(self.images)
    
    def __getitem__(self, idx):
        # idx~[0~len(images)]
        # self.images, self.labels
        # img: 'pokemon\\bulbasaur\\00000000.png'
        # label: 0
        M = torch.zeros((2, 3))
        # transformlabel = np.random.randint(0, 72)
        M[0, 0] = 1
        M[1, 1] = 1
        img, transformlabel = self.images[idx], self.labels[idx]
        tf = transforms.Compose([
            lambda x:Image.open(x).convert('RGB'), # string path= > image data
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        img = tf(img)
        if self.transformation_dic[transformlabel][0]:
            img = torch.fliplr(img)
        if self.transformation_dic[transformlabel][1] != 0 or self.transformation_dic[transformlabel][2] != 0:
            M[0, 2] = self.transformation_dic[transformlabel][1]
            M[1, 2] = self.transformation_dic[transformlabel][2]
            M = torch.unsqueeze(M, 0)
            grid = f.affine_grid(M, torch.unsqueeze(img, 0).size())
            img = f.grid_sample(input=torch.unsqueeze(img, 0), grid=grid, padding_mode='reflection', align_corners=False)
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

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
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

def main():
    # viz = visdom.Visdom()
    root = "E:\\Program Files\\praktikum\\UdemyTF_Template-main\\Chapter9_AdvancedDL\\Chapter9_1_CustomDataset"
    data = Cat_Dog_Detection(root, 64, 'train')
    Iter = iter(data)
    for i in range(5):
        x, y = next(Iter)
        print(y)
        plt.imshow(data.denormalize(x).permute(1, 2, 0))
        plt.show()
    # viz.image(data.denormalize(x), win='sample_x', opts=dict(title='sample_x'))

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

if __name__ =='__main__':
    main()
#     torch.nn.functional.grid_sample()