import os, glob
import random
import csv
import cv2
from PIL import Image
import numpy as np


def load_csv(filename, root):

    if not os.path.exists(os.path.join(root, filename)):
        images = []

        images += glob.glob(os.path.join(root, "*.png"))
        images += glob.glob(os.path.join(root, "*.jpg"))
        images += glob.glob(os.path.join(root, "*.jpeg"))

        #     # 5189,  'E:\\data_lai\\unimodel_data\\Train_normal\\1668789435.312676_0.png',
        print(len(images))

        random.shuffle(images)
        with open(os.path.join(root, filename), mode="w", newline="") as f:
            writer = csv.writer(f)
            for img in images:
                # name = img.split(os.sep)[-2]
                writer.writerow([img])
            print("writen into csv file:", filename)

    images = []
    with open(os.path.join(root, filename)) as f:
        reader = csv.reader(f)
        for row in reader:
            img = row
            images.append(img)

    return images


class UserCrop:
    def __init__(self, filename, root):
        self.path = "/mnt/data_sdb/datasets/BioreaktorAnomalieDaten/raw/unimodelspeed/speed400/1668782313753_camera_frame.png"
        self.root = root
        self.filename = filename

    def cut(self):
        img = cv2.imread(self.path, flags=cv2.IMREAD_COLOR)
        if not os.path.exists(os.path.join(self.root, self.filename)):
            cv2.namedWindow("ROI selector", cv2.WINDOW_NORMAL)
            bbox = cv2.selectROI(img, False)
            with open(os.path.join(self.root, self.filename), mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(bbox)
                print("writen into csv file:", self.filename)

        with open(os.path.join(self.root, self.filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                bbox = [int(x) for x in row]
                print(bbox)

        return img, bbox
        cut = img[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]
        print(bbox)

    # # read from csv file
    # images, labels = [], []
    # with open(os.path.join(self.root, filename)) as f:
    #     reader = csv.reader(f)
    #     for row in reader:
    #         # 'pokemon\\bulbasaur\\00000000.png', 0
    #         img, label = row
    #         label = int(label)

    #         images.append(img)
    #         labels.append(label)

    # assert len(images) == len(labels)

    # return images, labels

def cut_picture():
    img, bbox = UserCrop('bboxSpeednoStirrsmall', '/mnt/data_sdb/datasets/BioreaktorAnomalieDaten/raw/unimodelspeed').cut()
    cut = img[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]
    cv2.namedWindow("cut")
    cv2.imshow("cut", cut)
    cv2.waitKey(0)
    # root = 'F:\\data_lai\\unimodel_data\\train_normal'
    # images = load_csv("test_a", root)
    # img, bbox = UserCrop('bbox', 'E:\\Program Files\\Abschlussarbeit\\geoTrans').cut()
    # print(images)
    # cut = img[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]
    # cv2.namedWindow("cut")
    # cv2.imshow("cut", cut)
    # cv2.waitKey(0)
    # cut = cv2.resize(cut, [256, 256])
    # cv2.namedWindow("cut2")
    # cv2.imshow("cut2", cut)
    # cv2.waitKey(0)
    # # import PIL
    # for i in range(5):
    #     # print("".join(images[i]))
    #     image = Image.open("".join(images[i])).convert("L")
    #     cut = image.crop(bbox)
    #     cut = np.array(cut)
    #     cut = np.expand_dims(cut, -1)
    #     print((image.dtype))
    #     cv2.namedWindow("a")
    #     cv2.imshow("a", image)
    #     cv2.waitKey(0)
    #     #cut = image[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]
    #     #cut = cv2.resize(cut, [64, 64])
    #     cv2.namedWindow("s")
    #     cv2.imshow("s", cut)
    #     cv2.waitKey(0)
def preprocess():
    root = '/mnt/data_sdb/datasets/BioreaktorAnomalieDaten/raw/unimodelspeed'
    save_root = '/mnt/data_sdb/datasets/BioreaktorAnomalieDaten/processed/unimodelSpeedData'
    _, bbox = UserCrop('bboxSpeednoStirrsmall', '/mnt/data_sdb/datasets/BioreaktorAnomalieDaten/raw/unimodelspeed').cut()
    for name in os.listdir(root):
        if not os.path.isdir(os.path.join(root, name)):
            continue
        print(os.path.join(root, name))
        save_path = os.path.join(save_root, name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        i = 0
        for img in os.listdir(os.path.join(root, name)):
            if os.path.join(os.path.join(root, name), img).endswith('.png'):
                img_path = os.path.join(os.path.join(root, name), img)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                cut = image[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]
                img_path = os.path.join(save_path, img)
                cv2.imwrite(img_path, cut, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                i += 1
        print(i)

if __name__ =='__main__':
    preprocess()