import os, glob
import random
import csv
import cv2
from PIL import Image
import numpy as np
import json
import shutil


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
        self.path = "/mnt/data_sdb/datasets/BioreaktorAnomalieDaten/raw/MultiAll/Train10Zu/1668802503110_camera_frame.png"
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
    img, bbox = UserCrop('bboxMulti', '/mnt/data_sdb/datasets/BioreaktorAnomalieDaten/raw/MultiAll').cut()
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
    root = '/mnt/data_sdb/datasets/BioreaktorAnomalieDaten/raw/MultiAll/Vali10Zu'
    save_root = '/mnt/data_sdb/datasets/BioreaktorAnomalieDaten/processed/MultiModelAll'
    _, bbox = UserCrop('bboxMulti', root).cut()
    i = 0
    for json_name in os.listdir(root):
        if json_name.endswith('.json'):
            png_name = (json_name.split('.json')[0] + '_camera_frame' + ".png")
            png_root = os.path.join(root, png_name)
            json_root = os.path.join(root, json_name)
            image = cv2.imread(png_root, cv2.IMREAD_GRAYSCALE)
            cut = image[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]
            if i <= 999999999:
                save_path = os.path.join(save_root, 'Vali')
                if not os.path.exists(save_path):
                        os.makedirs(save_path)
                png_path = os.path.join(save_path, png_name)
                json_path = os.path.join(save_path, json_name)
            # else:
            #     save_path = os.path.join(save_root, 'Test')
            #     if not os.path.exists(save_path):
            #         os.makedirs(save_path)
            #     png_path = os.path.join(save_path, png_name)
            #     json_path = os.path.join(save_root, 'Test', json_name)
            cv2.imwrite(png_path, cut, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            shutil.copyfile(json_root, json_path)
            i += 1
    print(i)

def MakeDataSet():
    root = '/mnt/data_sdb/datasets/BioreaktorAnomalieDaten/processed/MultiModelAll/Vali'
    for json_name in os.listdir(root):
        if json_name.endswith('.json'):
            png_name = (json_name.split('.json')[0] + '_camera_frame' ".png")
            png_root = os.path.join(root, png_name)
            json_root = os.path.join(root, json_name)
            with open(os.path.join(json_root), 'r') as f2:
                temp = json.load(f2)
                multi_input = [temp["stirrer_rotational_speed"]["data"]["opcua_value"]["value"], temp["gas_flow_rate"]["data"]["opcua_value"]["value"]]
                if 250 < multi_input[0] < 750.0 and 25.0 < multi_input[1] < 75.0:
                    save_path = os.path.join(root, 'Vali')
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                else:
                    continue

                png_path = os.path.join(save_path, png_name)
                json_path = os.path.join(save_path, json_name)
            shutil.move(json_root, save_path)
            shutil.move(png_root, save_path)
    # for name in os.listdir(root):
    #     if not os.path.isdir(os.path.join(root, name)):
    #         continue
    #     print(os.path.join(root, name))
    #     i = 0
    #     for img in os.listdir(os.path.join(root, name)):
    #         if os.path.join(os.path.join(root, name), img).endswith('.png'):
    #             img_path = os.path.join(os.path.join(root, name), img)
    #             image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    #             cut = image[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]
    #             if i <= 3999:
    #                 save_path = os.path.join(save_root, 'Train', name)
    #                 if not os.path.exists(save_path):
    #                     os.makedirs(save_path)
    #                 img_path = os.path.join(save_path, img)
    #             else:
    #                 save_path = os.path.join(save_root, 'Test', name)
    #                 if not os.path.exists(save_path):
    #                     os.makedirs(save_path)
    #                 img_path = os.path.join(save_path, img)
    #             cv2.imwrite(img_path, cut, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    #             i += 1
    #     print(i)

if __name__ =='__main__':
    preprocess()
