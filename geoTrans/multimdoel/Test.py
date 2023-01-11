
<<<<<<< HEAD
# import sys
# sys.path.append('/mnt/projects_sdc/lai/GeoTransForBioreaktor/geoTrans')

# import re
# import utils.Config as cfg
# print(cfg.BATCH_SIZE)
# string = "Speed200Luft2"
# print([int(x) for x in re.findall(r"\d+", string)])
=======
import sys
sys.path.append('E:\\Program Files\\Abschlussarbeit\\GeoTransForBioreaktor-4\\geoTrans')
import re
import utils.Config as cfg
import yaml
import os
import csv
import glob
import json

root = os.path.join("E:\\Program Files\\Abschlussarbeit\\GeoTransForBioreaktor-4\\geoTrans\\multimdoel\\yaml\\")
data_list = []
png_list = []
gas = []
speed = []
for x in os.listdir("E:\\Program Files\\Abschlussarbeit\\GeoTransForBioreaktor-4\\geoTrans\\multimdoel\\yaml\\Train"):
    if x.endswith(".json"):
        data_list.append(x)
        y = (x.split('.')[0] + "_camera_frame" + ".png")
        png_list.append(y)
print(data_list, png_list)
for x in data_list:
    with open(root + x, 'r') as f:
        temp = json.load(f)
        # print(json.dumps(temp, indent=4))
        # print(temp["gas_flow_rate"]["data"]["opcua_value"]["value"])
        gas.append(temp["gas_flow_rate"]["data"]["opcua_value"]["value"])
        speed.append(temp["stirrer_rotational_speed"]["data"]["opcua_value"]["value"])
print(gas)
print(speed)


# data_list += glob.glob(os.path.join(root, '*.json'))
# print(data_list)

# jpg_list = [x.split('.')[0] for x in os.listdir(path) if x.endswith(".jpg")]

# # print(cfg.BATCH_SIZE)
# string = "[200, 2]"
# print([int(x) for x in re.findall(r"\d+", string)])


# label = [1, 200]
# img = 'asd0//asdas'

# print([img,label])


# import csv

# data = [[1, 2, 3,4], 5,6]
# some_data = ['Foo','Bar','Baz','Qux','Zoo']
# print_data = []

# with open('test.csv', 'w') as f:
#     writer = csv.writer(f)
#     # Following code flattens the list within a list,
#     # uses temporary 'print_data' to store values for printing to csv
#     for counter in range(len(data)):
#         if isinstance(data[counter], list)==1 :
#             print ('list found')
#             for val in data[counter]:
#                 print_data.append(val)
#         else:
#              print_data.append(data[counter])

#     writer.writerow(print_data)
#     writer.writerow(some_data)



# # Python program to normalize a tensor to
# # 0 mean and 1 variance
# # Step 1: Importing torch
# import torch

# # Step 2: creating a torch tensor
# t = torch.tensor([0.,20.,2.])
# print("Tensor before Normalize:\n", t)

# # Step 3: Computing the mean, std and variance
# mean, std, var = torch.mean(t), torch.std(t), torch.var(t)
# print("Mean, Std and Var before Normalize:\n",
#       mean, std, var)
>>>>>>> 33cdc532720264df7336225634f3d413ffbc456c

# # Step 4: Normalizing the tensor
# t  = (t-mean)/std
# print("Tensor after Normalize:\n", t)

<<<<<<< HEAD
# label = [1, 200]
# img = 'asd0//asdas'

# print([img,label])

from visdom import Visdom
import numpy as np
import time

wind = Visdom()
wind.line([0.], [0.], win='train_loss', opts=dict(title='train loss'))

for step in range(10):
    loss = np.random.randn()
    wind.line([loss], [step], win='train_loss', update='append')
    time.sleep(0.5)
=======
# # Step 5: Again compute the mean, std and variance
# # after Normalize
# mean, std, var = torch.mean(t), torch.std(t), torch.var(t)
# print("Mean, std and Var after normalize:\n",
#       mean, std, var)

>>>>>>> 33cdc532720264df7336225634f3d413ffbc456c
