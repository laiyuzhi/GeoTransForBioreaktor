import pandas as pd
import numpy as np
import mlflow
from sklearn.tree import DecisionTreeClassifier
import itertools
import sys
sys.path.append('/mnt/projects_sdc/lai/GeoTransForBioreaktor/geoTrans')

# 隨機產生資料
train_x = np.random.rand(500, 3)
train_y = np.random.choice(a=[False,True], size=(500,), p=[0.5,0.5])

test_x = np.random.rand(100, 3)
test_y = np.random.choice(a=[False,True], size=(100,), p=[0.5,0.5])

# 候選的超參數
max_depth = [1,2,3,4,5]
min_leaf = [1,2,3]

mlflow.set_experiment("experiment lai")
experiment = mlflow.get_experiment_by_name("experiment lai")
# 使用 itertools 產生所有超參數的組合
all_combination = list(itertools.product(max_depth, min_leaf))
for a_max_depth, a_min_leaf in all_combination:
    with mlflow.start_run(experiment_id=experiment.experiment_id): # 新增的程式碼
        mlflow.log_param("max_depth", a_max_depth) # 新增的程式碼
        mlflow.log_param("min_leaf", a_min_leaf) # 新增的程式碼
        model = DecisionTreeClassifier(max_depth=a_max_depth, min_samples_leaf=a_min_leaf)
        model.fit(train_x, train_y)

        train_pred = model.predict(train_x)
        train_acc = sum(train_pred==train_y) / train_y.shape[0]
        mlflow.log_metric("train_acc",train_acc) # 新增的程式碼

        test_pred = model.predict(test_x)
        test_acc = sum(test_pred==test_y) / test_y.shape[0]
        mlflow.log_metric("test_acc", test_acc) # 新增的程式碼
