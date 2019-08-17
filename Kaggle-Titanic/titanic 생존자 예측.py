import pandas as pd
import numpy as np
import math, csv
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
import sys

def sigmoid(z):
    return 1 / (1 + np.exp(-z.astype(float)))

def age_prepro_null(data):
    nan_age_avg = np.nanmedian(data["Age"])
    data["Age"] = data["Age"].fillna(nan_age_avg)

def age_prepro_disc(data):
    range = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
    group_range = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    data['Age_range'] = pd.cut(data['Age'], range, labels=group_range)

def sex_prepro(data):
    sex_mapping = {"male": 1, "female": 2}
    data['Sex'] = data['Sex'].map(sex_mapping)

def fare_prepro(data):
    data["Fare"] = data["Fare"].fillna(13.675)


def feature_sel(train_data, test_data):
    # "Name",  "Ticket", "Cabin","Embarked" no using
    using_columns = ["Pclass", "Sex", "Age_range", "SibSp", "Fare"]
    x_train = pd.DataFrame(train_data, columns=using_columns)
    y_train = train_data["Survived"]
    x_test = pd.DataFrame(test_data, columns=using_columns)
    return x_train, y_train, x_test, using_columns


def changeDataForUsing(x_train, y_train, x_test):
    x_train_data = x_train.get_values()
    y_train_data = np.array([y_train]).T
    x_test_data = x_test.get_values()
    return x_train_data, y_train_data, x_test_data


def load_data():
    train_data = pd.read_csv(sys.argv[1])
    test_data = pd.read_csv(sys.argv[2])

    age_prepro_null(train_data)
    age_prepro_null(test_data)

    age_prepro_disc(train_data)
    age_prepro_disc(test_data)

    sex_prepro(train_data)
    sex_prepro(test_data)

    fare_prepro(test_data)

    x_train, y_train, x_test, using_columns = feature_sel(train_data, test_data)
    x_train, y_train, x_test = changeDataForUsing(x_train, y_train, x_test)
    return x_train, y_train, x_test, using_columns


x_train, y_train, x_test, using_columns = load_data()
train_m = len(x_train)
epochs = 150000
lr = 0.0005
seed = 2019
np.random.seed(seed)
input_dim = len(using_columns)
w0 = np.ones((train_m, 1), dtype=float)
x_train = np.append(w0, x_train, axis=1)
w = np.random.randn(input_dim + 1).reshape(input_dim + 1, 1)
x = []
y = []
for epoch in range(epochs):
    h_x = x_train.dot(w)
    h_x = sigmoid(h_x)
    if epoch % 1000 == 0 and not epoch == 0:
        x.append(epoch)
        y.append(log_loss(y_train, h_x))
    for i in range(len(w)):
        gradient = (1 / train_m) * np.sum((h_x - y_train) * x_train[:, i].reshape(train_m, 1), axis=0)  # 편미분 구하는 부분
        w[i] = w[i] - (lr * gradient)  # 매개변수 업데이트 하는 부분

w0 = np.ones((len(x_test), 1), dtype=float)
x_test = np.append(w0, x_test, axis=1)
result = x_test.dot(w)

predict = []
for i in range(len(result)):
    if result[i] >= 0.5:
        predict.append(1)
    else:
        predict.append(0)
start_index = 892
csvfile = open("1614263.csv", "w")
csvwriter = csv.writer(csvfile)
csvwriter.writerow(["PassengerId", "Survived"])
for i in predict:
    csvwriter.writerow([str(start_index), i])
    start_index = start_index + 1