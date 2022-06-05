# https://github.com/adap/flower/tree/main/examples/advanced_tensorflow 참조

import argparse
import os

import tensorflow as tf
import tensorflow_addons as tfa

import flwr as fl

from collections import Counter

import health_dataset as dataset

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# keras에서 내장 함수 지원(to_categofical())
from keras.utils.np_utils import to_categorical

import wandb

from datetime import datetime

from urllib.request import urlopen
from urllib.error import URLError, HTTPError
import requests
from fastapi import FastAPI, BackgroundTasks
import asyncio
import uvicorn
from pydantic.main import BaseModel
import logging
import json


# Make TensorFlow logs less verbose
# TF warning log 필터링
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Parse command line argument `partition`
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument("--partition", type=int, choices=range(0, 10), required=True)
args = parser.parse_args()

global client_num
client_num = args.partition # client 번호

# FL client 상태 확인
app = FastAPI()

class FLclient_status(BaseModel):
    FL_client: int = client_num
    FL_client_online: bool = True
    FL_client_start: bool = False
    FL_client_fail: bool = False
    # FL_server_IP: str = None 

status = FLclient_status()

# Define Flower client
class PatientClient(fl.client.NumPyClient):
    global client_num
    
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]
        num_rounds: int = config["num_rounds"]

        # wandb에 파라미터값 upload
        wandb.config.update({"num_rounds": num_rounds, "epochs": epochs,"batch_size": batch_size, "client_num": client_num})

        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
            validation_split=0.1,
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            # "val_loss": history.history["val_loss"][0],
            # "val_accuracy": history.history["val_accuracy"][0],
        }

        # 매 round 마다 성능지표 확인을 위한 log
        loss = history.history["loss"][0]
        accuracy = history.history["accuracy"][0]
        precision = history.history["precision"][0]
        recall = history.history["recall"][0]
        auc = history.history["auc"][0]
        # f1_score = history.history["f1_score"][0]

        # print(history.history)

        # local model 성능지표 wandb에 upload
        wandb.log({"loss": loss, "accuracy": accuracy, "precision": precision, "recall": recall, "auc":auc})

        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy, precision, recall, auc = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps)
        num_examples_test = len(self.x_test)
        
        return loss, num_examples_test, {"accuracy": accuracy, "precision": precision, "recall": recall, "auc": auc}
        # return loss, num_examples_test, {"accuracy": accuracy, "precision": precision, "recall": recall, "auc": auc, 'f1_score': f1_score, 'auprc': auprc}

@app.on_event("startup")
def startup():
    pass
    # loop = asyncio.get_event_loop()
    # loop.set_debug(True)
    # loop.create_task(run_client())

@app.get('/online')
def get_info():
    return status

@app.get("/start/{Server_IP}")
def main() -> None:
    # # Parse command line argument `partition`
    # parser = argparse.ArgumentParser(description="Flower")
    # parser.add_argument("--partition", type=int, choices=range(0, 10), required=True)
    # args = parser.parse_args()

    global client_num, status

    # client_num=0

    # data load
    # 환자별로 partition 분리 => 개별 클라이언트 적용
    (x_train, y_train), (x_test, y_test), label_count = load_partition(client_num)

    # Load and compile Keras model
    # 모델 및 메트릭 정의
    METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        # tfa.metrics.F1Score(name='f1_score', num_classes=5),
        tf.keras.metrics.AUC(name='auprc', curve='PR'), # precision-recall curve
    ]

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(
            16, activation='relu',
            input_shape=(x_train.shape[-1],)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(5, activation='sigmoid'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=METRICS)


    # Start Flower client
    client = PatientClient(model, x_train, y_train, x_test, y_test)
    fl.client.start_numpy_client("[::]:8080", client=client)
    
    print('FL server start')
    status.FL_client_start = True


# client manager에서 train finish 정보 확인
async def notify_fin():
    global status
    status.FL_client_start = False
    loop = asyncio.get_event_loop()
    future2 = loop.run_in_executor(None, requests.get, 'http://localhost:8003/trainFin')
    r = await future2
    print('try notify_fin')
    if r.status_code == 200:
        print('trainFin')
    else:
        print(r.content)

# client manager에서 train fail 정보 확인
async def notify_fail():
    global status
    status.FL_client_start = False
    loop = asyncio.get_event_loop()
    future1 = loop.run_in_executor(None, requests.get, 'http://localhost:8003/trainFail')
    r = await future1
    print('try notify_fail')
    if r.status_code == 200:
        print('trainFin')
    else:
        print(r.content)

def load_partition(idx: int):
    # Load the dataset partitions
    data, p_list = dataset.data_load()
    p_df = data[data.patient_id==p_list[idx]]

    # label 까지 포함 dataframe
    train_df, test_df = train_test_split(p_df.iloc[:,1:], test_size=0.2)

    # 특정 환자의 label 추출 => 환자마다 보유한 label이 다름
    label_column = train_df.loc[:,'label']
    label_count = Counter(label_column)
    label_list = list(label_count) # 보유 label

    # one-hot encoding 범위 지정 => 5개 label
    train_labels = to_categorical(np.array(train_df.pop('label')),5)
    test_labels = to_categorical(np.array(test_df.pop('label')),5)

    # label 제외한 input dataframe
    train_df = train_df.iloc[:,:7]
    test_df = test_df.iloc[:,:7]

    train_features = np.array(train_df)
    test_features = np.array(test_df)

    # 정규화
    # standard scaler
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    train_features = np.clip(train_features, -5, 5)
    test_features = np.clip(test_features, -5, 5)

    return (train_df, train_labels), (test_df,test_labels), len(label_list) # 환자의 레이블 개수

if __name__ == "__main__":
    # 날짜
    today= datetime.today()
    today_time = today.strftime('%Y-%m-%d %H-%M-%S')

    # # client_manager 주소
    # cl_manager: str = 'http://0.0.0.0:8003/'
    # cl_res = requests.get(cl_manager+'info')

    # # 최신 global model 버전
    # latest_gl_model_v = int(cl_res.json()['GL_Model_V'])
    
    # # 다음 global model 버전
    # next_gl_model = latest_gl_model_v + 1


    # wandb login and init
    wandb.login(key=os.getenv('WB_KEY'))
    # wandb.init(entity='ccl-fl', project='health_flower', name='health_acc_loss v2')
    wandb.init(entity='ccl-fl', project='client_flower', name= 'client %s_V0'%client_num, dir='/Users/yangsemo/VScode/Flower_Health/wandb_client')
    # wandb.init(entity='ccl-fl', project='client_flower', name= 'client_V%s'%next_gl_model, dir='/Users/yangsemo/VScode/Flower_Health/wandb_client')

    try:
        # client api 생성 => client manager와 통신하기 위함
        # uvicorn.run("app:app", host='0.0.0.0', port=8002, reload=True)

        # client FL 수행
        main()
        # client FL 종료
        # notify_fin()
        # status.FL_client_fail=False
    # except Exception as e:
        # client error
        # status.FL_client_fail=True
        # notify_fail()
    finally:
        # wandb 종료
        wandb.finish()

        # FL client out
        # requests.get('http://localhost:8003/flclient_out')
        print('%s client close'%client_num)
