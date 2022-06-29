# https://github.com/adap/flower/tree/main/examples/advanced_tensorflow 참조

import os, time, logging, json

import tensorflow as tf

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
from functools import partial
from urllib.request import urlopen
import requests
from fastapi import FastAPI, BackgroundTasks
import asyncio
import uvicorn
from pydantic.main import BaseModel

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Make TensorFlow logs less verbose
# TF warning log 필터링
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

global client_num
client_num = 1 # client 번호

# FL client 상태 확인
app = FastAPI()

class FLclient_status(BaseModel):
    FL_client: int = client_num
    FL_client_online: bool = True
    FL_client_start: bool = False
    FL_client_fail: bool = False
    FL_server_IP: str = None 

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
        # wandb.config.update({"num_rounds": num_rounds, "epochs": epochs,"batch_size": batch_size, "client_num": client_num})

        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
            validation_split=0.2,
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }

        # 매 round 마다 성능지표 확인을 위한 log
        # loss = history.history["loss"][0]
        # accuracy = history.history["accuracy"][0]
        # precision = history.history["precision"][0]
        # recall = history.history["recall"][0]
        # auc = history.history["auc"][0]
        # f1_score = history.history["f1_score"][0]

        # print(history.history)

        # local model 성능지표 wandb에 upload
        # wandb.log({"loss": loss, "accuracy": accuracy, "precision": precision, "recall": recall, "auc":auc})

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

def build_model():
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
            input_shape=(6,)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(5, activation='sigmoid'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=METRICS)

    return model

@app.on_event("startup")
def startup():
    pass

@app.get('/online')
def get_info():
    return status

@app.get("/start/{Server_IP}")
async def flclientstart(background_tasks: BackgroundTasks, Server_IP: str):
    global status
    global model
    model = build_model()
    logging.info('bulid model')

    logging.info('FL start')
    status.FL_client_start = True
    status.FL_server_IP = Server_IP
    background_tasks.add_task(run_client)
    return status

async def run_client():
    global model
    try:
        logging.info('FL Run')
        
        # time.sleep(10)
        res = requests.get('http://10.152.183.18:8000/FLSe/info')
        latest_gl_model_v = res.json()['Server_Status']['GL_Model_V']
        model_list = os.listdir('/model')
        if f'model_V{latest_gl_model_v}.h5' in model_list:
            logging.info('latest model load_weights')
            model.load_weights(f'/model/model_V{latest_gl_model_v}.h5')
            # return model
        else:
            logging.info('NO latest model load_weights')
            pass
    except Exception as e:
        logging.info('[E][PC0001] learning', e)
        status.FL_client_fail = True
        await notify_fail()
        status.FL_client_fail = False

    await flower_client_start()

    return status

async def flower_client_start():
    logging.info('FL learning')
    global status
    global model

    # 환자별로 partition 분리 => 개별 클라이언트 적용
    (x_train, y_train), (x_test, y_test), label_count = load_partition()

    try:
        loop = asyncio.get_event_loop()
        client = PatientClient(model, x_train, y_train, x_test, y_test)
        # assert type(client).get_properties == fl.client.NumPyClient.get_properties
        logging.info(f'fl-server-ip: {status.FL_server_IP}')
        # fl.client.start_numpy_client(server_address=status.FL_server_IP, client=client)
        await asyncio.sleep(10) # FL-Server 켜질때 까지 잠시 대기
        request = partial(fl.client.start_numpy_client, server_address=status.FL_server_IP, client=client)
        await loop.run_in_executor(None, request)
        
        await asyncio.sleep(30) # excute 수행 시간동안 잠시 대기

        # inform_Payload = {
        #     'FL_learning_complete': True
        # }
        # res = requests.put('http://localhost:8003/training', data=json.dumps(inform_Payload))

        # if res.status_code ==200:
        #     logging.info('fl-client 정상작동 완료')
        # else:
        #     logging.info('http://localhost:8003/training Requests 오류')

        logging.info('fl learning finished')
        await model_save()
        logging.info('model_save')
        del client, request
        logging.info('fl client, request delete')
    except Exception as e:
        await notify_fail()
        logging.info('[E][PC0002] learning', e)
        # status.FL_client_fail = True
        # # await notify_fail()
        
        # status.FL_client_fail = False
        # raise e
    return status

async def model_save():
    
    global model
    try:
        # # client_manager 주소
        client_res = requests.get('http://localhost:8003/info/')

        # # 최신 global model 버전
        latest_gl_model_v = client_res.json()['GL_Model_V']
        
        # 다음 global model 버전
        next_gl_model = latest_gl_model_v + 1

        model.save('/model/model_V%s.h5'%next_gl_model)
        await notify_fin()
        model=None
    except Exception as e:
        logging.info('[E][PC0003] learning', e)
        status.FL_client_fail = True
        await notify_fail()
        status.FL_client_fail = False

    return status

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
        print('notify_fin error: ', r.content)

    return status

# client manager에서 train fail 정보 확인
async def notify_fail():
    logging.info('notify_fail start')
    global status
    status.FL_client_start = False
    try:
        logging.info('notify_fail try 문장 안 접근')
        loop = asyncio.get_event_loop()
        logging.info('notify_fail loop 통과')
        future1 = loop.run_in_executor(None, requests.get, 'http://localhost:8003/trainFail')
        logging.info('notify_fail future1 통과')
        r = await future1
        logging.info('notify_fail complete')
        if r.status_code == 200:
            logging.info('trainFin')
        else:
            logging.info('notify_fail error: ', r.content)
    except Exception as e:
        logging.info('[E] notify_fail: ', e)
        
    return status

def load_partition():
    # Load the dataset partitions
    data, p_list = dataset.data_load()

    # label 까지 포함 dataframe
    train_df, test_df = train_test_split(data.iloc[:,1:], test_size=0.2)

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
    # wandb login and init
    # wandb.login(key='6266dbc809b57000d78fb8b163179a0a3d6eeb37')
    # wandb.init(entity='ccl-fl', project='client_flower', name= 'client %s_V%s'%(client_num,next_gl_model), dir='/app')

    try:
        # client api 생성 => client manager와 통신하기 위함
        uvicorn.run("app:app", host='0.0.0.0', port=8002, reload=True)

        # client FL 수행
        # main()
        
    finally:

        # wandb 종료
        # wandb.finish()

        # FL client out
        requests.get('http://localhost:8003/flclient_out')
        logging.info('%s client close'%client_num)
