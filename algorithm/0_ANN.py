# 라이브러리 불러오기
import tensorflow as tf
import numpy as np
import datetime

# 파라미터 설정하기
algorithm = 'ANN'

data_size = 13

load_model = False

batch_size = 32
num_epoch = 500

learning_rate = 1e-3

date_time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")

save_path = "../saved_models/" + date_time + "_" + algorithm
load_path = "../saved_models/20190312_11_12_35_ANN/model/model " 

# boston_housing 데이터셋 불러오기 ((x_train, y_train), (x_test, y_test))
boston_housing = tf.keras.datasets.boston_housing.load_data(path='boston_housing.npz')

x_train = boston_housing[0][0]
y_train = np.reshape(boston_housing[0][1],(-1,1))

x_train, x_valid = x_train[:len(x_train)*8//10], x_train[len(x_train)*8//10:]
y_train, y_valid = y_train[:len(y_train)*8//10], y_train[len(y_train)*8//10:]

x_test = boston_housing[1][0]
y_test = np.reshape(boston_housing[1][1],(-1,1))

# 네트워크 구조 정의, 손실 함수 정의 및 학습 수행 
class Model():
    def __init__(self):

        # 입력 및 실제값 
        self.x_input  = tf.placeholder(tf.float32, shape = [None, data_size])
        self.y_target = tf.placeholder(tf.float32, shape=[None, 1])

        # 네트워크
        self.fc1 = tf.layers.dense(self.x_input, 128, activation=tf.nn.tanh)
        self.fc2 = tf.layers.dense(self.fc1, 128, activation=tf.nn.tanh)
        self.fc3 = tf.layers.dense(self.fc2, 128, activation=tf.nn.tanh)
        self.out = tf.layers.dense(self.fc3, 1)

        # 손실 함수 계산 및 학습 수행
        self.loss = tf.losses.mean_squared_error(self.y_target, self.out)
        self.UpdateModel = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

# 인공 신경망 학습을 위한 다양한 함수들 
class ANN():
    def __init__(self):
        self.model = Model()
        # Tensorflow 세션 초기화
        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        self.Saver = tf.train.Saver()
        self.Train_Summary, self.Val_Summary, self.Merge = self.Make_Summary()

        # 모델 불러오기
        if load_model == True:
            self.Saver.restore(self.sess, load_path)

    # 모델 학습
    def train_model(self, data_x, data_y, batch_idx):
        len_data = data_x.shape[0]

        if batch_idx + batch_size < len_data:
            batch_x = data_x[batch_idx : batch_idx + batch_size, :]
            batch_y = data_y[batch_idx : batch_idx + batch_size, :]
        else:
            batch_x = data_x[batch_idx : len_data, :]
            batch_y = data_y[batch_idx : len_data, :]

        _, loss, output = self.sess.run([self.model.UpdateModel, self.model.loss, self.model.out],
                                                feed_dict={self.model.x_input: batch_x, 
                                                            self.model.y_target: batch_y})
        return loss

    # 알고리즘 성능 테스트
    def test_model(self, data_x, data_y):
        loss, output = self.sess.run([self.model.loss, self.model.out],
                                      feed_dict={self.model.x_input: data_x, 
                                                 self.model.y_target: data_y})
        return loss

    # 모델 저장
    def save_model(self):
        self.Saver.save(self.sess, save_path + "/model/model.ckpt")

    # 텐서보드에 손실 함수값 및 정확도 저장
    def Make_Summary(self):
        self.summary_loss = tf.placeholder(dtype=tf.float32)
        tf.summary.scalar("loss", self.summary_loss)
        Train_Summary = tf.summary.FileWriter(logdir=save_path+"/train")
        Val_Summary = tf.summary.FileWriter(logdir=save_path+"/val")
        Merge = tf.summary.merge_all()
        return Train_Summary, Val_Summary, Merge

    def Write_Summray(self, train_loss, val_loss, batch):
        self.Train_Summary.add_summary(
            self.sess.run(self.Merge, feed_dict={self.summary_loss: train_loss}), batch)
        self.Val_Summary.add_summary(
            self.sess.run(self.Merge, feed_dict={self.summary_loss: val_loss}), batch)

if __name__ == '__main__':
    ann = ANN()
    data_train = np.zeros([x_train.shape[0], data_size + 1])
    data_train[:, :data_size] = x_train
    data_train[:, data_size:] = y_train

    # 학습 수행 
    for epoch in range(num_epoch):
        
        train_loss_list = []
        val_loss_list = []

        # 데이터를 섞은 후 입력과 실제값 분리
        np.random.shuffle(data_train)
        train_x = data_train[:, :data_size]
        train_y = data_train[:, data_size:]

        # 학습 수행, 손실 함수 값 계산 및 텐서보드에 값 저장
        for batch_idx in range(0, x_train.shape[0], batch_size):
            train_loss = ann.train_model(train_x, train_y, batch_idx)
            val_loss = ann.test_model(x_valid, y_valid)
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            
        # 학습 진행 상황 출력 
        print("Epoch: {} / Train loss: {:.5f} / Val loss: {:.5f} "
                .format(epoch+1, np.mean(train_loss_list), np.mean(val_loss_list)))
        ann.Write_Summray(np.mean(train_loss_list), np.mean(val_loss_list), epoch)

    # 테스트 수행 
    test_loss = ann.test_model(x_test, y_test)
    print('----------------------------------')
    print('Test Loss: {:.5f}'.format(test_loss))
    # 모델 저장

    ann.save_model()
    print("Model is saved in {}".format(save_path + "/model/model"))

