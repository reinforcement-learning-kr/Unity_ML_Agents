# 라이브러리 불러오기
import tensorflow as tf
import numpy as np
import datetime

# 파라미터 설정하기
algorithm = 'CNN'

img_size   = 28
data_size  = img_size**2

num_label = 10

batch_size = 256
num_epoch  = 50


learning_rate = 0.00025

date_time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")

save_path = "../saved_models/" + date_time + "_" + algorithm
load_path = "../saved_models/20190312-11-12-35_CNN/model/model" 

load_model = False

# MNIST 데이터셋 불러오기 ((x_train, y_train), (x_test, y_test))
mnist = tf.keras.datasets.mnist.load_data(path='mnist.npz')

x_train = mnist[0][0]
y_train = mnist[0][1]
x_test  = mnist[1][0]
y_test  = mnist[1][1]

x_test  = np.reshape(x_test, [-1, img_size, img_size, 1])

# 실제값을 onehot으로 변환
y_train_onehot = np.zeros([y_train.shape[0], num_label])
y_test_onehot  = np.zeros([y_test.shape[0], num_label])

for i in range(y_train.shape[0]):
    y_train_onehot[i, y_train[i]] = 1

for i in range(y_test.shape[0]):
    y_test_onehot[i, y_test[i]] = 1

# Validation Set 생성
x_train, x_val = x_train[:len(x_train)*9//10], x_train[len(x_train)*9//10:]
y_train_onehot, y_val_onehot = y_train_onehot[:len(y_train)*9//10], y_train_onehot[len(y_train)*9//10:]

x_val = np.reshape(x_val, [-1, img_size, img_size, 1])

# 네트워크 구조 정의, 손실 함수 정의 및 학습 수행 
class Model():
    def __init__(self):

        # 입력 및 실제값 
        self.x_input  = tf.placeholder(tf.float32, shape = [None, img_size, img_size, 1])
        self.x_normalize = (self.x_input - (255.0/2)) / (255.0/2)

        self.y_target = tf.placeholder(tf.float32, shape=[None, num_label])

        # 네트워크 (Conv-> 2, 은닉층 -> 1)
        self.conv1 = tf.layers.conv2d(self.x_normalize, filters=32, activation=tf.nn.relu, 
                                      kernel_size=[3,3], strides=[2,2], padding="same")
        self.conv2 = tf.layers.conv2d(self.conv1, filters=64, activation=tf.nn.relu, 
                                      kernel_size=[3,3], strides=[2,2], padding="same")

        self.flat = tf.layers.flatten(self.conv2)

        self.fc1 = tf.layers.dense(self.flat, 128, activation=tf.nn.relu)
        self.out = tf.layers.dense(self.fc1, num_label)

        # 손실 함수 계산 및 학습 수행
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_target, 
                                                                              logits=self.out))
        self.UpdateModel = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

# CNN 학습을 위한 다양한 함수들 
class CNN():
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
            batch_x = data_x[batch_idx : batch_idx + batch_size, :, :, :]
            batch_y = data_y[batch_idx : batch_idx + batch_size, :]
        else:
            batch_x = data_x[batch_idx : len_data, :, :, :]
            batch_y = data_y[batch_idx : len_data, :]

        _, loss, output = self.sess.run([self.model.UpdateModel, self.model.loss, self.model.out],
                                         feed_dict={self.model.x_input: batch_x, 
                                                    self.model.y_target: batch_y})

        accuracy = self.get_accuracy(output, batch_y)

        return loss, accuracy


    # 알고리즘 성능 테스트
    def test_model(self, data_x, data_y):
        loss, output = self.sess.run([self.model.loss, self.model.out],
                                      feed_dict={self.model.x_input: data_x, 
                                                 self.model.y_target: data_y})
        
        accuracy = self.get_accuracy(output, data_y)

        return loss, accuracy     

    # 정확도 계산
    def get_accuracy(self, pred, label):
        num_correct = 0.0
        for i in range(label.shape[0]):
            if np.argmax(label[i,:]) == np.argmax(pred[i,:]):
                num_correct += 1.0

        accuracy = num_correct / label.shape[0]

        return accuracy

    # 모델 저장
    def save_model(self):
        self.Saver.save(self.sess, save_path + "/model/model")

    # 텐서보드에 손실 함수값 및 정확도 저장
    def Make_Summary(self):
        self.summary_loss     = tf.placeholder(dtype=tf.float32)
        self.summary_acc      = tf.placeholder(dtype=tf.float32)

        tf.summary.scalar("Loss", self.summary_loss)
        tf.summary.scalar("Accuracy", self.summary_acc)

        Train_Summary = tf.summary.FileWriter(logdir=save_path+"/train")
        Val_Summary = tf.summary.FileWriter(logdir=save_path+"/val")
        Merge = tf.summary.merge_all()

        return Train_Summary, Val_Summary, Merge

    def Write_Summray(self, accuracy, loss, accuracy_val, loss_val, batch):
        self.Train_Summary.add_summary(
            self.sess.run(self.Merge, feed_dict={self.summary_loss: loss, 
                                                 self.summary_acc: accuracy}), batch)
        self.Val_Summary.add_summary(
            self.sess.run(self.Merge, feed_dict={self.summary_loss: loss_val,
                                                 self.summary_acc: accuracy_val}), batch)

if __name__ == '__main__':

    cnn = CNN()

    data_train = np.zeros([len(x_train), data_size + num_label])
    data_train[:, :data_size] = np.reshape(x_train, [-1, data_size])
    data_train[:, data_size:] = y_train_onehot 


    batch_num = 0

    loss_list = []
    acc_list  = []
    loss_val_list = []
    acc_val_list = []

    # 학습 수행 
    for epoch in range(num_epoch):

        # 데이터를 섞은 후 입력과 실제값 분리
        np.random.shuffle(data_train)

        train_x = data_train[:, :data_size]
        train_x = np.reshape(train_x, [-1, img_size, img_size, 1])

        train_y = data_train[:, data_size:]

        # 학습 수행, 손실 함수 값 계산 및 텐서보드에 값 저장
        for batch_idx in range(0, x_train.shape[0], batch_size):
            loss, accuracy = cnn.train_model(train_x, train_y, batch_idx)
            loss_val, accuracy_val = cnn.test_model(x_val, y_val_onehot)

            loss_list.append(loss)
            acc_list.append(accuracy)
            loss_val_list.append(loss_val)
            acc_val_list.append(accuracy_val)

            cnn.Write_Summray(accuracy, loss, accuracy_val, loss_val, batch_num)

            batch_num += 1

        # 학습 진행 상황 출력 
        print("Epoch: {} / Loss: {:.5f} / Val Loss: {:.5f} / Acc: {:.5f} / Val Acc: {:.5f}"
              .format(epoch+1, np.mean(loss_list), np.mean(loss_val_list), np.mean(acc_list), 
                     np.mean(acc_val_list)))

        loss_list = []
        acc_list  = []
        loss_val_list = []
        acc_val_list = []

    # 테스트 수행 
    loss_test, accuracy_test = cnn.test_model(x_test, y_test_onehot)
    print('----------------------------------')
    print('Test Accuracy: {:.3f}'.format(accuracy_test))
    print('Test Loss: {:.5f}'.format(loss_test))

    # 모델 저장
    cnn.save_model()
    print("Model is saved in {}".format(save_path + "/model/model"))
