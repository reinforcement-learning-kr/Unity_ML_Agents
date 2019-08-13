# 라이브러리 불러오기
import numpy as np
import random
import datetime
import time
import tensorflow as tf
from mlagents.envs import UnityEnvironment
from mlagents.trainers.demo_loader import demo_to_buffer

# Behavioral Cloning 학습 및 시험 파라미터 값 세팅
state_size = 30 * 2
action_size = 5

load_model = True
train_mode = False

batch_size = 1024
learning_rate = 0.001

train_epochs = 1000
test_episode = 10

print_interval = 1

# 닷지 환경 설정 (공 속도 = 2, 공 갯수 = 15, 공 유도 랜덤 수준 = 0.2 공 생성 랜덤 시드 = 77, 에이전트 속도 = 30)
env_config = {"ballSpeed": 2, "ballNum": 15, "ballRandom": 0.2, "randomSeed": 77, "agentSpeed": 30}

date_time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")

# 유니티 환경 경로
game = "Dodge"
env_name = "../env/" + game + "/Windows/" + game

# 전문가 실행 데이터 경로
demo_path = '../UnitySDK/Assets/Demonstrations/DodgeRecording_2.demo'

# 모델 저장 및 불러오기 경로
save_path = "../saved_models/" + game + "/" + date_time + "_BC"
load_path = "../saved_models/" + game + "/" + "20190814-02-39-38_BC/model/model"

# Model 클래스 -> 네트워크 정의 및 Loss 설정, 네트워크 최적화 알고리즘 결정
class Model():
    def __init__(self, model_name):        
        self.inputs = tf.placeholder(shape=[None, state_size], dtype=tf.float32)
        self.labels = tf.placeholder(shape=[None, 1], dtype=tf.int64)
        
        with tf.variable_scope(name_or_scope=model_name):
            # 3개의 FC layer
            self.fc1 = tf.layers.dense(self.inputs, 128, activation=tf.nn.relu)
            self.fc2 = tf.layers.dense(self.fc1, 128, activation=tf.nn.relu)
            self.fc3 = tf.layers.dense(self.fc2, 128, activation=tf.nn.relu)
            # sparse_softmax_cross_entropy를 위한 logits 출력값
            self.logits = tf.layers.dense(self.fc3, action_size)

        self.predict = tf.argmax(self.logits, axis=1)
        self.loss = tf.losses.sparse_softmax_cross_entropy(self.labels, self.logits)
        self.accuracy = tf.metrics.accuracy(self.labels, self.predict)

        self.UpdateModel = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        self.trainable_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, model_name)

# BCAgent 클래스 -> BC 알고리즘을 학습을 위한 함수들 정의
class BCAgent():
    def __init__(self):
        self.model = Model("BC")

        self.sess = tf.Session()
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

        self.Saver = tf.train.Saver()

        if train_mode:
            self.Summary, self.Merge = self.Make_Summary()

        if load_model:
            self.Saver.restore(self.sess, load_path)

    # 모델 액션 얻기
    def get_action(self, state):
        logits = self.sess.run(self.model.logits, feed_dict={self.model.inputs: state})
        policy = [np.exp(logit)/np.sum(np.exp(logits)) for logit in logits]
        action = np.random.choice(action_size,1,p=policy[0])
        return action

    # 네트워크 모델 저장
    def save_model(self):
        self.Saver.save(self.sess, save_path + "/model/model")

    # 학습 수행 전문가 데이터를 가지고 지도학습 실행
    def train_model(self, data, labels):
        _, loss, accuracy = self.sess.run(
            [self.model.UpdateModel, self.model.loss, self.model.accuracy],
            feed_dict={self.model.inputs: data, self.model.labels: labels})
        return loss, accuracy

    # 텐서 보드에 기록할 값 설정 및 데이터 기록
    def Make_Summary(self):
        self.summary_loss = tf.placeholder(dtype=tf.float32)
        self.summary_accuracy = tf.placeholder(dtype=tf.float32)
        tf.summary.scalar("loss", self.summary_loss)
        tf.summary.scalar("accuracy", self.summary_accuracy)
        return tf.summary.FileWriter(logdir=save_path, graph=self.sess.graph), tf.summary.merge_all()

    def Write_Summray(self, loss, accuracy, epoch):
        self.Summary.add_summary(
            self.sess.run(self.Merge, feed_dict={self.summary_loss: loss, self.summary_accuracy: accuracy}), epoch)

    # 전문가 실행 데이터 불러오기
    def load_demo(self):
        brain_params, demo_buffer = demo_to_buffer(demo_path,1)
        update_buffer = demo_buffer.update_buffer
        return update_buffer

# Main 함수 -> 전체적으로 BC 알고리즘을 진행
if __name__ == '__main__':
    # BCAgent 클래스를 agent로 정의
    agent = BCAgent()

    # 학습 모드
    if train_mode :
        # 전문가 데이터 버퍼로 가져오기
        buffer = agent.load_demo()

        # 버퍼에서 상태, 액션, 보상 가져오기
        state = np.array(buffer['vector_obs'])
        action = np.array(buffer['actions'])
        reward = np.array(buffer['rewards'])
        
        # -1 보상을 받은 인덱스를 찾아 삭제 (양질의 데이터를 가져오기 위함)
        neg_idx = np.where(reward == -1)
        
        state = np.delete(state, neg_idx, axis=0)
        action = np.delete(action, neg_idx, axis=0)

        losses = []
        accuracies = []

        for epoch in range(train_epochs):
            # 데이터의 연관성을 깨기 위해 셔플
            shuffle_idx = np.random.permutation(len(state))

            for i in range((len(state)//batch_size) + 1):
                if i == len(state)//batch_size:
                    idx = shuffle_idx[batch_size*i:]
                else:
                    idx = shuffle_idx[batch_size*i:batch_size*(i+1)]

                s, a = state[idx], action[idx]
                loss, accuracy = agent.train_model(s, a)
                losses.append(loss)
                accuracies.append(accuracy)
            
            # 게임 진행 상황 출력 및 텐서 보드에 loss, accuracy값 기록
            if epoch % print_interval == 0 and epoch != 0:
                print(f"epoch({epoch}) - loss: {np.mean(losses):.4f} accuracy: {np.mean(accuracies):.4f}")
                agent.Write_Summray(np.mean(losses), np.mean(accuracies), epoch)
                losses = []
                accuracies = []

        # 네트워크 모델 저장
        agent.save_model()
    else:
        env = UnityEnvironment(file_name=env_name)
        default_brain = env.brain_names[0]
        brain = env.brains[default_brain]

        env_info = env.reset(train_mode=train_mode, config=env_config)[default_brain]
        
        for episode in range(test_episode):
            done = False
            episode_rewards = 0
            while not done:
                action = agent.get_action(np.array([env_info.vector_observations[0]]))
                env_info = env.step(action)[default_brain]
                episode_rewards += env_info.rewards[0]
                done = env_info.local_done[0]

            print("Total reward this episode: {}".format(episode_rewards))

        env.close()