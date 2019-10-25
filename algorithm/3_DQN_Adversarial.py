# 라이브러리 불러오기
import numpy as np
import random
import datetime
import time
import tensorflow as tf
from collections import deque
from mlagents.envs import UnityEnvironment

# 적대적인 DQN을 위한 파라미터 세팅
state_size = 12 * 3
action_size = 3

load_model = False
train_mode = True

batch_size = 32
mem_maxlen = 50000
discount_factor = 0.99
learning_rate = 0.00025

run_episode = 10000
test_episode = 100

start_train_episode = 500

target_update_step = 10000
print_interval = 20
save_interval = 5000

epsilon_init = 1.0
epsilon_min = 0.1

date_time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")

# 유니티 환경 경로 
game = "Pong"
env_name = "../env/" + game + "/Windows/" + game

# 모델 저장 및 불러오기 경로
save_path = "../saved_models/" + game + "/" + date_time + "_DQN"
load_path = "../saved_models/" + game + "/20190220-16-27-5_DQN/model/model"

# Model 클래스 -> 네트워크 정의 및 손실함수 설정, 네트워크 최적화 알고리즘 결정
class Model():
    def __init__(self, model_name):
        self.input = tf.placeholder(shape=[None, state_size], dtype=tf.float32)

        # 네트워크 구축 -> 완전연결층만을 이용하여 구성
        with tf.variable_scope(name_or_scope=model_name):
            self.fc1 = tf.layers.dense(self.input,512,activation=tf.nn.relu)
            self.fc2 = tf.layers.dense(self.fc1,512,activation=tf.nn.relu)
            self.fc3 = tf.layers.dense(self.fc2,512,activation=tf.nn.relu)
            self.Q_Out = tf.layers.dense(self.fc3,action_size,activation=None)
        self.predict = tf.argmax(self.Q_Out, 1)

        self.target_Q = tf.placeholder(shape=[None, action_size], dtype=tf.float32)

        # 손실함수값 계산 및 네트워크 학습 수행 
        self.loss = tf.losses.mean_squared_error(self.target_Q, self.Q_Out)
        self.UpdateModel = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        self.trainable_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, model_name)

# DQNAgent 클래스 -> DQN 알고리즘을 위한 다양한 함수 
class DQNAgent():
    def __init__(self):
        self.model1 = Model("Q1")
        self.target_model1 = Model("target1")
        self.model2 = Model("Q2")
        self.target_model2 = Model("target2")

        self.memory1 = deque(maxlen=mem_maxlen)
        self.memory2 = deque(maxlen=mem_maxlen)
     
        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        self.epsilon = epsilon_init

        self.Saver = tf.train.Saver()
        self.Summary, self.Merge = self.Make_Summary()

        if load_model == True:
            self.Saver.restore(self.sess, load_path)

    # Epsilon greedy 기법에 따라 액션 결정
    def get_action(self, state1, state2, train_mode=True):
        if self.epsilon > np.random.rand():
            # 랜덤하게 액션 결정
            random_action1 = np.random.randint(0, action_size)
            random_action2 = np.random.randint(0, action_size)
            return random_action1, random_action2
        else:
            # 네트워크 연산에 따라 액션 결정
            predict1 = self.sess.run(self.model1.predict, 
                                     feed_dict={self.model1.input: [state1]})
            predict2 = self.sess.run(self.model2.predict, 
                                     feed_dict={self.model2.input: [state2]})
            return np.asscalar(predict1), np.asscalar(predict2)

    # 리플레이 메모리에 데이터 추가 (상태, 액션, 보상, 다음 상태, 게임 종료 여부)
    def append_sample(self, data1, data2):
        self.memory1.append((data1[0], data1[1], data1[2], data1[3], data1[4]))
        self.memory2.append((data2[0], data2[1], data2[2], data2[3], data2[4]))

    # 네트워크 모델 저장 
    def save_model(self):
        self.Saver.save(self.sess, save_path + "/model/model")

    # 학습 수행 
    def train_model(self, model, target_model, memory, done):
        # Epsilon 값 감소 
        if done:
            if self.epsilon > epsilon_min:
                self.epsilon -= 0.5 / (run_episode - start_train_episode)

        # DQN의 학습을 위한 미니 배치 데이터 샘플링
        mini_batch = random.sample(memory, batch_size)

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for i in range(batch_size):
            states.append(mini_batch[i][0])
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states.append(mini_batch[i][3])
            dones.append(mini_batch[i][4])

        # DQN의 학습을 위한 타겟값 계산 
        target = self.sess.run(model.Q_Out, feed_dict={model.input: states})
        target_val = self.sess.run(target_model.Q_Out, 
                                    feed_dict={target_model.input: next_states})

        for i in range(batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + discount_factor * np.amax(target_val[i])

        # DQN 알고리즘 학습 수행 및 loss 계산 
        _, loss = self.sess.run([model.UpdateModel, model.loss],
                                feed_dict={model.input: states, 
                                           model.target_Q: target})

        return loss

    # 타겟 네트워크 업데이트 
    def update_target(self, model, target_model):
        for i in range(len(model.trainable_var)):
            self.sess.run(target_model.trainable_var[i].assign(model.trainable_var[i]))

    # 텐서 보드에 기록할 값 설정 및 데이터 기록 
    def Make_Summary(self):
        self.summary_loss1 = tf.placeholder(dtype=tf.float32)
        self.summary_reward1 = tf.placeholder(dtype=tf.float32)
        self.summary_loss2 = tf.placeholder(dtype=tf.float32)
        self.summary_reward2 = tf.placeholder(dtype=tf.float32)
        
        tf.summary.scalar("loss1", self.summary_loss1)
        tf.summary.scalar("reward1", self.summary_reward1)
        tf.summary.scalar("loss2", self.summary_loss2)
        tf.summary.scalar("reward2", self.summary_reward2)

        return tf.summary.FileWriter(logdir=save_path, graph=self.sess.graph), tf.summary.merge_all()

    def Write_Summray(self, reward1, loss1, reward2, loss2, episode):
        self.Summary.add_summary(
            self.sess.run(self.Merge, feed_dict={self.summary_loss1: loss1, 
                                                 self.summary_reward1: reward1, 
                                                 self.summary_loss2: loss2, 
                                                 self.summary_reward2: reward2}), episode)

# Main 함수 -> 전체적으로 적대적인 DQN 알고리즘을 진행 
if __name__ == '__main__':
    # 유니티 환경 설정
    env = UnityEnvironment(file_name=env_name)

    # 유니티 브레인 설정 
    brain_name1 = env.brain_names[0]
    brain_name2 = env.brain_names[1]

    brain1 = env.brains[brain_name1]
    brain2 = env.brains[brain_name2]

    env_info = env.reset(train_mode=train_mode)
    
    # DQNAgent 클래스를 agent로 정의 
    agent = DQNAgent()

    step = 0

    rewards1 = []
    losses1 = []
    rewards2 = []
    losses2 = []

    # 게임 진행 반복문 
    for episode in range(run_episode + test_episode):
        if episode == run_episode:
            train_mode = False
        
        # 유니티 환경 리셋 및 학습 모드 설정 
        env_info = env.reset(train_mode=train_mode)
        
        done = False

        # 첫번째 에이전트의 상태, episode_rewards, done 초기화 
        state1 = env_info[brain_name1].vector_observations[0]
        episode_rewards1 = 0
        done1 = False

        # 두번째 에이전트의 상태, episode_rewards, done 초기화
        state2 = env_info[brain_name2].vector_observations[0]
        episode_rewards2 = 0
        done2 = False

        # 한 에피소드를 진행하는 반복문 
        while not done:
            step += 1

            # 액션 결정 및 유니티 환경에 액션 적용 
            action1, action2 = agent.get_action(state1, state2, train_mode)
            env_info = env.step(vector_action = {brain_name1: [action1], brain_name2: [action2]})

            # 첫번째 에이전트에 대한 다음 상태, 보상, 게임 종료 정보 취득 
            next_state1 = env_info[brain_name1].vector_observations[0]
            reward1 = env_info[brain_name1].rewards[0]
            episode_rewards1 += reward1
            done1 = env_info[brain_name1].local_done[0]

            # 두번째 에이전트에 대한 다음 상태, 보상, 게임 종료 정보 취득
            next_state2 = env_info[brain_name2].vector_observations[0]
            reward2 = env_info[brain_name2].rewards[0]
            episode_rewards2 += reward2
            done2 = env_info[brain_name2].local_done[0]

            done = done1 or done2

            # 학습 모드인 경우 리플레이 메모리에 데이터 저장 
            if train_mode:
                data1 = [state1, action1, reward1, next_state1, done1]
                data2 = [state2, action2, reward2, next_state2, done2]
                
                agent.append_sample(data1, data2)
            else:
                time.sleep(0.02)
                agent.epsilon = 0.0

            # 상태 정보 업데이트 
            state1 = next_state1
            state2 = next_state2

            if episode > start_train_episode and train_mode:
                # 학습 수행 
                loss1 = agent.train_model(agent.model1, agent.target_model1, agent.memory1, done)
                loss2 = agent.train_model(agent.model2, agent.target_model2, agent.memory2, done)
                losses1.append(loss1)
                losses2.append(loss2)

                # 타겟 네트워크 업데이트 
                if step % (target_update_step) == 0:
                    agent.update_target(agent.model1, agent.target_model1)
                    agent.update_target(agent.model2, agent.target_model2)

        rewards1.append(episode_rewards1)
        rewards2.append(episode_rewards2)

        # 게임 진행 상황 출력 및 텐서 보드에 보상과 손실함수값 기록 
        if episode % print_interval == 0 and episode != 0:
            print("step: {} / episode: {} / epsilon: {:.3f}".format(step, episode, agent.epsilon))
            print("reward1: {:.2f} / loss1: {:.4f} / reward2: {:.2f} / loss2: {:.4f}".format(
                  np.mean(rewards1), np.mean(losses1), np.mean(rewards2), np.mean(losses2)))
            print('------------------------------------------------------------')
            
            agent.Write_Summray(np.mean(rewards1), np.mean(losses1), 
                                np.mean(rewards2), np.mean(losses2), episode)
            rewards1 = []
            losses1 = []
            rewards2 = []
            losses2 = []

        # 네트워크 모델 저장 
        if episode % save_interval == 0 and episode != 0:
            agent.save_model()
            print("Save Model {}".format(episode))

    env.close()
