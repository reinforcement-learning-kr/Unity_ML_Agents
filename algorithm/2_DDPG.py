# 라이브러리 불러오기
import tensorflow as tf
import numpy as np
import random
import datetime
from collections import deque
from mlagents.envs import UnityEnvironment

# DDPG를 위한 파라미터 값 세팅
state_size = 9
action_size = 3

load_model = False
train_mode = True

batch_size = 128
mem_maxlen = 50000
discount_factor = 0.99
actor_lr = 1e-4
critic_lr = 5e-4
tau = 1e-3

mu = 0
theta = 1e-3
sigma = 2e-3

start_train_episode = 100
run_episode = 500
test_episode = 100

print_interval = 5
save_interval = 100

date_time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")

game = "Drone"
env_name = "../env/" + game + "/Windows/" + game

save_path = "../saved_models/" + game + "/" + date_time + "_DDPG"
load_path = "../saved_models/" + game + "/20190714-18-40-51-DDPG/model/model"

# OU_noise 클래스 -> ou noise 정의 및 파라미터 결정
class OU_noise:
    def __init__(self):
        self.reset()

    def reset(self):
        self.X = np.ones(action_size) * mu

    def sample(self):
        dx = theta * (mu - self.X) + sigma * np.random.randn(len(self.X))
        self.X += dx
        return self.X

# Actor 클래스 -> Actor 클래스를 통해 action을 출력
class Actor:
    def __init__(self, name):
        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, state_size])
            self.fc1 = tf.layers.dense(self.state, 128, activation=tf.nn.relu)
            self.fc2 = tf.layers.dense(self.fc1, 128, activation=tf.nn.relu)
            self.action = tf.layers.dense(self.fc2, action_size, activation=tf.nn.tanh)

        self.trainable_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)

# Critic 클래스 -> Critic 클래스를 통해 state와 action에 대한 Q-value를 출력
class Critic:
    def __init__(self, name):
        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, state_size])
            self.fc1 = tf.layers.dense(self.state, 128, activation=tf.nn.relu)
            self.action = tf.placeholder(tf.float32, [None, action_size])
            self.concat = tf.concat([self.fc1, self.action],axis=-1)
            self.fc2 = tf.layers.dense(self.concat, 128, activation=tf.nn.relu)
            self.fc3 = tf.layers.dense(self.fc2, 128, activation=tf.nn.relu)
            self.predict_q = tf.layers.dense(self.fc3, 1, activation=None)

        self.trainable_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)
    
# DDPGAgnet 클래스 -> Actor-Critic을 기반으로 학습하는 에이전트 클래스
class DDPGAgent:
    def __init__(self):
        self.actor = Actor("actor")
        self.critic = Critic("critic")
        self.target_actor = Actor("target_actor")
        self.target_critic = Critic("target_critic")
        
        self.target_q = tf.placeholder(tf.float32, [None, 1])
        critic_loss = tf.losses.mean_squared_error(self.target_q, self.critic.predict_q)
        self.train_critic = tf.train.AdamOptimizer(critic_lr).minimize(critic_loss)

        action_grad = tf.gradients(tf.squeeze(self.critic.predict_q), self.critic.action)
        policy_grad = tf.gradients(self.actor.action, self.actor.trainable_var, action_grad)

        for idx, grads in enumerate(policy_grad):
            policy_grad[idx] = -grads/batch_size
        self.train_actor = tf.train.AdamOptimizer(actor_lr).apply_gradients(
                                                            zip(policy_grad, self.actor.trainable_var))
  
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.Saver = tf.train.Saver()
        self.Summary, self.Merge = self.Make_Summary()
        self.OU = OU_noise()
        self.memory = deque(maxlen=mem_maxlen)

        self.soft_update_target = []
        for idx in range(len(self.actor.trainable_var)):
            self.soft_update_target.append(self.target_actor.trainable_var[idx].assign(
                ((1 - tau) * self.target_actor.trainable_var[idx].value())
                             + (tau * self.actor.trainable_var[idx].value())))
        for idx in range(len(self.critic.trainable_var)):
            self.soft_update_target.append(self.target_critic.trainable_var[idx].assign(
                ((1 - tau) * self.target_critic.trainable_var[idx].value())
                            + (tau * self.critic.trainable_var[idx].value())))
        
        init_update_target = []
        for idx in range(len(self.actor.trainable_var)):
            init_update_target.append(self.target_actor.trainable_var[idx].assign(
                                      self.actor.trainable_var[idx]))
        for idx in range(len(self.critic.trainable_var)):
            init_update_target.append(self.target_critic.trainable_var[idx].assign(
                                      self.critic.trainable_var[idx]))
        self.sess.run(init_update_target)

        if load_model == True:
            self.Saver.restore(self.sess, load_path)

    # Actor model에서 action을 예측하고 noise 설정
    def get_action(self, state):
        action = self.sess.run(self.actor.action, feed_dict={self.actor.state: state})
        noise = self.OU.sample()
        return action + noise if train_mode else action

    # replay memory에 입력
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # model 저장
    def save_model(self):
        self.Saver.save(self.sess, save_path + "/model/model")
    
    # replay memory를 통해 모델을 학습
    def train_model(self):
        mini_batch = random.sample(self.memory, batch_size)
        states = np.asarray([sample[0] for sample in mini_batch])
        actions = np.asarray([sample[1] for sample in mini_batch])
        rewards = np.asarray([sample[2] for sample in mini_batch])
        next_states = np.asarray([sample[3] for sample in mini_batch])
        dones = np.asarray([sample[4] for sample in mini_batch])

        target_actor_actions = self.sess.run(self.target_actor.action,
                                            feed_dict={self.target_actor.state: next_states})
        target_critic_predict_qs = self.sess.run(self.target_critic.predict_q,
                                                feed_dict={self.target_critic.state: next_states,
                                                self.target_critic.action: target_actor_actions})
        target_qs = np.asarray([reward + discount_factor * (1 - done) * target_critic_predict_q
                                for reward, target_critic_predict_q, done in zip(
                                                        rewards, target_critic_predict_qs, dones)])
        self.sess.run(self.train_critic, feed_dict={self.critic.state: states,
                                                    self.critic.action: actions,
                                                    self.target_q: target_qs})

        actions_for_train = self.sess.run(self.actor.action, feed_dict={self.actor.state: states})
        self.sess.run(self.train_actor, feed_dict={self.actor.state: states,
                                                   self.critic.state: states,
                                                   self.critic.action: actions_for_train})
                                                   
        self.sess.run(self.soft_update_target)

    def Make_Summary(self):
        self.summary_reward = tf.placeholder(tf.float32)
        self.summary_success_cnt = tf.placeholder(tf.float32)
        tf.summary.scalar("reward", self.summary_reward)
        tf.summary.scalar("success_cnt", self.summary_success_cnt)
        return tf.summary.FileWriter(logdir=save_path, graph=self.sess.graph), tf.summary.merge_all()

        
    def Write_Summray(self, reward, success_cnt, episode):
        self.Summary.add_summary(self.sess.run(self.Merge, feed_dict={
                                    self.summary_reward: reward, self.summary_success_cnt: success_cnt}), episode)

# Main 함수 -> DDPG 에이전트를 드론 환경에서 학습
if __name__ == '__main__':
    # 유니티 환경 설정
    env = UnityEnvironment(file_name=env_name)

    default_brain = env.brain_names[0]

    # DDPGAgnet 선언
    agent = DDPGAgent()
    rewards = deque(maxlen=print_interval)
    success_cnt = 0

    # 각 에피소드를 거치며 replay memory에 저장
    for episode in range(run_episode + test_episode):
        if episode == run_episode:
            train_mode = False

        env_info = env.reset(train_mode=train_mode)[default_brain]
        state = env_info.vector_observations[0]
        episode_rewards = 0
        done = False
        while not done:
            action = agent.get_action([state])[0]
            env_info = env.step(action)[default_brain]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            episode_rewards += reward
            if train_mode:
                agent.append_sample(state, action, reward, next_state, done)
            state = next_state

            # train_mode 이고 일정 이상 에피소드가 지나면 학습
            if episode > start_train_episode and train_mode :
                agent.train_model()
        success_cnt = success_cnt + 1 if reward == 1 else success_cnt
        rewards.append(episode_rewards)

        # 일정 이상의 episode를 진행 시 log 출력
        if episode % print_interval == 0 and episode != 0:
            print(f"episode({episode}) - reward: {np.mean(rewards):.3f} success_cnt: {success_cnt} mem_len: {len(agent.memory)}  ") 
            agent.Write_Summray(np.mean(rewards), success_cnt, episode)
            success_cnt = 0

        # 일정 이상의 episode를 진행 시 현재 모델 저장
        if train_mode and episode % save_interval == 0 and episode != 0:
            print("model saved")
            agent.save_model()

    env.close()