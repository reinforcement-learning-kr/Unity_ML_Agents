# 라이브러리 불러오기
import tensorflow as tf
import tensorflow.layers as layer
import numpy as np
import random

from collections import deque
from mlagents.envs import UnityEnvironment

# DDPG를 위한 파라미터 값 세팅
train_mode = True

mem_maxlen = 1000
TAU = 1e-3 # Target Network hyperparameter for soft update
LRA = 5e-4 # Actor learning rate
LRC = 5e-4 # Critic learning rate

discount_factor = 0.99

state_size = 3
action_size = 3
hidden_size = 32
batch_size = 64

max_episode_step = 100
episode_length = 100000
noise_option = 'ou_noise' # or None

# 모델 저장 및 불러오기 경로
save_path = ".ddpg"
load_path = ".ddpg"
load_model = False

# 유니티 환경 경로
env_name = "../envs/sucks/Drone"
logdir = "../Summary/ddpg"

print_episode_interval = 10
save_interval = 500

# OU_noise 클래스 -> ou noise 정의 및 노이즈 수치(탐색) 결정
class OU_noise:
    '''generate OU noise for continuous action space
    '''
    def __init__(self, action_size, mu=0, theta=0.1, sigma=0.2, dt=1e-2):
        '''ou noise initializer
        
        Arguments:
            action_size {int} -- size of action space
        
        Keyword Arguments:
            mu {int} -- mean of action space (default: {0})
            theta {float} -- regression speed to mean (default: {0.1})
            sigma {float} -- variation of noise (default: {0.2})
            dt {float} -- apply ratio of previous noise (default: {1e-2})
        '''
        self.action_size = action_size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.state = np.zeros(self.action_size)
        self.reset()

    def reset(self):
        '''initialize ou noise
        '''
        self.state = np.zeros(self.action_size)

    def noise(self):
        '''generate noise
        
        Returns:
            vector -- generate noise for action space
        '''
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(1, self.action_size) * np.sqrt(self.dt)
        self.state = x + dx
        return self.state

# ActorModel 클래스 -> DDPG Agent가 Action을 결정하는 네트워크
class ActorModel:
    '''Actor network description class
    '''
    def __init__(self, state_size, action_size, name, hidden_size):
        '''Actor network initializer
        
        Arguments:
            state_size {vector or int} -- discription of state size : image or vector
            action_size {int} -- action vector size
            name {str} -- network namespace
            hidden_size {int} -- hidden neuron size for network
        '''
        self.state_size = state_size
        self.action_size = action_size

        self.name = name

        with tf.variable_scope(name):
            self.observation = tf.placeholder(
                tf.float32, shape=[None, self.state_size], name="actor_observation")
            self.L1 = layer.dense(
                self.observation, hidden_size, activation=tf.nn.leaky_relu)
            self.L1 = tf.contrib.layers.batch_norm(self.L1, 
                                      center=True, scale=True, 
                                      is_training=train_mode)
            self.L2 = layer.dense(
                self.L1, hidden_size, activation=tf.nn.leaky_relu)
            self.action = layer.dense(
                self.L2, self.action_size, activation=tf.nn.tanh, name='actor_decide') * 0.1
        
        self.trainable_var = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

    # Actor모델 학습 함수 -> Policy Gradient
    def trainer(self, critic, batch_size):
        '''calculate gradient for action network
        
        Arguments:
            critic {tf network} -- critic network
            batch_size {int} -- batch size
        '''

        self.action_Grad = tf.gradients(critic.value, critic.action)
        self.policy_Grads = tf.gradients(
            ys=self.action, xs=self.trainable_var, grad_ys=self.action_Grad)
        
        for idx, grads in enumerate(self.policy_Grads):
            self.policy_Grads[idx] = -grads/batch_size
        
        self.Adam = tf.train.AdamOptimizer(LRA)
        self.Update = self.Adam.apply_gradients(
            zip(self.policy_Grads, self.trainable_var))

# CriticModel 클래스 -> state와 action에 대한 value를 결정
class CriticModel:
    '''Critic Network class
    '''
    def __init__(self, state_size, action_size, name, hidden_size):
        '''critic network initalizer
        
        Arguments:
            state_size {int or vector} -- state space : image or vector
            action_size {int} -- action vector space
            name {str} -- network name
            hidden_size {int} -- hidden neurons for ciritic network
        '''
        self.state_size = state_size
        self.action_size = action_size

        self.name = name
        
        with tf.variable_scope(name):
            self.observation = tf.placeholder(
                tf.float32, shape=[None, self.state_size], name="critic_observation")
            self.O1 = layer.dense(
                self.observation, hidden_size, activation=tf.nn.leaky_relu)
            self.action = tf.placeholder(
                tf.float32, shape=[None, self.action_size], name="critic_action")
            self.L1 = tf.concat([self.O1, self.action], axis=-1)
            self.L1 = layer.dense(
                self.L1, hidden_size, activation=tf.nn.leaky_relu)
            self.L1 = tf.contrib.layers.batch_norm(self.L1, 
                                      center=True, scale=True, 
                                      is_training=train_mode)
            self.L2 = layer.dense(
                self.L1, hidden_size, activation=tf.nn.leaky_relu)
            self.value = layer.dense(self.L2, 1, activation=None)
        
        # Mean square error를 대신하는 Huber Loss를 사용하여 value gradient 계산
        self.true_value = tf.placeholder(tf.float32, name='true_value')
        self.loss = tf.losses.huber_loss(self.true_value, self.value)
        self.Update = tf.train.AdamOptimizer(LRC).minimize(self.loss)

        self.trainable_var = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

# DDPGAgnet class는 Actor-Critic을 기반으로 Continous space 환경에 대해 학습하는 클래스
class DDPGAgent:
    def __init__(self, state_size, action_size, hidden_size,
            batch_size, mem_maxlen
        ):
        # soft target update를 위한 각 모델에 대해 2개 network 선언
        self.actor_model = ActorModel(state_size, action_size, 'actor', hidden_size)
        self.critic_model = CriticModel(state_size, action_size, 'critic', hidden_size)

        self.t_actor_model = ActorModel(state_size, action_size, 't_actor', hidden_size)
        self.t_critic_model = CriticModel(state_size, action_size, 't_critic', hidden_size)

        self.actor_model.trainer(self.critic_model, batch_size)

        self.noise_option = noise_option
        if noise_option is None:
            self.noiser = None
        elif noise_option == 'ou_noise':
            self.noiser = OU_noise(action_size)

        self.memory = deque(maxlen=mem_maxlen)

        self.batch_size = batch_size
    
        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        self.load_model = load_model
        self.Saver = tf.train.Saver()
        self.save_path = save_path
        self.load_path = load_path
        self.Summary, self.Merge = self.make_Summary()

        if self.load_model == True:
            ckpt = tf.train.get_checkpoint_state(self.save_path)
            self.Saver.restore(self.sess, ckpt.model_checkpoint_path)

        self.act_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor')
        self.t_act_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='t_actor')

        self.actor_soft_update = [tf.assign(target, (1-TAU) * target + TAU * origin) 
            for target, origin in zip(self.t_act_params, self.act_params)]

        self.cri_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic')
        self.t_cri_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='t_critic')

        self.critic_soft_update = [tf.assign(target, (1-TAU) * target + TAU * origin) 
            for target, origin in zip(self.t_cri_params, self.cri_params)]
        
    # Actor model에서 action을 예측하고, noise 추가 유무 설정
    def get_action(self, state, train_mode=True):
        action = self.sess.run(self.actor_model.action, feed_dict={self.actor_model.observation: state})

        if train_mode and noise_option=='ou_noise':
            return action + self.noiser.noise()
        else:
            return action

    # replay memory에 입력
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state[0], action[0], reward[0], next_state[0], done[0]))

    # model 저장
    def save_model(self):
        self.Saver.save(self.sess, self.save_path+'\model.ckpt')
    
    # replay memory를 통해 모델을 학습
    def train_model(self):
        mini_batch = random.sample(self.memory, self.batch_size)
        states = np.asarray([e[0] for e in mini_batch]) + 1e-6
        actions = np.asarray([e[1] for e in mini_batch])
        rewards = np.asarray([e[2] for e in mini_batch])
        next_states = np.asarray([e[3] for e in mini_batch]) + 1e-6
        dones = np.asarray([e[4] for e in mini_batch])
        
        target_critic_action_inputs = self.sess.run(self.t_actor_model.action, 
            feed_dict={self.t_actor_model.observation: next_states})
        target_values = self.sess.run(self.t_critic_model.value, 
            feed_dict={self.t_critic_model.observation: next_states, self.t_critic_model.action: target_critic_action_inputs})

        Q_targets = rewards + discount_factor * target_values * (1-dones)

        loss_val = self.sess.run(self.critic_model.loss, 
            feed_dict={self.critic_model.observation: states, self.critic_model.action: actions, self.critic_model.true_value: Q_targets}
        )
        
        # critic model soft update
        self.sess.run(self.critic_model.Update, 
            feed_dict={self.critic_model.observation: states, self.critic_model.action: actions, self.critic_model.true_value: Q_targets}
        )
        '''
        self.cri_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic')
        self.t_cri_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='t_critic')

        critic_soft_update = [tf.assign(target, (1-TAU) * target + TAU * origin) 
            for target, origin in zip(self.t_cri_params, self.cri_params)]
        '''
        self.sess.run(self.critic_soft_update)

        # actor model soft update
        action_for_train = self.sess.run(self.actor_model.action, feed_dict={
                                            self.actor_model.observation: states})
        self.sess.run([self.actor_model.Update, self.actor_model.policy_Grads], feed_dict={
                                   self.actor_model.observation: states, self.critic_model.observation: states, self.critic_model.action: action_for_train})
        '''
        self.act_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor')
        self.t_act_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='t_actor')

        actor_soft_update = [tf.assign(target, (1-TAU) * target + TAU * origin) 
            for target, origin in zip(self.t_act_params, self.act_params)]
        '''
        self.sess.run(self.actor_soft_update)
        
        return loss_val

    def make_Summary(self):
        self.summary_loss = tf.placeholder(dtype=tf.float32)
        self.summary_reward = tf.placeholder(dtype=tf.float32)
        tf.summary.scalar("loss", self.summary_loss)
        tf.summary.scalar("reward", self.summary_reward)
        return tf.summary.FileWriter(logdir=logdir, graph=self.sess.graph), tf.summary.merge_all()
        
    def Write_Summray(self, reward, loss, episode):
        self.Summary.add_summary(self.sess.run(self.Merge, feed_dict={
                                 self.summary_loss: loss, self.summary_reward: reward}), episode)

# Main 함수 -> DDQN 에이전트를 드론 환경에서 학습
if __name__ == '__main__':
    # 유니티 환경 설정
    env = UnityEnvironment(file_name=env_name)
    default_brain = env.brain_names[0]

    # DDPGAgnet 선언
    agent = DDPGAgent(state_size, action_size, hidden_size, 
        batch_size, mem_maxlen)

    losses = deque(maxlen=20)
    frame_count = 0

    # 각 에피소드를 거치며 replay memory에 저장
    for episode in range(episode_length):
        episode_rewards = []
        env_info = env.reset(train_mode=train_mode)[default_brain]

        state = env_info.vector_observations
        
        for _ in range(max_episode_step):
            frame_count += 1
            action = agent.get_action(state, train_mode)
            action = action
            env_info = env.step(action)[default_brain]
            next_state = env_info.vector_observations
            temp_reward = env_info.rewards
            
            reward = temp_reward
            done = env_info.local_done
            episode_rewards.append(reward[0])
            if train_mode:
                agent.append_sample(state, action, reward, next_state, done)
            state = next_state

            if done[0]:
                break
        
        # 일정 이상 memory가 쌓이면 학습
        if train_mode and len(agent.memory) > agent.batch_size:
            loss = agent.train_model()
            losses.append(loss)

        # 일정 이상의 episode를 진행 시 log 출력
        if episode % print_episode_interval == 0:
            print(f"episode({episode}) - reward: {np.mean(episode_rewards):.2f} loss: {np.mean(losses):.4f} mem_len {len(agent.memory)}")
            agent.Write_Summray(np.mean(episode_rewards), np.mean(losses), episode)
        
        # 일정 이상의 episode를 진행 시 현재 모델 저장
        if train_mode and episode % save_interval == 0 and episode != 0:
            print("model saved")
            agent.save_model()

    env.close()