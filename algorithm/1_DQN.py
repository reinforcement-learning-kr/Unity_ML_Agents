import numpy as np
import random
import datetime
import time
import tensorflow as tf
import tensorflow.layers as layer
from collections import deque
from mlagents.envs import UnityEnvironment

state_size = [84, 84, 3]
action_size = 5

load_model = False
train_mode = True

batch_size = 32
mem_maxlen = 50000
discount_factor = 0.9
learning_rate = 0.00025

run_episode = 25000
test_episode = 1000

start_train_episode = 1000

target_update_step = 10000
print_interval = 100
save_interval = 5000

epsilon_init = 1.0
epsilon_min = 0.1

env_config = {"gridSize": 5, "numGoals": 1, "numBoxes": 1}

date_time = str(datetime.date.today()) + '_' + \
            str(datetime.datetime.now().hour) + '_' + \
            str(datetime.datetime.now().minute) + '_' + \
            str(datetime.datetime.now().second)

game = "Sokoban"
env_name = "../env/" + game + "/Windows/" + game

save_path = "../saved_models/" + game + "/" + date_time + "_DQN"
load_path = "../saved_models/" + game + "/2019-02-14_1_8_8_DQN/model/model.ckpt"

class Model():
    def __init__(self, model_name):
        self.state_size = state_size
        self.action_size = action_size

        self.input = tf.placeholder(shape=[None, self.state_size[0], 
                                                 self.state_size[1], 
                                                 self.state_size[2]],
                                                 dtype=tf.float32)
        self.input_normalize = (self.input - (255.0 / 2)) / (255.0 / 2)

        with tf.variable_scope(name_or_scope=model_name):
            self.conv1 = layer.conv2d(inputs=self.input_normalize,filters=32,
                                      activation=tf.nn.relu,kernel_size=[8,8],
                                      strides=[4,4],padding="SAME")
            self.conv2 = layer.conv2d(inputs=self.conv1,filters=64,
                                      activation=tf.nn.relu,kernel_size=[4,4],
                                      strides=[2,2],padding="SAME")
            self.conv3 = layer.conv2d(inputs=self.conv2,filters=64,
                                      activation=tf.nn.relu,kernel_size=[3,3],
                                      strides=[1,1],padding="SAME")

            self.flat = layer.flatten(self.conv3)

            self.fc1 = layer.dense(self.flat,512,activation=tf.nn.relu)
            self.Q_Out = layer.dense(self.fc1,self.action_size,activation=None)
        self.predict = tf.argmax(self.Q_Out, 1)

        self.target_Q = tf.placeholder(shape=[None, self.action_size], dtype=tf.float32)

        self.loss = tf.losses.huber_loss(self.target_Q, self.Q_Out)
        self.UpdateModel = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        self.trainable_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, model_name)

class DQNAgent():
    def __init__(self):
        self.state_size = state_size
        self.action_size = action_size

        self.model = Model("Q")
        self.target_model = Model("target")

        self.memory = deque(maxlen=mem_maxlen)

        self.batch_size = batch_size
        self.run_episode = run_episode

        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        self.epsilon = epsilon_init
        self.epsilon_min = epsilon_min
        self.discount_factor = discount_factor

        self.load_model = load_model
        self.Saver = tf.train.Saver()
        self.save_path = save_path
        self.load_path = load_path
        self.Summary, self.Merge = self.make_Summary()

        self.update_target()

        if self.load_model == True:
            self.Saver.restore(self.sess, self.load_path)

    def get_action(self, state, train_mode=True):
        if train_mode == True and self.epsilon > np.random.rand():
            return np.random.randint(0, self.action_size)
        else:
            predict = self.sess.run(self.model.predict, feed_dict={self.model.input: state})
            return np.asscalar(predict)

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state[0], action, reward, next_state[0], done))

    def save_model(self):
        self.Saver.save(self.sess, self.save_path + "/model/model.ckpt")

    def train_model(self, done):
        if done:
            if self.epsilon > self.epsilon_min:
                self.epsilon -= 1 / (self.run_episode - start_train_episode)

        mini_batch = random.sample(self.memory, self.batch_size)

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for i in range(self.batch_size):
            states.append(mini_batch[i][0])
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states.append(mini_batch[i][3])
            dones.append(mini_batch[i][4])

        target = self.sess.run(self.model.Q_Out, feed_dict={self.model.input: states})
        target_val = self.sess.run(self.target_model.Q_Out, 
                                   feed_dict={self.target_model.input: next_states})

        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * np.amax(target_val[i])

        _, loss = self.sess.run([self.model.UpdateModel, self.model.loss],
                                feed_dict={self.model.input: states, 
                                           self.model.target_Q: target})
        return loss

    def update_target(self):
        for i in range(len(self.model.trainable_var)):
            self.sess.run(self.target_model.trainable_var[i].assign(self.model.trainable_var[i]))

    def make_Summary(self):
        self.summary_loss = tf.placeholder(dtype=tf.float32)
        self.summary_reward = tf.placeholder(dtype=tf.float32)
        tf.summary.scalar("loss", self.summary_loss)
        tf.summary.scalar("reward", self.summary_reward)
        return tf.summary.FileWriter(logdir=save_path, graph=self.sess.graph), tf.summary.merge_all()

    def Write_Summray(self, reward, loss, episode):
        self.Summary.add_summary(
            self.sess.run(self.Merge, feed_dict={self.summary_loss: loss, 
                                                 self.summary_reward: reward}), episode)

if __name__ == '__main__':

    env = UnityEnvironment(file_name=env_name)

    default_brain = env.brain_names[0]
    brain = env.brains[default_brain]

    agent = DQNAgent()

    step = 0
    rewards = []
    losses = []

    env_info = env.reset(train_mode=train_mode, config=env_config)[default_brain]

    for episode in range(run_episode + test_episode):
        if episode == run_episode:
            train_mode = False
            env_info = env.reset(train_mode=train_mode)[default_brain]
        
        state = np.uint8(255 * np.array(env_info.visual_observations[0]))
        episode_rewards = 0
        done = False

        while not done:
            step += 1

            action = agent.get_action(state, train_mode)
            env_info = env.step(action)[default_brain]

            next_state = np.uint8(255 * np.array(env_info.visual_observations[0]))
            reward = env_info.rewards[0]
            episode_rewards += reward
            done = env_info.local_done[0]

            if train_mode:
                agent.append_sample(state, action, reward, next_state, done)
            else:
                time.sleep(0.01)
                agent.epsilon = 0.0

            state = next_state

            if episode > start_train_episode and train_mode:
                loss = agent.train_model(done)
                losses.append(loss)

                if step % (target_update_step) == 0:
                    agent.update_target()

        rewards.append(episode_rewards)

        if episode % print_interval == 0 and episode != 0:
            print("step: {} / episode: {} / reward: {:.2f} / loss: {:.4f} / epsilon: {:.3f}".format
                  (step, episode, np.mean(rewards), np.mean(losses), agent.epsilon))
            agent.Write_Summray(np.mean(rewards), np.mean(losses), episode)
            rewards = []
            losses = []

        if episode % save_interval == 0 and episode != 0:
            agent.save_model()
            print("Save Model {}".format(episode))

    env.close()