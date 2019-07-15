import numpy as np
import random
import datetime
import time
import tensorflow as tf
from collections import deque
from mlagents.envs import UnityEnvironment

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

date_time = str(datetime.date.today()) + '_' + \
            str(datetime.datetime.now().hour) + '_' + \
            str(datetime.datetime.now().minute) + '_' + \
            str(datetime.datetime.now().second)

game = "Pong"
env_name = "../env/" + game + "/Windows/" + game

save_path = "../saved_models/" + game + "/" + date_time + "_DQN"
load_path = "../saved_models/" + game + "/2019-02-20_16_27_5_DQN/model/model.ckpt"

class Model():
    def __init__(self, model_name):
        self.input = tf.placeholder(shape=[None, state_size], dtype=tf.float32)

        with tf.variable_scope(name_or_scope=model_name):
            self.fc1 = tf.layers.dense(self.input,512,activation=tf.nn.relu)
            self.fc2 = tf.layers.dense(self.fc1,512,activation=tf.nn.relu)
            self.fc3 = tf.layers.dense(self.fc2,512,activation=tf.nn.relu)
            self.Q_Out = tf.layers.dense(self.fc3,action_size,activation=None)
        self.predict = tf.argmax(self.Q_Out, 1)

        self.target_Q = tf.placeholder(shape=[None, action_size], dtype=tf.float32)

        self.loss = tf.losses.mean_squared_error(self.target_Q, self.Q_Out)
        self.UpdateModel = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        self.trainable_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, model_name)

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

        self.Saver = tf.train.Saver
        self.Summary, self.Merge = self.Make_Summary()

        self.update_target()

        if load_model == True:
            self.Saver.restore(self.sess, load_path)

    def get_action(self, state1, state2, train_mode=True):
        if train_mode == True and self.epsilon > np.random.rand():
            random_action1 = np.random.randint(0, action_size)
            random_action2 = np.random.randint(0, action_size)
            return random_action1, random_action2
        else:
            predict1 = self.sess.run(self.model1.predict, 
                                     feed_dict={self.model1.input: [state1]})
            predict2 = self.sess.run(self.model2.predict, 
                                     feed_dict={self.model2.input: [state2]})
            return np.asscalar(predict1), np.asscalar(predict2)

    def append_sample(self, data1, data2):
        self.memory1.append((data1[0], data1[1], data1[2], data1[3], data1[4]))
        self.memory2.append((data2[0], data2[1], data2[2], data2[3], data2[4]))

    def save_model(self):
        self.Saver.save(self.sess, save_path + "/model/model.ckpt")

    def train_model(self, model, target_model, memory, done):
        if done:
            if self.epsilon > epsilon_min:
                self.epsilon -= 1 / (run_episode - start_train_episode)

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

        target = self.sess.run(model1.Q_Out, feed_dict={model.input: states})
        target_val = self.sess.run(target_model.Q_Out, 
                                    feed_dict={target_model.input: next_states})

        for i in range(batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + discount_factor * np.amax(target_val[i])

        _, loss = self.sess.run([model.UpdateModel, model.loss],
                                feed_dict={model.input: states, 
                                           model.target_Q: target})

        return loss

    def update_target(self):
        for i in range(len(self.model1.trainable_var)):
            self.sess.run(self.target_model1.trainable_var[i].assign(self.model1.trainable_var[i]))

        for i in range(len(self.model2.trainable_var)):
            self.sess.run(self.target_model2.trainable_var[i].assign(self.model2.trainable_var[i]))

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

if __name__ == '__main__':

    env = UnityEnvironment(file_name=env_name)

    default_brain = env.brain_names[0]
    brain = env.brains[default_brain]

    agent = DQNAgent()

    step = 0

    rewards1 = []
    losses1 = []
    rewards2 = []
    losses2 = []

    # Set the brain for each players
    brain_name1 = env.brain_names[0]
    brain_name2 = env.brain_names[1]

    brain1 = env.brains[brain_name1]
    brain2 = env.brains[brain_name2]

    # Reset Environment  
    env_info = env.reset(train_mode=train_mode)

    for episode in range(run_episode + test_episode):
        if episode == run_episode:
            train_mode = False
            env_info = env.reset(train_mode=train_mode)
        
        done = False

        state1 = env_info[brain_name1].vector_observations[0]
        episode_rewards1 = 0
        done1 = False

        state2 = env_info[brain_name2].vector_observations[0]
        episode_rewards2 = 0
        done2 = False

        while not done:
            step += 1

            action1, action2 = agent.get_action(state1, state2, train_mode)
            env_info = env.step(vector_action = {brain_name1: [action1], brain_name2: [action2]})

            next_state1 = env_info[brain_name1].vector_observations[0]
            reward1 = env_info[brain_name1].rewards[0]
            episode_rewards1 += reward1
            done1 = env_info[brain_name1].local_done[0]

            next_state2 = env_info[brain_name2].vector_observations[0]
            reward2 = env_info[brain_name2].rewards[0]
            episode_rewards2 += reward2
            done2 = env_info[brain_name2].local_done[0]

            done = done1 or done2

            if train_mode:
                data1 = [state1, action1, reward1, next_state1, done1]
                data2 = [state2, action2, reward2, next_state2, done2]
                
                agent.append_sample(data1, data2)
            else:
                time.sleep(0.02)

            state1 = next_state1
            state2 = next_state2

            if episode > start_train_episode and train_mode:
                loss1 = agent.train_model(agent.model1, agent.target_model1, agent.memory1, done)
                loss2 = agent.train_model(agent.model2, agent.target_model2, agent.memory2, done)
                losses1.append(loss1)
                losses2.append(loss2)

                if step % (target_update_step) == 0:
                    agent.update_target()

        rewards1.append(episode_rewards1)
        rewards2.append(episode_rewards2)

        if episode % print_interval == 0 and episode != 0:
            print("step: {} / episode: {} / reward1: {:.2f} / loss1: {:.4f} / reward2: {:.2f} \
                  / loss2: {:.4f} / epsilon: {:.3f}".format(step, episode, np.mean(rewards1), 
                  np.mean(losses1), np.mean(rewards2), np.mean(losses2), agent.epsilon))
            
            agent.Write_Summray(np.mean(rewards1), np.mean(losses1), 
                                np.mean(rewards2), np.mean(losses2), episode)
            rewards1 = []
            losses1 = []
            rewards2 = []
            losses2 = []

        if episode % save_interval == 0 and episode != 0:
            agent.save_model()
            print("Save Model {}".format(episode))

    env.close()