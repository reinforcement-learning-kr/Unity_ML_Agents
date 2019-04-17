import numpy as np
import random
import datetime
import time
import tensorflow as tf
import tensorflow.layers as layer
from collections import deque
from mlagents.envs import UnityEnvironment
import logging

########################################
state_size = [80, 80, 3]
action_size = 4

load_model = False
train_mode = True

batch_size = 32
hidden_layer_size = 512
mem_maxlen = 50000
discount_factor = 0.9
learning_rate = 0.00007

run_episode = 100000
test_episode = 500

start_train_episode = 500

target_update_step = 10000
print_interval = 100
save_interval = 5000

epsilon_min = 0.01

date_time = str(datetime.date.today()) + '_' + \
            str(datetime.datetime.now().hour) + '_' + \
            str(datetime.datetime.now().minute) + '_' + \
            str(datetime.datetime.now().second)

env_worker_id = 0

env_name = "../envs/Sokoban_win_new_reward_action_4/Sokoban"     # Window OS
# env_name = "../envs/Sokoban_linux_new_reward_action_4/Sokoban" # Linux OS

save_path = "saved_models/" + date_time + "_ddqn"
load_path = "./saved_models/2019-02-18_17_39_4_ddqn/model/"

sokoban_reset_parameters = \
[
    {"gridSize": 4, "numGoals": 3, "numBoxes": 1, "numObstacles": 1},  # Level 0
    {"gridSize": 6, "numGoals": 4, "numBoxes": 1, "numObstacles": 1},  # Level 1
    {"gridSize": 8, "numGoals": 6, "numBoxes": 1, "numObstacles": 1},  # Level 2
    {"gridSize": 10, "numGoals": 7, "numBoxes": 2, "numObstacles": 1},  # Level 3
    {"gridSize": 10, "numGoals": 6, "numBoxes": 3, "numObstacles": 1},  # Level 4
    {"gridSize": 10, "numGoals": 5, "numBoxes": 4, "numObstacles": 1},  # Level 5
]

curriculum_config = {
    'game_level': [0, 1, 2, 3, 4, 5],  # the index of sokoban_reset_parameters
    'thresholds': [0.7, 0.7, 0.7, 0.7, 0.7, None],  # thresholds about the success rate per each game_level
    'start_epsilon': [1.0, 0.7, 0.7, 0.7, 0.7, 0.7],
    'epsilon_decay': [0.00005, 0.00005, 0.00005, 0.00005, 0.00005, 0.00005],
    'min_lesson_length': 500,  # minimum length of episodes per each game_level
}

###########################################
class DDQN_Model():

    def __init__(self,state_size,action_size,hidden_layer_size,learning_rate=0.00025,model_name="model",global_step=None):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_layer_size = hidden_layer_size
        self.input = tf.placeholder(shape=[None, self.state_size[0],self.state_size[1],self.state_size[2]],dtype=tf.float32)
        self.input_normalize = (self.input - (255.0 / 2)) / (255.0 / 2)
        with tf.variable_scope(name_or_scope=model_name):
            self.conv1 = layer.conv2d(inputs=self.input_normalize,filters=32,activation=tf.nn.relu,kernel_size=[8,8],strides=[4,4],padding="SAME")
            self.conv2 = layer.conv2d(inputs=self.conv1,filters=64,activation=tf.nn.relu,kernel_size=[4,4],strides=[2,2],padding="SAME")
            self.conv3 = layer.conv2d(inputs=self.conv2,filters=64,activation=tf.nn.relu,kernel_size=[3,3],strides=[1,1],padding="SAME")

            self.flat = layer.flatten(self.conv3)

            self.L1 = layer.dense(self.flat,self.hidden_layer_size,activation=tf.nn.relu)

            self.A1 = layer.dense(self.L1,self.hidden_layer_size,activation=tf.nn.relu)
            self.Adventage = layer.dense(self.A1,self.action_size,activation=None)

            self.V1 = layer.dense(self.L1,self.hidden_layer_size,activation=tf.nn.relu)
            self.Value = layer.dense(self.V1,1,activation=None)

            self.Q_Out = self.Value + tf.subtract(self.Adventage,tf.reduce_mean(self.Adventage,axis=1,keep_dims=True))
        self.predict = tf.arg_max(self.Q_Out,1)

        self.target_Q = tf.placeholder(shape=[None,self.action_size],dtype=tf.float32)

        self.loss = tf.losses.mean_squared_error(self.target_Q,self.Q_Out)
        self.UpdateModel = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss, global_step=global_step)
        self.trainable_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, model_name)



class DDQNAgent():
    def __init__(self, state_size, action_size, mem_maxlen, save_path, load_path, learning_rate=0.00025,
                 load_model=True, batch_size=32, run_episode=10000, hidden_layer_size=512, epsilon=1.0, epsilon_decay=0.00005, epsilon_min=0.1, discount_factor=0.99):
        self.state_size = state_size
        self.action_size = action_size

        self.global_step = tf.Variable(0, trainable=False)
        self.model = DDQN_Model(state_size, action_size, hidden_layer_size=hidden_layer_size, model_name="Q", learning_rate=learning_rate, global_step=self.global_step)
        self.target_model = DDQN_Model(state_size, action_size, hidden_layer_size=hidden_layer_size, model_name="target")

        self.memory = deque(maxlen=mem_maxlen)

        self.batch_size = batch_size
        self.run_episode = run_episode

        # 세션 초기화
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tf_config)
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.discount_factor = discount_factor

        self.load_model = load_model
        self.Saver = tf.train.Saver(max_to_keep=5)
        self.Saver_level_up = tf.train.Saver(max_to_keep=len(curriculum_config['game_level']))
        self.save_path = save_path
        self.load_path = load_path
        self.Summary, self.Merge = self.make_Summary()

        self.update_target()

        print('save path : ', self.save_path)

        if self.load_model:
            ckpt = tf.train.get_checkpoint_state(self.load_path)
            if ckpt is not None:
                print('Load File : ', ckpt.model_checkpoint_path)
                self.Saver.restore(self.sess, ckpt.model_checkpoint_path)

    def get_action(self, state, train_mode=True):
        if train_mode and self.epsilon > np.random.rand():
            return np.random.randint(0, self.action_size)
        else:
            predict = self.sess.run(self.model.predict, feed_dict={self.model.input: state})
            return np.asscalar(predict)

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state[0], action, reward, next_state[0], done))

    def save_model(self, current_level=None, is_level_up=False):
        if not is_level_up:
            self.Saver.save(self.sess, self.save_path + "/model/model.ckpt", global_step=self.global_step)
        else:
            self.Saver_level_up.save(
                self.sess, self.save_path + '/Level_Up_{}'.format(current_level) + "/model.bytes",
                global_step=self.global_step)

    def train_model(self, done):
        if done:
            if self.epsilon > self.epsilon_min:
                # self.epsilon -= 1 / (self.run_episode - start_train_episode)
                self.epsilon -= self.epsilon_decay
            elif self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

        mini_batch = random.sample(self.memory, self.batch_size)
        states = np.zeros(shape=[self.batch_size, self.state_size[0], self.state_size[1], self.state_size[2]])
        next_states = np.zeros(shape=[self.batch_size, self.state_size[0], self.state_size[1], self.state_size[2]])
        actions = []
        rewards = []
        dones = []
        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        target = self.sess.run(self.model.Q_Out, feed_dict={self.model.input: states})
        action_t = self.sess.run(self.model.predict, feed_dict={self.model.input: next_states})
        n_target = self.sess.run(self.target_model.Q_Out, feed_dict={self.target_model.input: next_states})

        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * n_target[i][action_t[i]]

        _, loss = self.sess.run([self.model.UpdateModel, self.model.loss],
                                feed_dict={self.model.input: states, self.model.target_Q: target})
        return loss

    def update_target(self):
        for i in range(len(self.model.trainable_var)):
            self.sess.run(self.target_model.trainable_var[i].assign(self.model.trainable_var[i]))

    def make_Summary(self):
        with tf.name_scope(name='game'):
            self.summary_loss = tf.placeholder(dtype=tf.float32)
            self.summary_reward = tf.placeholder(dtype=tf.float32)
            self.summary_game_level = tf.placeholder(dtype=tf.int32)
            self.summary_success = tf.placeholder(dtype=tf.float32)
            tf.summary.scalar("loss", self.summary_loss)
            tf.summary.scalar("reward", self.summary_reward)
            tf.summary.scalar("success", self.summary_success)
            tf.summary.scalar("game_level", self.summary_game_level)
        with tf.name_scope(name='parameters'):
            self.summary_batch_size = tf.placeholder(dtype=tf.int32)
            self.summary_mem_maxlen = tf.placeholder(dtype=tf.float32)
            self.summary_discount_factor = tf.placeholder(dtype=tf.float32)
            self.summary_learning_rate = tf.placeholder(dtype=tf.float32)
            tf.summary.scalar("batch_size", self.summary_batch_size)
            tf.summary.scalar("mem_maxlen", self.summary_mem_maxlen)
            tf.summary.scalar("discount_factor", self.summary_discount_factor)
            tf.summary.scalar("learning_rate", self.summary_learning_rate)
        return tf.summary.FileWriter(logdir=save_path, graph=self.sess.graph), tf.summary.merge_all()

    def Write_Summray(self, reward, loss, success, game_level, episode):
        self.Summary.add_summary(
            self.sess.run(self.Merge, feed_dict={self.summary_loss: loss,
                                                 self.summary_reward: reward,
                                                 self.summary_success: success,
                                                 self.summary_game_level: game_level,
                                                 self.summary_batch_size: batch_size,
                                                 self.summary_discount_factor: discount_factor,
                                                 self.summary_learning_rate: learning_rate,
                                                 self.summary_mem_maxlen: mem_maxlen}), episode)
        self.Summary.flush()

    def level_up(self, start_epsilon, epsilon_decay):
        self.epsilon = start_epsilon
        self.epsilon_decay = epsilon_decay
        self.memory.clear()


if __name__ == '__main__':

    env = UnityEnvironment(file_name=env_name,
                           worker_id=env_worker_id)
    default_brain = env.brain_names[0]
    brain = env.brains[default_brain]
    # logging.getLogger().setLevel(logging.ERROR)

    # 커리큘럼 초기화
    curriculum_num = 0
    smoothing_value = 0
    min_lesson_length = curriculum_config['min_lesson_length']
    game_level = curriculum_config['game_level'][curriculum_num]
    threshold = curriculum_config['thresholds'][curriculum_num]
    start_epsilon = curriculum_config['start_epsilon'][curriculum_num]
    epsilon_decay = curriculum_config['epsilon_decay'][curriculum_num]

    agent = DDQNAgent(state_size,
                     action_size,
                     mem_maxlen,
                     save_path,
                     load_path,
                     learning_rate,
                     load_model,
                     batch_size,
                     run_episode,
                     hidden_layer_size=hidden_layer_size,
                     epsilon=start_epsilon,
                     epsilon_decay=epsilon_decay,
                     epsilon_min=epsilon_min,
                     discount_factor=discount_factor)

    # Env reset
    env_info = env.reset(train_mode=train_mode, config=sokoban_reset_parameters[game_level])[default_brain]

    step = 0
    start_episode = 0
    rewards = deque(maxlen=print_interval)
    losses = deque(maxlen=print_interval)
    successes = deque(maxlen=print_interval)
    start_time = datetime.datetime.now()

    # Learning ...
    for episode in range(run_episode + test_episode):
        if episode > run_episode:
            train_mode = False
            env_info = env.reset(train_mode=train_mode)[default_brain]

        state = np.uint8(255 * env_info.visual_observations[0])
        episode_rewards = 0
        done = False
        success = 0

        while not done:
            step += 1

            action = agent.get_action(state, train_mode)
            env_info = env.step(action)[default_brain]

            next_state = np.uint8(255 * env_info.visual_observations[0])
            reward = env_info.rewards[0]
            episode_rewards += reward
            done = env_info.local_done[0]

            if done and reward == 2:
                success = 1

            if train_mode:
                agent.append_sample(state, action, reward, next_state, done)
            else:
                time.sleep(0.01)

            state = next_state

            if episode > start_train_episode and train_mode and len(agent.memory) >= batch_size:
                loss = agent.train_model(done)
                losses.append(loss)

                if step % (target_update_step) == 0:
                    agent.update_target()

        rewards.append(episode_rewards)
        successes.append(success)

        if episode % print_interval == 0 and episode != 0:
            print("game_level: {} / step: {} / episode: {} / reward: {:.2f} / success: {:.3f} / loss: {:.4f} / epsilon: {:.3f} / memory_len:{}".format
                  (game_level, step, episode, np.mean(rewards), np.mean(successes), np.mean(losses), agent.epsilon, len(agent.memory)))
            agent.Write_Summray(np.mean(rewards), np.mean(losses), np.mean(successes), game_level, episode)

        if episode % save_interval == 0 and episode != 0:
            agent.save_model()
            print("Save Model {}".format(episode))

        # Check the Level Up
        if threshold is not None and np.mean(successes) >= threshold and (episode - start_episode) >= min_lesson_length:
            start_episode = episode
            end_time = datetime.datetime.now()
            print('[LEVEL UP] Lv.{} | Time elapse : {} | success : {}'.format(game_level, str(end_time - start_time),
                                                                              np.mean(successes)))

            agent.save_model(current_level=game_level, is_level_up=True)
            print("Save Model {}".format(episode))

            curriculum_num += 1

            game_level = curriculum_config['game_level'][curriculum_num]
            threshold = curriculum_config['thresholds'][curriculum_num]
            start_epsilon = curriculum_config['start_epsilon'][curriculum_num]
            epsilon_decay = curriculum_config['epsilon_decay'][curriculum_num]
            agent.level_up(start_epsilon, epsilon_decay)

            # 환경 리셋
            env_info = env.reset(train_mode=train_mode, config=sokoban_reset_parameters[game_level])[default_brain]

            rewards.clear()
            successes.clear()

        if agent.epsilon == agent.epsilon_min:
            print('Reset the agent epsilon!')
            agent.epsilon = 0.3

    agent.save_model()
    print("Complete Sokoban!")
    agent.sess.close()
    env.close()

