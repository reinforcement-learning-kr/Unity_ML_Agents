#라이브러리 불러오기
import numpy as np
import random
import datetime
import time
import tensorflow as tf
from collections import deque
from mlagents.envs import UnityEnvironment

# 파라미터 설정
state_size = [80, 80, 3]
action_size = 4

load_model = False
train_mode = True

batch_size = 32
hidden_layer_size = 512
mem_maxlen = 50000
discount_factor = 0.9
learning_rate = 0.00005

run_episode = 75000
test_episode = 500

start_train_episode = 500

target_update_step = 10000
print_interval = 100
save_interval = 5000

start_epsilon = 1.0
epsilon_min = 0.1

date_time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")

game = "Sokoban_Curriculum"
env_name = "../env/" + game + "/Windows/" + game

save_path = "../saved_models/" + game + "/" + date_time + "_dddqn"
load_path = "../saved_models/" + game + "/20190714-18-40-51_dddqn/model/model"

# 소코반 커리큘럼 환경의 레벨 별 리셋 파라미터 설정
sokoban_reset_parameters = \
[
    {"gridSize": 4, "numGoals": 3, "numBoxes": 1, "numObstacles": 1},  # Level 0
    {"gridSize": 6, "numGoals": 3, "numBoxes": 1, "numObstacles": 1},  # Level 1
    {"gridSize": 6, "numGoals": 3, "numBoxes": 1, "numObstacles": 2},  # Level 2
    {"gridSize": 6, "numGoals": 2, "numBoxes": 2, "numObstacles": 2}   # Level 3
    #{"gridSize": 6, "numGoals": 3, "numBoxes": 3, "numObstacles": 2},  # Level 4
]

# 커리큘럼 설정
# curriculum_config = {
#     'game_level': [0, 1, 2, 3, 4],  # 게임의 레벨
#     'thresholds': [0.7, 0.7, 0.7, 0.7, None],  # 각 게임 레벨 별 클리어 성공률
#     'start_epsilon': [1.0, 0.7, 0.7, 0.7, 0.7], # 시작 앱실론 값
#     'epsilon_decay': [0.00005, 0.00005, 0.00005, 0.00005, 0.000025],
#     'min_lesson_length': 500,  # 각 게임 레벨 별 최소 수행 해야할 에피소드 수
# }

curriculum_config = {
    'game_level': [0, 1, 2, 3],  # 게임의 레벨
    'thresholds': [0.7, 0.7, 0.7, None],  # 각 게임 레벨 별 클리어 성공률
    'start_epsilon': [1.0, 0.7, 0.7, 0.7], # 시작 앱실론 값
    'epsilon_decay': [0.00005, 0.00005, 0.00005, 0.000025],
    'min_lesson_length': 500,  # 각 게임 레벨 별 최소 수행 해야할 에피소드 수
}

# DDDQN 네트워크
class DDDQN_Model():
    def __init__(self, model_name):
        self.input = tf.placeholder(shape=[None, state_size[0],
                                           state_size[1],
                                           state_size[2]],
                                    dtype=tf.float32)

        self.input_normalize = (self.input - (255.0 / 2)) / (255.0 / 2)

        with tf.variable_scope(name_or_scope=model_name):
            self.conv1 = tf.layers.conv2d(inputs=self.input_normalize, filters=32, 
                                          activation=tf.nn.relu, kernel_size=[8, 8], 
                                          strides=[4, 4], padding="SAME")
            self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=64, activation=tf.nn.relu, 
                                          kernel_size=[4, 4], strides=[2, 2], padding="SAME")
            self.conv3 = tf.layers.conv2d(inputs=self.conv2, filters=64, activation=tf.nn.relu, 
                                          kernel_size=[3, 3], strides=[1, 1], padding="SAME")

            self.flat = tf.layers.flatten(self.conv3)

            # Dueling DQN 부분
            self.L1 = tf.layers.dense(self.flat, hidden_layer_size, activation=tf.nn.relu)
            self.A1 = tf.layers.dense(self.L1, hidden_layer_size, activation=tf.nn.relu)
            self.Advantage = tf.layers.dense(self.A1, action_size, activation=None)

            # Critic
            self.V1 = tf.layers.dense(self.L1, hidden_layer_size, activation=tf.nn.relu)
            self.Value = tf.layers.dense(self.V1, 1, activation=None)

            self.Q_Out = self.Value + (self.Advantage - tf.reduce_mean(self.Advantage,
                                                                       axis=1, keep_dims=True))

        self.predict = tf.arg_max(self.Q_Out, 1)

        self.target_Q = tf.placeholder(shape=[None, action_size], dtype=tf.float32)

        self.loss = tf.losses.mean_squared_error(self.target_Q, self.Q_Out)
        self.UpdateModel = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
        self.trainable_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, model_name)

# DDDQN 에이전트
class DDDQNAgent():
    def __init__(self):
        # 타겟 네트워크와 일반 네트워크 생성
        self.model = DDDQN_Model(model_name="Q")
        self.target_model = DDDQN_Model(model_name="target")
        self.memory = deque(maxlen=mem_maxlen)

        # 세션 초기화
        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        # 하이퍼 파라미터
        self.epsilon = start_epsilon

        # 모델 저장 및 로드
        self.Saver = tf.train.Saver(max_to_keep=len(curriculum_config['game_level']))
        self.Summary, self.Merge = self.Make_Summary()

        self.update_target()

        if load_model == True:
            self.Saver.restore(self.sess, load_path)

    # 엡실론 값에 따라 행동을 선택하는 함수
    def get_action(self, state, train_mode=True):
        if train_mode and self.epsilon > np.random.rand():
            return np.random.randint(0, action_size)
        else:
            predict = self.sess.run(self.model.predict, feed_dict={self.model.input: state})
            return np.asscalar(predict)

    # 학습 데이터를 리플레이 메모리에 저장하는 함수
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state[0], action, reward, next_state[0], done))

    # 모델 저장 함수
    def save_model(self, current_level):
        self.Saver.save(self.sess, save_path + '/Level_Up_{}'.format(current_level) + "/model")

    # 학습을 위한 함수
    def train_model(self, done):
        if done:
            if self.epsilon > epsilon_min:
                self.epsilon -= epsilon_decay

        mini_batch = random.sample(self.memory, batch_size)

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

        # Double DQN 부분
        target = self.sess.run(self.model.Q_Out, feed_dict={self.model.input: states})
        action_t = self.sess.run(self.model.predict, feed_dict={self.model.input: next_states})
        target_val = self.sess.run(self.target_model.Q_Out,
                                   feed_dict={self.target_model.input: next_states})

        for i in range(batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + discount_factor * target_val[i][action_t[i]]

        _, loss = self.sess.run([self.model.UpdateModel, self.model.loss],
                                feed_dict={self.model.input: states, self.model.target_Q: target})
        return loss

    # 타겟 네트워크와 일반 네트워크의 파라미터를 동기화 시켜주는 함수
    def update_target(self):
        for i in range(len(self.model.trainable_var)):
            self.sess.run(self.target_model.trainable_var[i].assign(self.model.trainable_var[i]))

    # 텐서보드에서 학습 과정의 값을 보기위해 설정하는 함수
    def Make_Summary(self):
        with tf.name_scope(name='game'):
            self.summary_loss = tf.placeholder(dtype=tf.float32)
            self.summary_reward = tf.placeholder(dtype=tf.float32)
            self.summary_game_level = tf.placeholder(dtype=tf.int32)
            self.summary_success = tf.placeholder(dtype=tf.float32)
            tf.summary.scalar("loss", self.summary_loss)
            tf.summary.scalar("reward", self.summary_reward)
            tf.summary.scalar("success", self.summary_success)
            tf.summary.scalar("game_level", self.summary_game_level)

        return tf.summary.FileWriter(logdir=save_path, graph=self.sess.graph), tf.summary.merge_all()

    # 텐서보드에 입력을 위한 함수
    def Write_Summray(self, reward, loss, success, game_level, episode):
        self.Summary.add_summary(
            self.sess.run(self.Merge, feed_dict={self.summary_loss: loss,
                                                 self.summary_reward: reward,
                                                 self.summary_success: success,
                                                 self.summary_game_level: game_level}), episode)

    # 커리큘럼 레벨이 높아졌을 때 변하는 변수들을 관리하는 함수
    def level_up(self):
        self.epsilon = start_epsilon
        self.epsilon_decay = epsilon_decay
        self.memory.clear()

if __name__ == '__main__':
    # 환경 생성
    env = UnityEnvironment(file_name=env_name)

    default_brain = env.brain_names[0]
    brain = env.brains[default_brain]

    # 커리큘럼 초기화
    curriculum_num = 0
    min_lesson_length = curriculum_config['min_lesson_length']
    game_level = curriculum_config['game_level'][curriculum_num]
    threshold = curriculum_config['thresholds'][curriculum_num]
    start_epsilon = curriculum_config['start_epsilon'][curriculum_num]
    epsilon_decay = curriculum_config['epsilon_decay'][curriculum_num]

    # 환경 및 변수 초기화
    env_info = env.reset(train_mode=train_mode,
                         config=sokoban_reset_parameters[game_level])[default_brain]

    # DDDQN 에이전트 생성
    agent = DDDQNAgent()

    step = 0
    start_level_episode = 0
    rewards = deque(maxlen=print_interval)
    losses = deque(maxlen=print_interval)
    successes = deque(maxlen=print_interval)
    
    start_time = datetime.datetime.now()

    # 학습 과정
    for episode in range(run_episode + test_episode):
        if episode > run_episode:
            train_mode = False
            env_info = env.reset(train_mode=train_mode)[default_brain]

        state = np.uint8(255 * np.array(env_info.visual_observations[0]))
        episode_rewards = 0
        done = False
        success = 0

        while not done:
            step += 1
            action = agent.get_action(state, train_mode)
            env_info = env.step(action)[default_brain]
            next_state = np.uint8(255 * np.array(env_info.visual_observations[0]))

            reward = env_info.rewards[0]
            episode_rewards += reward
            done = env_info.local_done[0]

            # 에피소드가 끝났을 때, 보상을 2 받았으면 성공
            if done and reward == 2:
                success = 1

            if train_mode:
                agent.append_sample(state, action, reward, next_state, done)
            else:
                time.sleep(0.01)

            state = next_state

            # 학습 수행
            if (episode - start_level_episode) > start_train_episode and train_mode:
                loss = agent.train_model(done)
                losses.append(loss)

                if step % target_update_step == 0:
                    agent.update_target()

        rewards.append(episode_rewards)
        successes.append(success)

        # 학습 결과 출력
        if episode % print_interval == 0 and episode != 0:
            print("game_level: {} / step: {} / episode: {} / reward: {:.2f} / success: {:.3f} / "
                  "loss: {:.4f} / epsilon: {:.3f} / memory_len:{}".format
                  (game_level, step, episode, np.mean(rewards), np.mean(successes),
                   np.mean(losses), agent.epsilon, len(agent.memory)))
            agent.Write_Summray(np.mean(rewards), np.mean(losses),
                                np.mean(successes), game_level, episode)

        # 커리큘럼 레벨을 높일지 확인
        if threshold is not None and np.mean(successes) >= threshold and \
                (episode - start_level_episode) >= min_lesson_length:
            start_level_episode = episode
            end_time = datetime.datetime.now()
            print('[LEVEL UP] Lv.{} | Time elapse : {} | success : {}'.format(
                game_level, str(end_time - start_time), np.mean(successes)))

            agent.save_model(current_level=game_level)
            print("Save Model {}".format(episode))

            curriculum_num += 1
            game_level = curriculum_config['game_level'][curriculum_num]
            threshold = curriculum_config['thresholds'][curriculum_num]
            start_epsilon = curriculum_config['start_epsilon'][curriculum_num]
            epsilon_decay = curriculum_config['epsilon_decay'][curriculum_num]
            agent.level_up()

            # 환경 리셋
            env_info = env.reset(train_mode=train_mode,
                                 config=sokoban_reset_parameters[game_level])[default_brain]
            rewards.clear()
            successes.clear()

    agent.save_model(current_level=game_level)
    env.close()

