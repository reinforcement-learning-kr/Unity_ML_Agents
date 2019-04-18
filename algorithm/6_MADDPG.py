import numpy as np
import tensorflow as tf
import random
import datetime
import time
from collections import deque
from mlagents.envs import UnityEnvironment
import numpy as np
import tensorflow as tf


# Training Parameters ==================================================================================================
load_model = False
train_mode = True

batch_size = 128
mem_maxlen = 50000
discount_factor = 0.99
learning_rate = 0.0025

run_episode = 10000
start_train_episode = 500
target_update_step = 5000

print_interval = 100
save_interval = 1000

env_name = "C:/Users/asdfw/Desktop/OneLegAgent_MultiBrain/One_Leg_Agent"
save_path = "./Save/"
load_path = "./Save/"

state_size = 30
action_size = 1
numAgent = 3

###########################################

class Critic(object):
    def __init__(self, input, action_input, other_action, model_name="Critic", agent_num=3, reuse=False):

        self.input = input
        self.action_input = action_input
        self.other_actions = other_action

        with tf.variable_scope(name_or_scope=model_name, reuse=reuse):
            self.mlp1 = tf.layers.dense(inputs=self.input, units=512, activation = tf.nn.leaky_relu)
            self.concat_action = tf.concat([self.action_input, self.other_actions], axis=1)
            self.concat = tf.concat([self.mlp1, self.concat_action], axis=1)
            self.mlp2 = tf.layers.dense(inputs=self.concat, units=512, activation = tf.nn.leaky_relu)
            self.mlp3 = tf.layers.dense(inputs=self.mlp2, units=512, activation = tf.nn.leaky_relu)
            self.mlp4 = tf.layers.dense(inputs=self.mlp3, units=512, activation = tf.nn.leaky_relu)

            output_weight_mu = mu_variable([512, 1])
            output_weight_sig = sigma_variable([512, 1])
            output_bias_mu = mu_variable([1])
            output_bias_sig = sigma_variable([1])

            self.Q_Out = noisy_dense(self.mlp4, [512, 1],
                                      output_weight_mu, output_weight_sig,
                                      output_bias_mu, output_bias_sig)

        self.q_predict = self.Q_Out
        self.critic_optimizer = tf.train.AdamOptimizer(learning_rate)

class Actor(object):
    def __init__(self, input, model_name="Actor"):

        self.input = input

        with tf.variable_scope(name_or_scope=model_name):
            self.mlp1 = tf.layers.dense(inputs=self.input, units=512, activation = tf.nn.leaky_relu)
            self.mlp2 = tf.layers.dense(inputs=self.mlp1, units=512, activation = tf.nn.leaky_relu)
            self.mlp3 = tf.layers.dense(inputs=self.mlp2, units=512, activation = tf.nn.leaky_relu)
            self.mlp4 = tf.layers.dense(inputs=self.mlp3, units=512, activation = tf.nn.leaky_relu)

            output_weight_mu = mu_variable([512, action_size])
            output_weight_sig = sigma_variable([512, action_size])
            output_bias_mu = mu_variable([action_size])
            output_bias_sig = sigma_variable([action_size])

            self.noise_out = noisy_dense(self.mlp4, [512, action_size],
                                        output_weight_mu, output_weight_sig,
                                        output_bias_mu, output_bias_sig)
            self.pi_predict = tf.nn.tanh(self.noise_out)

        self.actor_optimizer = tf.train.AdamOptimizer(learning_rate)

class MADDPGAgent(object):
    def __init__(self, idx):

        # Experience Buffer ===================
        self.memory = deque(maxlen=mem_maxlen)
        self.batch_size = batch_size
        # =====================================

        # Placeholer =============================================================================
        self.input = tf.placeholder(shape=[None, state_size], dtype=tf.float32)
        self.action_input = tf.placeholder(shape=[None, action_size], dtype=tf.float32)
        self.other_actions = tf.placeholder(shape=[None, action_size * (numAgent-1)], dtype=tf.float32)
        self.target_Q = tf.placeholder(shape=[None,1],dtype=tf.float32)
        self.reward = tf.placeholder(shape=[None,1], dtype=tf.float32)
        # ========================================================================================
        self.actor = Actor(self.input, "Actor_" + idx)
        self.critic = Critic(self.input, self.action_input, self.other_actions, "Critic_" + idx, reuse=False)

        self.onActor_vars = [i for i in tf.trainable_variables() if "Actor_" + idx in i.name]
        self.onCritic_vars = [i for i in tf.trainable_variables() if "Critic_" + idx in i.name]

        self.target_actor = Actor(self.input, "ActorTarget_" + idx)
        self.target_critic = Critic(self.input, self.action_input, self.other_actions, "CriticTarget" + idx, reuse=False)

        self.targetActor_vars = [i for i in tf.trainable_variables() if "ActorTarget_" + idx in i.name]
        self.targetCritic_vars = [i for i in tf.trainable_variables() if "CriticTarget_" + idx in i.name]

        action_Grad = tf.clip_by_value(tf.gradients(self.critic.q_predict, self.action_input), -0.1, 0.1)
        self.policy_Grads = tf.gradients(ys=self.actor.pi_predict, xs=self.onActor_vars, grad_ys=action_Grad)
        for idx, grads in enumerate(self.policy_Grads):
            self.policy_Grads[idx] = -grads / batch_size
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_actor_op = tf.train.AdamOptimizer(learning_rate=learning_rate).apply_gradients(zip(self.policy_Grads, self.onActor_vars))

        self.loss = tf.losses.mean_squared_error(self.target_Q, self.critic.q_predict)
        self.train_critic_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

        # Target Update
        tau = 0.1
        self.target_update_actor = [tf.assign(target, (tau) * online + (1 - tau) * target) for online, target in
                                    zip(self.onActor_vars, self.targetActor_vars)]
        self.target_update_critic = [tf.assign(target, (tau) * online + (1 - tau) * target) for online, target in
                                     zip(self.onCritic_vars, self.targetCritic_vars)]
        self.target_init_actor = [tf.assign(target, online) for online, target in
                                  zip(self.onActor_vars, self.targetActor_vars)]
        self.target_init_critic = [tf.assign(target, online) for online, target in
                                   zip(self.onCritic_vars, self.targetCritic_vars)]

    def append_sample(self, data):
        self.memory.append((data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8]))

    def train_models(self, t_actor1, t_actor2, sess):

        mini_batch = random.sample(self.memory, batch_size)

        states = []
        actions = []
        other_actions1 = []
        other_actions2 = []
        rewards = []
        next_states = []
        next_other_states1 = []
        next_other_states2 = []
        dones = []

        for i in range(batch_size):
            states.append(mini_batch[i][0])
            actions.append(mini_batch[i][1])
            other_actions1.append(mini_batch[i][2])
            other_actions2.append(mini_batch[i][3])
            rewards.append(mini_batch[i][4])
            next_states.append(mini_batch[i][5])
            next_other_states1.append(mini_batch[i][6])
            next_other_states2.append(mini_batch[i][7])
            dones.append(mini_batch[i][8])

        states = np.reshape(states, newshape=[-1, state_size])
        actions = np.reshape(actions, newshape=[-1, action_size])
        next_states = np.reshape(next_states, newshape=[-1, state_size])
        next_other_states1 = np.reshape(next_other_states1, newshape=[-1, state_size])
        next_other_states2 = np.reshape(next_other_states2, newshape=[-1, state_size])

        other_actions = np.hstack([other_actions1, other_actions2])
        critic_action_input = sess.run(self.target_actor.pi_predict, feed_dict={self.input: next_states})

        critic_other_action_input = np.hstack([sess.run(t_actor1.pi_predict, feed_dict={t_actor1.input: next_other_states1})
        ,sess.run(t_actor2.pi_predict, feed_dict={t_actor2.input: next_other_states2})])

        target_q_value = sess.run(self.target_critic.q_predict, feed_dict={self.input: next_states,
                                                                       self.action_input: critic_action_input,
                                                                       self.other_actions: critic_other_action_input
                                                                       })

        rewards = np.reshape(rewards, newshape=[-1, 1])
        target_q_value = np.reshape(target_q_value, newshape=[-1, 1])
        targets = np.zeros([batch_size, 1])
        for i in range(batch_size):
            if dones[i]:
                targets[i] = rewards[i]
            else:
                targets[i] = rewards[i] + discount_factor * target_q_value[i]

        # Model Updates
        _, loss = sess.run([self.train_critic_op, self.loss],
                           feed_dict={self.input: states,
                                      self.action_input: actions,
                                      self.other_actions: other_actions,
                                      self.target_Q: targets})

        _, grads = sess.run([self.train_actor_op, self.policy_Grads],
                            feed_dict={self.input: states,
                                       self.action_input: actions,
                                       self.other_actions: other_actions})

        return loss

    def target_update(self, sess):
        sess.run([self.target_update_actor, self.target_update_critic])

    def target_init(self, sess):
        sess.run([self.target_init_actor, self.target_init_critic])

    def action(self, state, sess):
        return sess.run(self.actor.pi_predict, {self.input: state})

    def value(self, state, action, other_action, sess):
        return sess.run(self.critic.q_predict,
                        {self.input: state, self.action_input: action, self.other_actions: other_action})


# Parameter Noise ======================================================================================================
def mu_variable(shape):
    return tf.Variable(tf.random_uniform(shape, minval = -tf.sqrt(5/shape[0]), maxval = tf.sqrt(5/shape[0])))
def sigma_variable(shape):
    return tf.Variable(tf.constant(0.03, shape = shape))

def noisy_dense(input_, input_shape, mu_w, sig_w, mu_b, sig_b):
    eps_w = tf.random_normal(input_shape)
    eps_b = tf.random_normal([input_shape[1]])
    w_fc = tf.add(mu_w, tf.multiply(sig_w, eps_w))
    b_fc = tf.add(mu_b, tf.multiply(sig_b, eps_b))
    return tf.matmul(input_, w_fc) + b_fc
# ======================================================================================================================

def get_agents_action(state1, state2, state3, sess):
    agent1_action = agent1_ddpg.action(state=np.reshape(state1, newshape=[-1,state_size]), sess=sess)
    agent2_action = agent2_ddpg.action(state=np.reshape(state2, newshape=[-1,state_size]), sess=sess)
    agent3_action = agent3_ddpg.action(state=np.reshape(state3, newshape=[-1,state_size]), sess=sess)

    return agent1_action, agent2_action, agent3_action



if __name__ == "__main__":

    env = UnityEnvironment(file_name=env_name, worker_id=0)
    brain_name1 = env.brain_names[1]
    brain_name2 = env.brain_names[2]
    brain_name3 = env.brain_names[3]

    brain1 = env.brains[brain_name1]
    brain2 = env.brains[brain_name2]
    brain3 = env.brains[brain_name3]

    env_info = env.reset(train_mode=train_mode)


    # Agent Generation =======================================
    agent1_ddpg = MADDPGAgent('1')
    agent2_ddpg = MADDPGAgent('2')
    agent3_ddpg = MADDPGAgent('3')

    # Save & Load ============================================
    Saver = tf.train.Saver(max_to_keep=5)
    load_path = load_path
    # ========================================================

    # Session Initialize =====================================
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    sess = tf.Session(config=config)

    if load_model == True:
        ckpt = tf.train.get_checkpoint_state(load_path)
        Saver.restore(sess, ckpt.model_checkpoint_path)
        print("[Restore Model]")
    else:
        init = tf.global_variables_initializer()
        sess.run(init)
        print("[Initialize Model]")

    # TargetNework 초기화 ====================================
    agent1_ddpg.target_init(sess)
    agent2_ddpg.target_init(sess)
    agent3_ddpg.target_init(sess)
    print("[Target Networks Initialized]")
    # ========================================================

    # Tensorboard ============================================
    reward_history = [tf.Variable(0, dtype=tf.float32) for i in range(numAgent)]
    reward_op = [tf.summary.scalar('Agent_' + str(i) + '_reward', reward_history[i]) for i in range(numAgent)]
    summary_writer = tf.summary.FileWriter('./three_summary', graph=tf.get_default_graph())
    print("Tensorbard Initialized")
    # ========================================================

    # Reset Environment =======================
    env_info = env.reset(train_mode=train_mode)
    print("[Env Reset]")
    # =========================================
    step = 0
    for episode in range(run_episode):

        env_info = env.reset(train_mode=train_mode)
        done = False

        # Brain Set ====================================
        state1 = env_info[brain_name1].vector_observations[0]
        episode_rewards1 = 0
        done1 = False

        state2 =  env_info[brain_name2].vector_observations[0]
        episode_rewards2 = 0
        done2 = False

        state3 =  env_info[brain_name3].vector_observations[0]
        episode_rewards3 = 0
        done3 = False

        # =============================================
        rewards1 = []
        losses1 = []
        rewards2 = []
        losses2 = []
        rewards3 = []
        losses3 = []
        while not done:
            step += 1
            agent1_action, agent2_action, agent3_action = get_agents_action(state1, state2, state3, sess)

            env_info = env.step(vector_action = {brain_name1: [agent1_action[0]], brain_name2: [agent2_action[0]], brain_name3: [agent3_action[0]]})

            next_state1 = env_info[brain_name1].vector_observations[0]
            reward1 = env_info[brain_name1].rewards[0]
            episode_rewards1 += reward1
            done1 = env_info[brain_name1].local_done[0]

            next_state2 = env_info[brain_name2].vector_observations[0]
            reward2 = env_info[brain_name2].rewards[0]
            episode_rewards2 += reward2
            done2 = env_info[brain_name2].local_done[0]

            next_state3 = env_info[brain_name3].vector_observations[0]
            reward3 = env_info[brain_name3].rewards[0]
            episode_rewards3 += reward3
            done3 = env_info[brain_name3].local_done[0]

            done = done1 or done2 or done3

            # Memory Set ==============================
            if train_mode:
                data1 = [state1, agent1_action[0], agent2_action[0], agent3_action[0],reward1,
                         next_state1, next_state2, next_state3, done1]
                data2 = [state2, agent2_action[0], agent3_action[0], agent1_action[0],reward2,
                         next_state2, next_state3, next_state1, done2]
                data3 = [state3, agent3_action[0], agent1_action[0], agent2_action[0],reward3,
                         next_state3, next_state1, next_state2, done3]

                agent1_ddpg.append_sample(data1)
                agent2_ddpg.append_sample(data2)
                agent3_ddpg.append_sample(data3)

            state1 = next_state1
            state2 = next_state2
            state3 = next_state3

            if episode > start_train_episode and len(agent1_ddpg.memory) > batch_size:
                loss1 = agent1_ddpg.train_models(agent2_ddpg.target_actor, agent3_ddpg.target_actor,  sess)
                loss2 = agent2_ddpg.train_models(agent3_ddpg.target_actor, agent1_ddpg.target_actor,  sess)
                loss3 = agent3_ddpg.train_models(agent1_ddpg.target_actor, agent2_ddpg.target_actor,  sess)

                losses1.append(loss1)
                losses2.append(loss2)
                losses3.append(loss3)

            if episode > 0 and step % target_update_step == 0:
                agent1_ddpg.target_update(sess)
                agent2_ddpg.target_update(sess)
                agent3_ddpg.target_update(sess)

        rewards1.append(episode_rewards1)
        rewards2.append(episode_rewards2)
        rewards3.append(episode_rewards3)

        if episode % print_interval == 0:
            print(
                "step {} / episode: {} / reward1: {:.2f} / reward2: {:.2f} / reward3: {:.2f} / memory_len:{}".format
                (step, episode, np.mean(rewards1), np.mean(rewards2),  np.mean(rewards3), len(agent1_ddpg.memory)))
            print(
                "loss1: {:.2f} / loss2: {:.2f} / loss3: {:.2f}  ".format
                (np.mean(losses1), np.mean(losses2), np.mean(losses3)))
            rewards1 = []
            rewards2 = []
            rewards3 = []
            losses1 = []
            losses2 = []
            losses3 = []

        if episode % save_interval == 0:
            Saver.save(sess, save_path + "model.ckpt")


