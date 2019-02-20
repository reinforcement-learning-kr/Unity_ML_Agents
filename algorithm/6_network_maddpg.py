import tensorflow as tf
from collections import deque
import random
import numpy as np

########################################
discount_factor = 0.99
learning_rate = 0.0001
batch_size = 1024
mem_maxlen = 1000000
noise_std = 0.01
noise_option = 'layer_noise'
###########################################

class Critic(object):
    def __init__(self, state_size, action_size, input, action_input, other_action, target_value, model_name="critic", agent_num=2, reuse=False):

        self.state_size = state_size
        self.action_size = action_size
        self.agent_num = agent_num

        # =================================
        self.input = input
        self.action = action_input
        self.other_action = other_action
        self.target_value = target_value
        self.concat_action = tf.concat([self.action, self.other_action], axis=1)
        # =================================

        with tf.variable_scope(name_or_scope=model_name, reuse=reuse):

            x = tf.layers.dense(self.input, 128)
            x = tf.contrib.layers.layer_norm(x, center=True, scale=True)
            self.fc1 = tf.nn.relu(x)
            self.concat = tf.concat([self.fc1, self.concat_action], axis=1)
            x2 = tf.layers.dense(self.concat, 64)
            x2 = tf.layers.dense(x2, 64)
            x2 = tf.contrib.layers.layer_norm(x2, center=True, scale=True)
            self.fc2 = tf.nn.relu(x2)
            self.value = tf.layers.dense(self.fc2, units=1, activation=None, kernel_initializer= tf.contrib.layers.xavier_initializer())

        self.loss = tf.losses.mean_squared_error(self.target_value, self.value)
        self.train_critic_op = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.loss)
        self.trainable_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, model_name)


class Actor(object):
    def __init__(self, state_size, action_size, input, model_name="actor"):

        self.agent_num = agent_num
        self.state_size = state_size
        self.action_size = action_size

        # =================================
        self.input = input
        # =================================
        with tf.variable_scope(name_or_scope=model_name):

            x = tf.layers.dense(self.input, 128)
            x = tf.contrib.layers.layer_norm(x, center=True, scale=True)
            self.fc1 = tf.nn.relu(x)
            x2 = tf.layers.dense(self.fc1, 64)
            x2 = tf.contrib.layers.layer_norm(x2, center=True, scale=True)
            self.fc2 = tf.nn.relu(x2)
            self.action = tf.multiply(tf.layers.dense(self.fc2, units=self.action_size, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer()), 3)

        # action bound
        self.trainable_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, model_name)



class MADDPGAgent(object):
    def __init__(self, agent_num, state_size, action_size, idx):

        # Default Environment Information =====
        self.state_size = state_size
        self.action_size = action_size
        self.agent_num = agent_num
        self.gamma = 0.95
        self.idx = idx
        # =====================================

        # Experience Buffer ===================
        self.memory = deque(maxlen=mem_maxlen)
        self.batch_size = batch_size
        # =====================================

        # Placeholer =============================================================================
        self.input = tf.placeholder(shape=[None, self.state_size], dtype=tf.float32)
        self.action_input = tf.placeholder(shape=[None, self.action_size], dtype=tf.float32)
        self.other_actions = tf.placeholder(shape=[None, self.action_size * (self.agent_num-1)], dtype=tf.float32)
        self.target_value = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.reward = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        # ========================================================================================

        self.actor = Actor(self.state_size, self.action_size, self.input, "actor_" + idx)
        self.critic = Critic(self.state_size, self.action_size, self.input, self.action_input, self.other_actions, self.target_value, "critic_" + idx, self.agent_num, reuse=False)

        self.onActor_vars = [i for i in tf.trainable_variables() if "actor_" + idx in i.name]
        self.onCritic_vars = [i for i in tf.trainable_variables() if "critic_" + idx in i.name]

        self.target_actor = Actor(self.state_size, self.action_size, self.input, "target_actor_" + idx)
        self.target_critic = Critic(self.state_size, self.action_size, self.input, self.action_input, self.other_actions, self.target_value, "target_critic" + idx, self.agent_num, reuse=False)

        self.targetActor_vars = [i for i in tf.trainable_variables() if "target_actor_" + idx in i.name]
        self.targetCritic_vars = [i for i in tf.trainable_variables() if "target_critic_" + idx in i.name]

        action_Grad = tf.clip_by_value(tf.gradients(self.critic.value, self.action_input), -0.1, 0.1)
        self.policy_Grads = tf.gradients(ys=self.actor.action, xs=self.onActor_vars, grad_ys=action_Grad)
        for idx, grads in enumerate(self.policy_Grads):
            self.policy_Grads[idx] = -grads / batch_size
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_actor_op = tf.train.AdamOptimizer(learning_rate = learning_rate).apply_gradients(zip(self.policy_Grads, self.onActor_vars))


        # Target Update
        tau = 0.1
        self.target_update_actor = [tf.assign(target, (tau) * online + (1 - tau) * target) for online, target in zip(self.onActor_vars, self.targetActor_vars)]
        self.target_update_critic = [tf.assign(target, (tau) * online + (1 - tau) * target) for online, target in zip(self.onCritic_vars, self.targetCritic_vars)]
        self.target_init_actor = [tf.assign(target, online) for online, target in zip(self.onActor_vars, self.targetActor_vars)]
        self.target_init_critic = [tf.assign(target, online) for online, target in zip(self.onCritic_vars, self.targetCritic_vars)]

    def append_sample(self, data):
        self.memory.append((data[0], data[1], data[2], data[3], data[4], data[5], data[6]))

    def train_models(self, t_actor1, sess):

        mini_batch = random.sample(self.memory, batch_size)

        states = []
        actions = []
        other_actions1 = []
        rewards = []
        next_states = []
        next_other_states1 = []
        dones = []

        for i in range(batch_size):
            states.append(mini_batch[i][0])
            actions.append(mini_batch[i][1])
            other_actions1.append(mini_batch[i][2])
            rewards.append(mini_batch[i][3])
            next_states.append(mini_batch[i][4])
            next_other_states1.append(mini_batch[i][5])
            dones.append(mini_batch[i][6])

        states = np.reshape(states, newshape=[-1, self.state_size])
        actions = np.reshape(actions, newshape=[-1, self.action_size])
        next_states = np.reshape(next_states, newshape=[-1, self.state_size])
        next_other_states1 = np.reshape(next_other_states1, newshape=[-1, self.state_size])


        # S(t)에서의 other actions
        other_actions = other_actions1
        # S(t+1)에서의 actions
        critic_action_input = sess.run(self.target_actor.action, feed_dict={self.input: next_states})
        # S(t+1)에서의 other actions
        critic_other_action_input = sess.run(t_actor1.action, feed_dict={t_actor1.input: next_other_states1})
        target_q_value = sess.run(self.target_critic.value, feed_dict={self.input: next_states,
                                                                       self.action_input: critic_action_input,
                                                                       self.other_actions: critic_other_action_input
                                                                      })

        rewards = np.reshape(rewards, newshape=[-1, 1])
        target_q_value = np.reshape(target_q_value, newshape=[-1,1])
        targets = np.zeros([batch_size, 1])
        for i in range(batch_size):
            if dones[i]:
                targets[i] = rewards[i]
            else:
                targets[i] = rewards[i] + self.gamma * target_q_value[i]

        # Model Updates
        _, loss = sess.run([self.critic.train_critic_op, self.critic.loss],
                                   feed_dict={self.input: states,
                                              self.action_input: actions,
                                              self.other_actions: other_actions,
                                              self.target_value: targets})

        _, grads = sess.run([self.train_actor_op,self.policy_Grads],
                                   feed_dict={self.input: states,
                                              self.action_input: actions,
                                              self.other_actions: other_actions})

        return loss

    def target_update(self, sess):
        sess.run([self.target_update_actor, self.target_update_critic])

    def target_init(self, sess):
        sess.run([self.target_init_actor, self.target_init_critic])

    def action(self, state, train_mode, sess):
        if train_mode == True:
            if noise_option == 'layer_noise':
                self.apply_noise(noise_std, sess)
                action = sess.run(self.actor.action, {self.input: state})
                self.reset_vars(sess)
                return action
            else:
                action = sess.run(self.actor.action, {self.input: state})
                return action
        else:
            return sess.run(self.actor.action, {self.input: state})

    def value(self, state, action, other_action, sess):
        return sess.run(self.critic.value, {self.input: state, self.action_input: action, self.other_actions: other_action})

    def apply_noise(self, noise_std=1.0, sess=None):
        var_names = [var for var in tf.global_variables() if "actor_" + self.idx in var.op.name]
        self.old_var = sess.run(var_names)
        var_shapes = [i.shape for i in self.old_var]
        new_var = [i + np.random.normal(0, noise_std, size=j) for i, j in zip(self.old_var, var_shapes)]
        for i, j in zip(var_names, new_var):
            sess.run(i.assign(j))
        return

    def reset_vars(self, sess):
        var_names = [var for var in tf.global_variables() if "actor_" + self.idx in var.op.name]
        for i, j in zip(var_names, self.old_var):
            sess.run(i.assign(j))
        return



