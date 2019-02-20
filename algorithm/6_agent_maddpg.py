import numpy as np
import tensorflow as tf
from maddpg import MADDPGAgent
from mlagents.envs import UnityEnvironment


########################################
agent_num = 2
state_size = 39 * 3
action_size = 3
batch_size = 1024

run_episode = 1000000
test_episode = 1000
start_train_episode = 10000
print_interval = 100
save_interval = 500

load_model = False
train_mode = True

env_name = "C:/Users/asdfw/Desktop/Two_Leg_Agent_Multi/Compile/Compile"
save_path = "./maddpg_models/twoleg/"
load_path = "./maddpg_models/twoleg/"
#######################################

def get_agents_action(state1, state2, sess):
    agent1_action = agent1_ddpg.action(state=np.reshape(state1, newshape=[-1, state_size]), train_mode=train_mode, sess=sess)
    agent2_action = agent2_ddpg.action(state=np.reshape(state2, newshape=[-1, state_size]), train_mode=train_mode, sess=sess)
    return agent1_action, agent2_action

if __name__ == "__main__":

    # Environment Setting =================================
    env = UnityEnvironment(file_name=env_name, worker_id=0)

    # Set the brain for each players ======================
    brain_name1 = env.brain_names[1]
    brain_name2 = env.brain_names[2]
    brain1 = env.brains[brain_name1]
    brain2 = env.brains[brain_name2]

    step = 0
    rewards1 = []
    losses1 = []
    rewards2 = []
    losses2 = []
    agent_num = 2

    # Agent Generation =======================================
    agent1_ddpg = MADDPGAgent(agent_num, state_size, action_size, '1')
    agent2_ddpg = MADDPGAgent(agent_num, state_size, action_size, '2')

    # Save & Load ============================================
    Saver = tf.train.Saver(max_to_keep=5)
    load_path = load_path
    # ========================================================

    # Session Initialize =====================================
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
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
    print("[Target Networks Initialized]")
    # ========================================================


    # Tensorboard ============================================
    reward_history = [tf.Variable(0, dtype=tf.float32) for i in range(agent_num)]
    reward_op = [tf.summary.scalar('Agent_' + str(i) + '_reward', reward_history[i]) for i in range(agent_num)]
    summary_writer = tf.summary.FileWriter('./three_summary', graph=tf.get_default_graph())
    print("Tensorbard Initialized")
    # ========================================================

    # Reset Environment =======================
    env_info = env.reset(train_mode=train_mode)
    print("[Env Reset]")
    # =========================================

    for episode in range(run_episode + test_episode):
        if episode > run_episode:
            train_mode = False

        env_info = env.reset(train_mode=train_mode)
        done = False
        # Brain Set ====================================
        state1 = env_info[brain_name1].vector_observations[0]
        episode_rewards1 = 0
        done1 = False

        state2 =  env_info[brain_name2].vector_observations[0]
        episode_rewards2 = 0
        done2 = False
        # =============================================
        while not done:
            step += 1

            agent1_action, agent2_action = get_agents_action(state1, state2, sess)
            # e-greedy ======================================
            action1 = agent1_action[0]
            action2 = agent2_action[0]

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

            # Memory Set ==============================
            if train_mode:
                data1 = [state1, agent1_action[0], agent2_action[0], reward1,
                         next_state1, next_state2, done1]
                data2 = [state2, agent2_action[0], agent1_action[0], reward2,
                         next_state2, next_state1, done2]

                agent1_ddpg.append_sample(data1)
                agent2_ddpg.append_sample(data2)
            # =========================================

            state1 = next_state1
            state2 = next_state2

            if step > 1000:
                break

        if episode > start_train_episode and len(agent1_ddpg.memory) > batch_size:
            loss1 = agent1_ddpg.train_models(agent2_ddpg.target_actor, sess)
            loss2 = agent2_ddpg.train_models(agent1_ddpg.target_actor, sess)

            losses1.append(loss1)
            losses2.append(loss2)

        if episode > 0 and episode % 15 == 0:
            agent1_ddpg.target_update(sess)
            agent2_ddpg.target_update(sess)

        rewards1.append(episode_rewards1)
        rewards2.append(episode_rewards2)

        if episode % print_interval == 0 and episode != 0:
            print(
                "episode: {} / reward1: {:.2f} / reward2: {:.2f} /  memory_len:{}".format
                (episode, np.mean(rewards1), np.mean(rewards2),  len(agent1_ddpg.memory)))
            print(
                "loss1: {:.2f} / loss2: {:.2f} ".format
                (np.mean(losses1), np.mean(losses2)))
            rewards1 = []
            rewards2 = []
            losses1 = []
            losses2 = []

        if episode % save_interval == 0:
            Saver.save(sess, save_path + "model.ckpt")
