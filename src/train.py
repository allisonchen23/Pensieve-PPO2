import multiprocessing as mp
import numpy as np
import logging
import os
import sys
from abr import ABREnv
import ppo2_feed_forward_dense as network
import tensorflow as tf
import global_constants as settings
import utils
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

S_DIM = [6, 8]
A_DIM = 6
ACTOR_LR_RATE =1e-4
CRITIC_LR_RATE = 1e-3
NUM_AGENTS = 12 #20
TRAIN_SEQ_LEN = 300  # take as a train batch
TRAIN_EPOCH = 100000 # 1000000
MODEL_SAVE_INTERVAL = 5000 # 300
RANDOM_SEED = 42
RAND_RANGE = 10000
MODEL_DIR = './models'
TRAIN_TRACES = './cooked_traces/'

PPO_TRAINING_EPO = 10


NN_MODEL = None
# clean up the summary results folder
os.system('rm -r ' + settings.SUMMARY_DIR)

if not os.path.exists(settings.SUMMARY_DIR):
    os.makedirs(settings.SUMMARY_DIR)

def testing(epoch, nn_model, log_file):
    # clean up the test results folder
    os.system('rm -r ' + settings.TEST_LOG_FOLDER)
    #os.system('mkdir ' + TEST_LOG_FOLDER)

    if not os.path.exists(settings.TEST_LOG_FOLDER):
        os.makedirs(settings.TEST_LOG_FOLDER)
    # run test script
    os.system('python rl_test.py ' + nn_model)

    # append test performance to the log
    rewards = []
    test_log_files = os.listdir(settings.TEST_LOG_FOLDER)
    for test_log_file in test_log_files:
        reward = []
        with open(settings.TEST_LOG_FOLDER + '/' + test_log_file, 'rb') as f:
            for line in f:
                parse = line.split()
                try:
                    reward.append(float(parse[-1]))
                except IndexError:
                    break
        rewards.append(np.mean(reward[1:]))

    rewards = np.array(rewards)

    rewards_min = np.min(rewards)
    rewards_5per = np.percentile(rewards, 5)
    rewards_mean = np.mean(rewards)
    rewards_median = np.percentile(rewards, 50)
    rewards_95per = np.percentile(rewards, 95)
    rewards_max = np.max(rewards)

    log_file.write(str(epoch) + '\t' +
                   str(rewards_min) + '\t' +
                   str(rewards_5per) + '\t' +
                   str(rewards_mean) + '\t' +
                   str(rewards_median) + '\t' +
                   str(rewards_95per) + '\t' +
                   str(rewards_max) + '\n')
    log_file.flush()

    return rewards_mean

def central_agent(net_params_queues, exp_queues):
    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS

    if not os.path.exists(settings.RESULTS_DIR):
        os.makedirs(settings.RESULTS_DIR)

    with tf.Session() as sess, open(settings.LOG_FILE + '_test.txt', 'w') as test_log_file:

        actor = network.Network(sess,
                state_dim=S_DIM, action_dim=A_DIM,
                learning_rate=ACTOR_LR_RATE,
                n_dense=settings.N_DENSE_LAYERS)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=1000)  # save neural net parameters

        # restore neural net parameters
        nn_model = NN_MODEL
        if nn_model is not None:  # nn_model is the path to file
            saver.restore(sess, nn_model)
            print("Model restored.")

        # while True:  # assemble experiences from agents, compute the gradients
        for epoch in range(TRAIN_EPOCH):
            # synchronize the network parameters of work agent
            actor_net_params = actor.get_network_params()
            for i in range(NUM_AGENTS):
                net_params_queues[i].put(actor_net_params)

            s, a, p, g = [], [], [], []
            for i in range(NUM_AGENTS):
                s_, a_, p_, g_ = exp_queues[i].get()
                s += s_
                a += a_
                p += p_
                g += g_
            s_batch = np.stack(s, axis=0)
            a_batch = np.vstack(a)
            p_batch = np.vstack(p)
            v_batch = np.vstack(g)

            for _ in range(PPO_TRAINING_EPO):
                actor.train(s_batch, a_batch, p_batch, v_batch, epoch)

            if epoch % MODEL_SAVE_INTERVAL == 0:
                print("Checkpoint at epoch {} / {}".format(epoch, TRAIN_EPOCH))
                # Save the neural net parameters to disk.
                save_path = saver.save(sess, settings.SUMMARY_DIR + "/nn_model_ep_" +
                                       str(epoch) + ".ckpt")
                testing(epoch,
                    settings.SUMMARY_DIR + "/nn_model_ep_" + str(epoch) + ".ckpt",
                    test_log_file)

def agent(agent_id, net_params_queue, exp_queue, dump_input_data=False):
    env = ABREnv(agent_id)
    with tf.Session() as sess, open(settings.SUMMARY_DIR + '/log_agent_' + str(agent_id), 'w') as log_file:
        actor = network.Network(sess,
                                state_dim=S_DIM, action_dim=A_DIM,
                                learning_rate=ACTOR_LR_RATE,
                                n_dense=settings.N_DENSE_LAYERS)

        # initial synchronization of the network parameters from the coordinator
        actor_net_params = net_params_queue.get()
        actor.set_network_params(actor_net_params)

        time_stamp = 0

        # Initialize variables for flattening data and create output directory
        flattened_input = []
        if dump_input_data \
            and agent_id == 0:
            if os.path.isdir(settings.TRAIN_DATA_DUMP_DIR):
                os.system("rm -rf {}".format(settings.TRAIN_DATA_DUMP_DIR))
            os.makedirs(settings.TRAIN_DATA_DUMP_DIR)

        for epoch in range(TRAIN_EPOCH):
            obs = env.reset()
            s_batch, a_batch, p_batch, r_batch = [], [], [], []
            for step in range(TRAIN_SEQ_LEN):
                s_batch.append(obs)
                if dump_input_data \
                    and agent_id == 0 \
                    and epoch % settings.SAVE_INPUT_DATA_INTERVAL == 0:
                    flattened_obs = utils.flatten_input_data(
                        obs,
                        a_dim=A_DIM,
                        s_dim=S_DIM
                    )
                    flattened_input.append(flattened_obs)

                action_prob, scalar_val = actor.predict(
                    np.reshape(obs, (1, S_DIM[0], S_DIM[1])))

                action_cumsum = np.cumsum(action_prob)
                bit_rate = (action_cumsum > np.random.randint(
                    1, RAND_RANGE) / float(RAND_RANGE)).argmax()

                obs, rew, done, info = env.step(bit_rate)

                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1
                a_batch.append(action_vec)
                r_batch.append(rew)
                p_batch.append(action_prob)
                if done:
                    break
            v_batch = actor.compute_v(s_batch, a_batch, r_batch, done)
            exp_queue.put([s_batch, a_batch, p_batch, v_batch])

            actor_net_params = net_params_queue.get()
            actor.set_network_params(actor_net_params)
            if dump_input_data \
                and agent_id == 0 \
                and epoch % settings.SAVE_INPUT_DATA_INTERVAL == 0:
                flattened_input = np.array(flattened_input)
                flattened_input = utils.make_np_data_comet_compatible(
                    np_data=flattened_input,
                    gt_name='bit_rate',
                    headers=[
                        "prev_bit_rate",
                        "buffer_size",
                        "bandwidth_throughput_0",
                        "bandwidth_throughput_1",
                        "bandwidth_throughput_2",
                        "bandwidth_throughput_3",
                        "bandwidth_throughput_4",
                        "bandwidth_throughput_5",
                        "bandwidth_throughput_6",
                        "bandwidth_throughput_7",
                        "bandwidth_time_0",
                        "bandwidth_time_1",
                        "bandwidth_time_2",
                        "bandwidth_time_3",
                        "bandwidth_time_4",
                        "bandwidth_time_5",
                        "bandwidth_time_6",
                        "bandwidth_time_7",
                        "next_chunk_sizes_0",
                        "next_chunk_sizes_1",
                        "next_chunk_sizes_2",
                        "next_chunk_sizes_3",
                        "next_chunk_sizes_4",
                        "next_chunk_sizes_5",
                        "n_chunks_remaining"]
                )
                # np.savetxt(
                #     settings.SUMMARY_DIR + '/log_agent_' + str(agent_id),
                #     flattened_input,
                #     delimiter=",")
                
                flattened_input.to_csv(os.path.join(settings.TRAIN_DATA_DUMP_DIR, 'train_data_epoch_{}.csv'.format(epoch)))
                # Reset flattened_input
                flattened_input = []


def main():

    np.random.seed(RANDOM_SEED)
    dump_input_data = settings.DUMP_INPUT_DATA
    # inter-process communication queues
    net_params_queues = []
    exp_queues = []
    for i in range(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    # create a coordinator and multiple agent processes
    # (note: threading is not desirable due to python GIL)
    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queues, exp_queues))
    coordinator.start()

    agents = []
    for i in range(NUM_AGENTS):
        agents.append(mp.Process(target=agent,
                                 args=(i,
                                       net_params_queues[i],
                                       exp_queues[i],
                                       dump_input_data)))
    for i in range(NUM_AGENTS):
        agents[i].start()

    # wait unit training is done
    coordinator.join()

    # Not sure if this will fix the hanging at the end
    for i in range(NUM_AGENTS):
        agents[i].join()


if __name__ == '__main__':
    main()
