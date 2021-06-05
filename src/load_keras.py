import ppo2_keras_ffd as network
import global_constants as settings
import os
import tensorflow as tf
import numpy as np
from tensorflow.python.tools import inspect_checkpoint

# Taken from sim/agent.py
S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
# NN_MODEL = 'test/models/pretrain_linear_reward.ckpt'
NN_MODEL = 'sim/results/pretrain_linear_reward.ckpt'
KERAS_MODEL_PATH = 'keras_models/keras_model.h5'

def load_ckpt_store_h5(actor,
                       ckpt_path,
                       save_path=None,
                       csv_save_dir=None,
                       ckpt_layer_names=[
                        'actor/FullyConnected',
                        'actor/FullyConnected_1',
                        'actor/FullyConnected_2',
                        'actor/FullyConnected_3',
                        'actor/FullyConnected_4'
                        ],
                       keras_layer_names = [
                        'dense_1',
                        'dense_2',
                        'dense_3',
                        'dense_4',
                        'dense_5'
                        ]):
    '''
    Restore weights of Keras model from CKPT file and save as h5 file
    Arg(s):
        ckpt_path : str
            path to ckpt file
        save_path : None or str
            path to save h5 model in. If None, stores in same path as ckpt_path, but with .h5 extension
        csv_save_dir : None or str
            if None, do not save weights as CSV. Otherwise save weights as CSV files in this directory
        ckpt_layer_names : list[str]
            list of names of ckpt layers (not including '/W' or '/b' for weights and biases)
        keras_layer_names : list[str]
            list of names of corresponding h5 layers
    Returns:
        None
    '''

    # Checks for paths
    if save_path is None:
        save_path = ckpt_path.replace('.ckpt', '.h5')

    if csv_save_dir is not None:
        # Clear directory if it exists
        if os.path.isdir(csv_save_dir):
            os.system("rm -rf {}".format(csv_save_dir))
        os.makedirs(csv_save_dir)

    assert len(ckpt_layer_names) == len(keras_layer_names)

    # Obtain reader to understand ckpt checkpoints
    reader = tf.train.NewCheckpointReader(ckpt_path)

    for layer_idx, (ckpt_layer_name, keras_layer_name) in enumerate(zip(ckpt_layer_names, keras_layer_names)):
        # Obtain old weights (to check shape later)
        print("Old layer: {}\nNew Layer: {}".format(ckpt_layer_name, keras_layer_name))
        old_weights = actor.model.get_layer(keras_layer_name).get_weights()

        # Extract weight values from ckpt layers and store into keras layers
        weights = reader.get_tensor(ckpt_layer_name + '/W')

        print("keras weights shape: {} ckpt weight shape: {}".format(old_weights[0].shape, weights.shape))
        # Assume that one dimension is a 1 and can be squeezed to match
        if old_weights[0].shape != weights.shape:
            weights = np.squeeze(weights)
            assert old_weights[0].shape == weights.shape

        biases = reader.get_tensor(ckpt_layer_name + '/b')


        actor.model.get_layer(keras_layer_name).set_weights([weights, biases])

        new_weights = actor.model.get_layer(keras_layer_name).get_weights()

        # Sanity check
        for old_weight, new_weight in zip(old_weights, new_weights):
            assert not (old_weight == new_weight).all()

        # Save to CSV, if desired
        if csv_save_dir is not None:
            '''
            Save weights
            '''

            # Check if 3d tensor
            if 2 < len(weights.shape):
                # iterate through each kernel
                for kernel_idx in range(weights.shape[0]):
                    kernel_weights = weights[kernel_idx]
                    np.savetxt(
                        os.path.join(csv_save_dir, "weights_layer{}_kernel{}.csv".format(layer_idx, kernel_idx)),
                        kernel_weights,
                        delimiter=",")
            else:
                np.savetxt(
                    os.path.join(csv_save_dir, "weights_layer{}.csv".format(layer_idx)),
                    weights,
                    delimiter=",")

            '''
            Save biases
            '''
            np.savetxt(
                os.path.join(csv_save_dir, "bias_layer{}.csv".format(layer_idx)),
                    biases,
                    delimiter=",")


    print("Saving model to {}".format(save_path))
    actor.model.save(save_path)

def load_model_to_keras(ckpt_path=NN_MODEL,
                        h5_save_path=None,
                        csv_save_dir=None,
                        state_info=S_INFO,
                        state_len=S_LEN,
                        actor_dim=A_DIM,
                        actor_lr=ACTOR_LR_RATE,
                        ckpt_layer_names=[
                            'actor/FullyConnected',
                            'actor/FullyConnected_1',
                            'actor/Conv1D',
                            'actor/Conv1D_1',
                            'actor/Conv1D_2',
                            'actor/FullyConnected_2',
                            'actor/FullyConnected_3',
                            'actor/FullyConnected_4'
                            ],
                        keras_layer_names = [
                            'dense_1',
                            'dense_2',
                            'conv1d_1',
                            'conv1d_2',
                            'conv1d_3',
                            'dense_3',
                            'dense_4',
                            'dense_5'
                            ]):
    '''
    Create actor model in keras, load in a ckpt checkpoint, and save as h5
        Arg(s):
            ckpt_path : str
                path to ckpt file
            h5_save_path : None or str
                path to save h5 model in. If None, stores in same path as ckpt_path, but with .h5 extension
            csv_save_dir : None or str
                if None, do not save weights as CSV. Otherwise save weights as CSV files in this directory
            state_info : int
                number of variables in state
            state_len : int
                length of state
            actor_dim : int
                number of dimensions in actor
            actor_lr : float
                learning rate for actor
            ckpt_layer_names : list[str]
                list of names of ckpt layers (not including '/W' or '/b' for weights and biases)
            keras_layer_names : list[str]
                list of names of corresponding h5 layers
        Returns:
            actor model in Keras
    '''
    with tf.Session() as sess:

        actor = network.Network(
            sess=sess,
            state_dim=[state_info, state_len],
            action_dim=actor_dim,
            learning_rate=actor_lr,
            n_dense=settings.N_DENSE_LAYERS
        )
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters
        if ckpt_path is not None:  # nn_model is the path to file
            load_ckpt_store_h5(actor,
                ckpt_path=ckpt_path,
                save_path=h5_save_path,
                csv_save_dir=csv_save_dir,
                ckpt_layer_names=ckpt_layer_names,
                keras_layer_names=keras_layer_names
                )
            print("Model restored.")

        # Return saved model
        return actor

def load_actor(h5_path,
               state_info=S_INFO,
               state_len=S_LEN,
               actor_dim=A_DIM,
               actor_lr=ACTOR_LR_RATE):
    with tf.Session() as sess:

        actor = network.Network(
            sess=sess,
            state_dim=[state_info, state_len],
            action_dim=actor_dim,
            learning_rate=actor_lr,
            n_dense=settings.N_DENSE_LAYERS
        )
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  # save neural net parameters

        actor.load_weights(h5_path)
    return actor

def save_actor_end(actor,
                   save_h5_path=None,
                   save_csv_dir=None):
    '''
    Save the final two layers of the actor
    Arg(s):
        actor: ActorNetwork object
            original model from which to save weights
        save_h5_path : str or None
                save .h5 model to path if specified
        save_csv_dir : str or None
            save weights as CSV files in this directory if specified
    '''
    actor_end = actor.save_actor_end(
        save_h5_path=save_h5_path,
        save_csv_dir=save_csv_dir)
    return actor_end
def save_actor_end2(h5_path,
               state_info=S_INFO,
               state_len=S_LEN,
               actor_dim=A_DIM,
               actor_lr=ACTOR_LR_RATE,
               save_h5_path=None,
                   save_csv_dir=None):
    with tf.Session() as sess:

        actor = network.Network(
            sess=sess,
            state_dim=[state_info, state_len],
            action_dim=actor_dim,
            learning_rate=actor_lr
        )
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  # save neural net parameters

        actor.load_weights(h5_path)
        actor_end = actor.save_actor_end(
            save_h5_path=save_h5_path,
        save_csv_dir=save_csv_dir)
        return actor_end

def get_fc_layer_names():
    '''
    From the hyperparameter settings.N_DENSE_LAYER, extract the number of layers
    Return [layer_names_ckpt], [layer_names_keras]

    '''
    n_dense = settings.N_DENSE_LAYERS + 4
    ckpt_layer_names = ['actor/FullyConnected']
    keras_layer_names = []
    for i in range(1, n_dense + 1):
        # remove the layer that outputs the 6 bit rate probabilities
        if i == n_dense - 1:
            ckpt_layer_names[-1] = 'actor/FullyConnected_{}'.format(i)
            continue
        if i < n_dense:
            ckpt_layer_names.append('actor/FullyConnected_{}'.format(i))
        keras_layer_names.append('dense_{}'.format(i))
    return ckpt_layer_names, keras_layer_names


if __name__=="__main__":

    '''
    Load entire .ckpt model into Keras
    '''
    # Obtain appropriate layer names depending on architecture
    ckpt_layer_names, keras_layer_names = get_fc_layer_names()
    print(ckpt_layer_names, keras_layer_names)
    step = 95000
    date = '06022021'
    actor = load_model_to_keras(
        ckpt_path="results/{}_{}/summary/nn_model_ep_{}.ckpt".format(settings.MODEL_ARCH, date, step),
        csv_save_dir="keras_models/{}_06022021_{}/pensieve_{}.csv".format(settings.MODEL_ARCH, step, settings.MODEL_ARCH),
        h5_save_path='keras_models/{}_06022021_{}/pensieve_{}.h5'.format(settings.MODEL_ARCH, step, settings.MODEL_ARCH),
        ckpt_layer_names=ckpt_layer_names,
        keras_layer_names=keras_layer_names
    )
