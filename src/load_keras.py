import a3c_keras
import tensorflow as tf
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

        actor = a3c_keras.ActorNetwork(
            sess=sess,
            state_dim=[state_info, state_len],
            action_dim=actor_dim,
            learning_rate=actor_lr
        )
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters
        if ckpt_path is not None:  # nn_model is the path to file
            actor.load_ckpt_store_h5(
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

        actor = a3c_keras.ActorNetwork(
            sess=sess,
            state_dim=[state_info, state_len],
            action_dim=actor_dim,
            learning_rate=actor_lr
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

        actor = a3c_keras.ActorNetwork(
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

if __name__=="__main__":

    '''
    Load entire .ckpt model into Keras
    '''
    # actor = load_model_to_keras(
    #     csv_save_dir="keras_models/pensieve_csv",
    #     h5_save_path='keras_models/pensieve.h5'
    # )

    # actor = load_actor(KERAS_MODEL_PATH)
    # actor_end = save_actor_end(
    #     actor=actor,
    #     save_h5_path='keras_models/pensieve_end.h5',
    #     save_csv_dir='keras_models/pensieve_end_csv')

    save_actor_end2(
        h5_path='keras_models/pensieve.h5',
        save_csv_dir="keras_models/pensieve_csv",
        save_h5_path='keras_models/pensieve.h5')
