import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plotCols(file_path,
             delim='\t',
             x_axis_idx=0,
             y_axis_idx=1,
             x_label='',
             y_label='',
             save_path='TEST.png'):
    df = pd.read_csv(
        file_path,
        delimiter=delim,
        header=None)

    df = df.dropna()
    x_axis = np.array(df.iloc[:, x_axis_idx])
    y_axis = np.array(df.iloc[:, y_axis_idx])

    # Remove outliers in x
    x_mean = np.mean(x_axis)
    x_sd = np.std(x_axis)
    valid_idx = np.squeeze(np.argwhere((x_axis < (x_mean + 3 * x_sd)) & (x_axis > (x_mean - 3 * x_sd))))
    x_axis = x_axis[valid_idx]
    y_axis = y_axis[valid_idx]

    # Remove outliers in y
    y_mean = np.mean(y_axis)
    y_sd = np.std(y_axis)
    valid_idx = np.squeeze(np.argwhere((y_axis < (y_mean + 3 * y_sd)) & (y_axis > (y_mean - 3 * y_sd))))
    x_axis = x_axis[valid_idx]
    y_axis = y_axis[valid_idx]

    # Sort
    idx_order = np.argsort(x_axis)
    x_axis = x_axis[idx_order]
    y_axis = y_axis[idx_order]

    # print(x_axis, y_axis)
    plt.scatter(x_axis, y_axis)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(y_label + ' vs ' + x_label)
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()

def plot_reward_and_bit_rate(file_path,
                             save_dir="",
                             reward_idx=-2,
                             bit_rate_idx=0):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    test_case = os.path.basename(file_path)

    plotCols(file_path,
        x_axis_idx=-1,
        y_axis_idx=-2,
        x_label='Scalar Value Output',
        y_label='Reward',
        save_path=os.path.join(save_dir, test_case + "_{}_scalar.png".format("reward")))

    plotCols(file_path,
        x_axis_idx=-1,
        y_axis_idx=0,
        x_label='Scalar Value Output',
        y_label='Bit Rate',
        save_path=os.path.join(save_dir, test_case + "_{}_scalar.png".format("bit_rate")))

def flatten_input_data(obs,
                       a_dim=6,
                       s_dim=[6,8]):
    '''
    Given an observation (shape 6 X 8), flatten to 1D input for input to Keras
    We want to keep:
        * last bit rate
        * last buffer size
        * all bandwidth throughputs
        * all bandwidth times
        * last a_dim next_chunk_sizes
        * last n_chunk_rem
    Arg(s):
        obs : 6 X 8 2D np array
            original input data
        a_dim : int
            number of indices for next_chunk_sizes
        s_dim : list(int)
            shape of the state/obs
    Returns:
        np 1D array (25 elements)
    '''
    assert obs.shape == tuple(s_dim)
    flattened = []
    # Append previous bit rate
    flattened.append(obs[0, -1])

    # Append previous buffer size
    flattened.append(obs[1, -1])

    # Append all of throughput
    flattened += list(obs[2, :])

    # Append all of time
    flattened += list(obs[3, :])

    # Append last a_dim of next chunk sizes
    flattened += list(obs[4, :a_dim])

    # Append number chunks remaining
    flattened.append(obs[5, -1])

    assert len(flattened) == (3 + 2 * s_dim[1] + a_dim)

    return flattened

'''
Make data COMET compatible
'''
def add_headers(df,
                headers=[]):
    '''
    Add headers to the dataframe
    '''
    assert len(headers) == len(df.columns)
    df.columns = headers

def add_col(df,
            col_name,
            index=0,
            random=True,
            min_val=0,
            max_val=1,
            values=None):
    '''
    Add column to df. Randomly fill with values betwen min and max if random is True
    Otherwise fill in with values.
    random cannot be false while values is None
    '''
    n_rows = len(df)
    assert 0 <= index and index <= len(df.columns)

    if not random:
        assert values is not None
    else:
        values = (max_val - min_val) * np.random.rand(n_rows) + min_val

    # insert values at the index specified
    df.insert(
        loc=index,
        column=col_name,
        value=values
    )

def make_np_data_comet_compatible(np_data,
                          gt_name,
                          headers):
    '''
    Convert np_data to pandas data frame and make comet compatible.
    Return dataframe
    '''
    # Convert to Pandas dataframe
    df = pd.DataFrame(np_data)

    # Add headers to columns
    add_headers(
        df=df,
        headers=headers
    )

    # Add GT column
    add_col(
        df=df,
        col_name=gt_name,
        index=0,
        random=True,
        min_val=df['prev_bit_rate'].min(),
        max_val=df['prev_bit_rate'].max(),
        values=None
    )

    return df

if __name__ == "__main__":
    vehicles = ['bus', 'car', 'ferry', 'metro', 'train', 'tram']
    n_max = 10
    save_dir = 'results/ffd_3_64/graphs'
    for vehicle in vehicles:
        for i in range(1, n_max + 1):
            file_path = 'results/ffd_3_64/ffd_3_64_test_results/log_sim_rl_norway_{}_{}'.format(vehicle, i)
            plot_reward_and_bit_rate(
                file_path,
                save_dir=save_dir,
            )




    # plotCols(file_path,
    #     x_axis_idx=-1,
    #     y_axis_idx=-2,
    #     x_label='Scalar Value Output',
    #     y_label='Reward',
    #     save_path=os.path.join(save_dir, 'log_sim_rl_norway_{}_{}_reward_scalar.png'.format(vehicle, n)))

    # plotCols(file_path,
    #     x_axis_idx=-1,
    #     y_axis_idx=0,
    #     x_label='Scalar Value Output',
    #     y_label='Bit Rate',
    #     save_path=os.path.join(save_dir, 'log_sim_rl_norway_{}_{}_bitrate_scalar.png'.format(vehicle, n)))