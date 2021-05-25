import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
    df.head()
    x_axis = np.array(df.iloc[:, x_axis_idx])
    y_axis = np.array(df.iloc[:, y_axis_idx])

    # Remove outliers
    x_mean = np.mean(x_axis)
    x_sd = np.std(x_axis)
    print(x_mean, x_sd)
    valid_idx = np.squeeze(np.argwhere((x_axis < (x_mean + 3 * x_sd)) & (x_axis > (x_mean - 3 * x_sd)))) #np.argwhere(np.where(x_axis > THRESH, x_axis, np.zeros_like(x_axis)))
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

if __name__ == "__main__":
    plotCols('TEST_log_sim_rl_norway_metro_10',
        x_axis_idx=-1,
        y_axis_idx=-2,
        x_label='Scalar Value Output',
        y_label='Reward',
        save_path='reward_scalar.png')

    plotCols('TEST_log_sim_rl_norway_metro_10',
        x_axis_idx=-1,
        y_axis_idx=0,
        x_label='Scalar Value Output',
        y_label='Bit Rate',
        save_path='bitrate_scalar.png')