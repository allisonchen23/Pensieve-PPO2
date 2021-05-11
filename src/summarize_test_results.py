import pandas as pd
import os
from sklearn.metrics import mean_squared_error

ORIGINAL_RESULTS_DIR = './original_test_results_05_09_2021'
REPEAT_DENSE_RESULTS_DIR = './repeat_dense_test_results_05_08_2021'
COLUMN_NAMES = [
    'timestamp',
    'bit_rate_prediction',
    'buffer_size',
    'rebuffer_time',
    'video_chunk_size',
    'delay',
    'reward']

def load_results_as_df(filepath):
    '''
    Arg(s):
        filepath : str
            path to the test results file (7 column, tab separated)
    Returns:
        df : pandas.Dataframe
    '''
    df = pd.read_csv(
        filepath,
        sep='\t',
        header=None)
    df.columns = COLUMN_NAMES
    return df

def compare_result_log(filepath1, filepath2, compare_cols=[]):
    '''
    Compare the columns in compare_cols of filepath1 and filepath2
    Returns: dict
        indexed by column name; each element contains:
        {
            mean_difference : float
            mse: float
            n_diff : int
                number of differing rows
            diffs : list[float]
                the difference between the two columns

        }
    '''
    df1 = load_results_as_df(filepath1)
    df2 = load_results_as_df(filepath2)
    summary_diff = {}
    for col_name in compare_cols:
        summary = {}
        col1= df1[col_name]
        col2 = df2[col_name]
        mse = mean_squared_error(col1, col2)

        diff = col1 - col2
        mean_diff = diff.mean()
        n_diff = diff.astype(bool).sum()
        summary['mean_difference'] = mean_diff
        summary['mse'] = mse
        summary['n_diff'] = n_diff
        # summary['diffs'] = list(diff)

        summary_diff[col_name] = summary
    return summary_diff

def pretty_print(summary):
    str = ""
    for col_header in summary.keys():
        str += "{}:\n".format(col_header)
        str +="\t{}\n".format(summary[col_header])
    return str

def compare_directories(base_dirpath,
                        test_dirpath,
                        log_path='diff.log',
                        compare_cols=[]):
    '''
    Compare the results files from the two directories.
    Extract filenames from base_dirpath
    '''
    # Create log dir
    log_dirname = os.path.dirname(log_path)
    if log_dirname is not "" and not os.path.isdir(log_dirname):
        os.makedirs(log_dirname)

    # Check valid directories
    if not os.path.isdir(base_dirpath) or not os.path.isdir(test_dirpath):
        print("Invalid directory")
        return None
    with open(log_path, 'w') as log_file:
        for result_file in os.listdir(base_dirpath):
            base_filepath = os.path.join(base_dirpath, result_file)
            test_filepath = os.path.join(test_dirpath, result_file)
            summary = compare_result_log(
                base_filepath,
                test_filepath,
                compare_cols=compare_cols
            )
            log_file.write('Summary of {}:\n{}\n'.format(result_file, pretty_print(summary)))


if __name__ == "__main__":
    filepath1 = os.path.join(ORIGINAL_RESULTS_DIR, 'log_sim_rl_norway_bus_1')
    filepath2 = os.path.join(REPEAT_DENSE_RESULTS_DIR, 'log_sim_rl_norway_bus_1')
    compare_result_log(
        filepath1,
        filepath2,
        compare_cols=['bit_rate_prediction', 'reward'])
    compare_directories(
        ORIGINAL_RESULTS_DIR,
        REPEAT_DENSE_RESULTS_DIR,
        log_path='test_diff.log',
        compare_cols=['bit_rate_prediction', 'reward'])
