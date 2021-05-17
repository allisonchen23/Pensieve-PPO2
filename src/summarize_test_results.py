import pandas as pd
import os
from sklearn.metrics import mean_squared_error
import numpy as np

ORIGINAL_RESULTS_DIR = './original_test_results_05_09_2021'
REPEAT_DENSE_RESULTS_DIR = './repeat_dense_test_results_05_08_2021'
NORWAY_RESULTS_DIR = './norway'
KEEP_SCALAR_RESULTS_DIR = './keep_scalar_test_results_05_10_2021'
SUMMARY_DIR = './diff_summaries'
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
    Returns: dict indexed by column name
        value: np.array of [mean_difference, mse, n_diff]
    dict
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
        col1= df1[col_name]
        col2 = df2[col_name]
        mse = mean_squared_error(col1, col2)

        diff = col1 - col2
        mean_diff = diff.mean()
        n_diff = diff.astype(bool).sum()
        # summary['mean_difference'] = mean_diff
        # summary['mse'] = mse
        # summary['n_diff'] = n_diff
        summary = np.array([mean_diff, mse, n_diff])
        summary_diff[col_name] = summary
    return summary_diff

def pretty_print(summary, col_names=[]):
    str = ""
    for col_header in summary.keys():
        str += "{}:\n".format(col_header)
        for i in range(len(col_names)):
            str +="\t{}: {}".format(col_names[i], summary[col_header][i])
        str += "\n"
    return str

def compare_directories(base_dirpath,
                        test_dirpath,
                        log_path='diff.log',
                        compare_cols=[]):
    '''
    Compare the results files from the two directories.
    Extract filenames from base_dirpath
    '''
    metric_names = ['mean_diff', 'mse', 'n_diff']
    # Create log dir
    log_dirname = os.path.dirname(log_path)
    if log_dirname is not "" and not os.path.isdir(log_dirname):
        os.makedirs(log_dirname)

    # Check valid directories
    if not os.path.isdir(base_dirpath) or not os.path.isdir(test_dirpath):
        print("Invalid directory")
        return None

    diff_summaries = []
    with open(log_path, 'w') as log_file:
        for result_file in os.listdir(base_dirpath):
            file_summary = []
            base_filepath = os.path.join(base_dirpath, result_file)
            if "rl" not in result_file:
                continue

            test_filepath = os.path.join(test_dirpath, result_file)
            summary = compare_result_log(
                base_filepath,
                test_filepath,
                compare_cols=compare_cols
            )
            for col in compare_cols:
                file_summary.append(summary[col])
            diff_summaries.append(file_summary)
            log_file.write('Summary of {}:\n{}\n'.format(
                result_file,
                pretty_print(summary, col_names=metric_names)))
        diff_summaries = np.array(diff_summaries)
        means = np.mean(diff_summaries, axis=0)
        maxs = np.amax(diff_summaries, axis=0)
        mins = np.amin(diff_summaries, axis=0)
        medians = np.median(diff_summaries, axis=0)

        summary_str = "---***---\nOverall Summary:"
        for stat, stat_name in zip([means, maxs, mins, medians], ["mean", "max", "min", "median"]):
            summary_str += "\n---{}---".format(stat_name)
            for col_idx, col in enumerate(compare_cols):
                summary_str += "\n{}:\n\t".format(col)
                for metric_idx in range(len(metric_names)):
                    summary_str += "{}: {}\t".format(metric_names[metric_idx], stat[col_idx, metric_idx])
        log_file.write(summary_str)


if __name__ == "__main__":
    BASE_DIR = './ffd_1_test_results'
    COMP_DIR = './ffd_2_test_results'
    compare_directories(
        BASE_DIR,
        COMP_DIR,
        log_path=os.path.join(SUMMARY_DIR, 'ffd_1_ffd_2.log'),
        compare_cols=['bit_rate_prediction', 'reward'])
