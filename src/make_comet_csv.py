import pandas as pd
import numpy as np
import os

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
        print(values)

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
    df = pd.Dataframe(np_data)

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

def make_comet_compatible(read_path,
                          save_path,
                          gt_name,
                          headers):
    '''
    Read in CSV from read_path, make following changes:
        1. Add headers to existing columns
        2. Add GT column
    Save new CSV to save_path
    '''
    # Create output directory if it doesn't exist
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    df = pd.read_csv(read_path)

    add_headers(
        df=df,
        headers=headers
    )

    add_col(
        df=df,
        col_name=gt_name,
        index=0,
        random=True,
        min_val=df['prev_bit_rate'].min(),
        max_val=df['prev_bit_rate'].max(),
        values=None
    )

    df.to_csv(save_path)

if __name__ == "__main__":
    read_path = 'results/flattened_inputs/test/test_data_norway_bus_1.csv'
    save_path = 'results/flattened_inputs/comet_compat/test/test_data_norway_bus_1.csv'
    headers = [
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
    make_comet_compatible(
        read_path=read_path,
        save_path=save_path,
        gt_name='bit_rate',
        headers=headers
    )
