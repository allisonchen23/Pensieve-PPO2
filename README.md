# Pensieve PPO2
This is an easy tensorflow implementation of Pensieve[1]. 
In details, we trained Pensieve via PPO2 rather than A3C.
It's a stable version, which has already prepared the training set and the test set, and you can run the repo easily: just type

```
python train.py
```

instead. Results will be evaluated on the test set (from HSDPA) every 300 epochs.

Please feel free to let me know if you have any questions.

[1] Mao H, Netravali R, Alizadeh M. Neural adaptive video streaming with pensieve[C]//Proceedings of the Conference of the ACM Special Interest Group on Data Communication. ACM, 2017: 197-210.

## Additions for COMET Compatability

### Model Architecture

Currently, the PPO2 code obtains the network architecture from `src/ppo2_feed_forward_dense.py` instead of `src/a2c.py`

This architecture differs from the original in that
* it takes in all inputs as a 1-d 25-length vector
* it solely consists of fully connected layers
* the user can specify how many additional fully connected layers (on top of the base layers) are added. The base is 2 FC layers after the input, then the outputs

In order to change the file from which the model architecture is imported from, you need to edit the following:
* In `src/rl_test.py` change `import ppo2_feed_forward_dense as network` (~line 7) to `import <your_filename> as network`
* In `src/train.py` change `import ppo2_feed_forward_dense as network` (~line 7) to `import <your_filename> as network`

Although the models are different, the interface with which the `train.py` and `rl_test.py` files interact with it should be uniform. User parameters are configured via the `src/global_constants.py` file.

#### Modifying Architecture

For COMET compatability, we wanted the model's architecture to be flexible, allowing us to try different size of networks. To configure your network, edit the following values in `src/global_constants.py`.
* `FEATURE_NUM`: the number of neurons in each fully connected layer
* `N_DENSE_LAYERS`: the number of fully connected layers *on top of the base 3*

The remaining values need not be changed as they create directories and names for saving the model.

### Training

To train a modified PPO2 model, follow these steps:
1. In `src/global_constants.py` edit the `FEATURE_NUM` and `N_DENSE_LAYERS` values as specified in the section above.
2. Ensure there is not a directory of the same name as `RESULTS_DIR`. The contents will get overwritten.
3. From the base of the repository, run `cd src` and then `python train.py`

When the model is done training, change the name of the results directory to include the date in this format: `<current dirname>_MMDDYYYY`. For example, `ffd_0_10` becomes `ffd_0_10_06022021`. 
* The first number (0) represents the number of additional dense layers
* The second number (10) represents the number of neurons/layer
* The last is the date.

Keeping this format will simplify steps to load the model.

### Converting .ckpt model to Keras

Once the model has finished training, we need to find the best checkpoint and convert it to be COMET compatible. This means the model needs to be saved as .csv and .h5 files and the output needs to be a scalar values. These requirements are implemented in `src/ppo2_keras.py` in the model architecture.

#### Determine the best checkpoint.

Under the `results` directory where the model was saved to, there should be a text file titled `<model architecture>_log_test.txt`. This file contains the statistics of the reward after each test run. The columns are
* Epoch
* Min Reward
* 5th Percentile 
* Mean
* Median
* 95th Percentile
* Max

Depending on your metric, select the best step from this sheet.

#### Run `src/load_keras.py`

`src/load_keras.py` will take in the path of a .ckpt checkpoint, and output the corresponding .h5 and .csv files that represent the same model with one caveat:

There are 2 outputs with PPO2: a vector used to predict bit rate and a scalar used to compute loss. Currently, the keras version of PPO2 tosses a way the vector and only utilizes the scalar output.

To convert the .ckpt checkpoint, follow these steps:
1. Open the file `src/load_keras.py`
2. Scroll to the section under `if __name__=="__main__":` 
3. Assuming you did not change anything with the `global_constants.py` configuration file, make the following changes:
* Edit the `step` variable to be the checkpoint number you want to save. This should be an integer.
* Edit the `date` variable to be the date in the format of `MMDDYYYY` in the directory name that you added. This should be a string
* If you **did** customize your own path names, simply edit the parameter `ckpt_path` in line 295 of the `load_model_to_keras` function call to be the path to your checkpoint.
4. (Opt) edit the `csv_save_dir` and `h5_save_path` parameters in the `load_model_to_keras` function call to be the paths you would like them to be saved to. The default options should work fine.

#### Some Implementation Details

This code works on a high level by the following steps:
1. Obtain the names of the layers for the .ckpt file and for the keras model in the `get_fc_layer_names()` function. These are based off the default values that tflearn and tensorflow use. This also throws out the name of the layer for the vector output, ensuring that layer does not get converted
2. The function `load_model_to_keras()` in `load_keras.py` initializes a new actor model, then restores the weights based on the .ckpt file. Each layer saves the weights and biases into .csv files and the overall model weights are saved into a .h5 file in the paths specified.

### Running COMET

#### Obtain Data

The input data for COMET has to be a csv where each line represents the data value for each input. In order to save an example of the data used in training, in the file `global_constants.py` find the constant `DUMP_INPUT_DATA` and set it to `True`. You can additionally edit the variable `SAVE_INPUT_DATA_INTERVAL` to be how often you want to save the training data. Setting `DUMP_INPUT_DATA` will also save the test data in the file `rl_test.py`. To configure the paths to directories for dumping the data, edit the constants in `global_constants.py` `TRAIN_DATA_DUMP_DIR` and `TEST_DATA_DUMP_DIR`.

Saving the test data looks a little differently.
* get training data
* copy over files (h5, data, csvs)
* here we refer to comet documentation
* run verifier
* run envelope


### Next Steps
* Understand how our architecture changes to PPO2 affect our performance compared to the original model. We do not want it to be too off.
* Edit the Keras model such that it would output a scalar that is the predicted bit rate. 