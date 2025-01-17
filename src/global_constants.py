import os

FEATURE_NUM                     = 10
N_DENSE_LAYERS                  = 0
MODEL_ARCH                      = 'ffd_{}_{}'.format(N_DENSE_LAYERS, FEATURE_NUM)
RESULTS_DIR                     = os.path.join('results', MODEL_ARCH)
TEST_LOG_FOLDER                 = os.path.join(RESULTS_DIR, '{}_test_results'.format(MODEL_ARCH))
TEST_LOG_FILE                   = os.path.join(TEST_LOG_FOLDER, 'log_sim_rl')
LOG_FILE                        = os.path.join(RESULTS_DIR, '{}_log'.format(MODEL_ARCH))
SUMMARY_DIR                     = os.path.join(RESULTS_DIR, 'summary')
DUMP_INPUT_DATA                 = True
DATA_DUMP_DIR                   = os.path.join('results', 'flattened_inputs')
TRAIN_DATA_DUMP_DIR             = os.path.join(DATA_DUMP_DIR, 'train')
TEST_DATA_DUMP_DIR             = os.path.join(DATA_DUMP_DIR, 'test')
SAVE_INPUT_DATA_INTERVAL        = 5000