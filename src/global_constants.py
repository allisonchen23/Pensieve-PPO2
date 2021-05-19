import os

FEATURE_NUM                     = 64 #128
N_DENSE_LAYERS                  = 3
MODEL_ARCH                      = 'ffd_{}_64'.format(N_DENSE_LAYERS)
RESULTS_DIR                     = os.path.join('results', MODEL_ARCH)
TEST_LOG_FOLDER                 = os.path.join(RESULTS_DIR, '{}_test_results'.format(MODEL_ARCH))
TEST_LOG_FILE                   = os.path.join(TEST_LOG_FOLDER, 'log_sim_rl')
LOG_FILE                        = os.path.join(RESULTS_DIR, '{}_log'.format(MODEL_ARCH))
SUMMARY_DIR                     = os.path.join(RESULTS_DIR, 'summary')