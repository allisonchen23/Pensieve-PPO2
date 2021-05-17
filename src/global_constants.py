import os

N_DENSE_LAYERS                  = 1
MODEL_ARCH                      = 'ffd_{}'.format(N_DENSE_LAYERS)
RESULTS_DIR                     = os.path.join('results', MODEL_ARCH)
TEST_LOG_FOLDER                 = os.path.join(RESULTS_DIR, '{}_test_results'.format(MODEL_ARCH))
TEST_LOG_FILE                   = os.path.join(TEST_LOG_FOLDER, 'log_sim_rl')
LOG_FILE                        = os.path.join(RESULTS_DIR, '{}_log'.format(MODEL_ARCH))
SUMMARY_DIR                     = os.path.join(RESULTS_DIR, 'summary')