""" Global constants used in training """

NUM_EPISODES = 1000
BATCH_SIZE = 32
GAMMA = 0.99

LR = 1e-3

TARGET_UPDATE_FREQ = 10

REPLAY_BUFFER_SIZE = 10000

SAVE_PATH = "weights/dqn_final_weights.pt"
