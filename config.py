import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

SAMPLE_RATE = 22050
N_MFCC = 52

HF_REPO_ID = "Jojo1088/heart-murmur"
HF_MODEL_FILENAME = "lstm_model.keras"