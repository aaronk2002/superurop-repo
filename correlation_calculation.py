import torch
import numpy as np
import argparse
from tools.generate_data import data_generator
from tools.train import full_train

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--att_svm", help="result file to get att-svm solution", type=str)
parser.add_argument("--Ws", help="result file to get Ws history", type=str)
parser.add_argument("--output", help="output destination", type=str)
args = parser.parse_args()

# Extract data
Ws = torch.load(args.Ws)["Ws"]
att_svm = torch.load(args.att_svm)["att-svm"]
epochs = len(Ws)

# Get correlation
W_corrs = np.zeros((epochs,))
for it in range(epochs):
    W = Ws[it] / np.linalg.norm(Ws[it])
    W_corrs[it] = att_svm.reshape(-1).dot(W.reshape(-1))

# Save
torch.save(W_corrs, args.output)
