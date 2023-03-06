import thinker.util as util
import os
import csv
import numpy as np
import torch
import re

def eval_(x):
    if "tensor" in x: 
        match = re.search(r"\d+\.\d+", x)
        return float(match.group()) if match else 0.
    if x == "inf": return np.inf
    if x=="": return 0
    return eval(x)

if __name__ == "__main__":
    flags = util.parse()
    
    wlogger = util.Wandb(flags, subname="_model")
    filename = os.path.join(flags.load_checkpoint, "logs_model.csv")
    with open(filename, 'r') as file:
        csvreader = csv.reader((line.replace('\0','') for line in file))   
        keys = next(csvreader)
        for m, row in enumerate(csvreader):            
            if m % 50 == 0:
                stats = {keys[n]: eval_(i) for n, i in enumerate(row)}
                wlogger.wandb.log(stats, step=int(stats['real_step']))
    print("finished uploading model logs")

    wlogger = util.Wandb(flags, subname="")
    filename = os.path.join(flags.load_checkpoint, "logs.csv")
    with open(filename, 'r') as file:
        csvreader = csv.reader((line.replace('\0','') for line in file))   
        keys = next(csvreader)
        for m, row in enumerate(csvreader):            
            if m % 50 == 0:
                stats = {keys[n]: eval_(i) for n, i in enumerate(row)}
                wlogger.wandb.log(stats, step=int(stats['real_step']))
    print("finished uploading actor logs")