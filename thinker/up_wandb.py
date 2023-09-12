import os
import re
import argparse
import thinker.util as util
import time

def parse_line(header, line):
    if header is None or line is None: return None
    data = re.split(r',(?![^\(]*\))', line.strip())
    data_dict = {}    
    if len(header) != len(data): 
        print(f"Header size and data size mismatch")
        return None
    for n, (key, value) in enumerate(zip(header, data)):
        try:
            if not value:                     
                value = None
            else:
                value = eval(value)
            if type(value) == str: value = eval(value)
        except (SyntaxError, NameError, TypeError) as e:
            print(f"Cannot read value {value} for key {key}: {e}")
            return None
        data_dict[key] = value
        if n == 0: data_dict['_tick'] = value # assume first column is the tick
    return data_dict

def read_file(file, freq, start_step=0):    
    stats = []
    with open(file, 'r') as f:
        fields_ = f.readline()
        fields = fields_.strip().split(',')
        cur_real_step = 0
        while(True):
            line = f.readline()
            if not line: break
            out = parse_line(fields, line)   
            if out['real_step'] > cur_real_step:
                if out['real_step'] > start_step: stats.append(out)
                cur_real_step += freq
    return stats

def merge_stat(stats_a, stats_b):
    l = min(len(stats_a), len(stats_b))
    stats = []
    for n in range(l):
        stat = stats_b[n]
        stat.update(stats_a[n])
        stats.append(stat)
    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Upload file to wandb")
    parser.add_argument("--load_checkpoint", default="",
                        help="Load checkpoint directory.")   
    parser.add_argument("--start_step", default=0, type=int, help="Step begins to be uploaded.")
    flags = parser.parse_args() 
    flags_ = util.parse(["--load_checkpoint", flags.load_checkpoint])    

    path = os.path.abspath(os.path.expanduser(flags.load_checkpoint))
    if os.path.islink(path):
        path = os.readlink(path)            

    actor_log_path = os.path.join(path, 'logs.csv')
    model_log_path = os.path.join(path, 'logs_model.csv')
    upload_freq = 10000
    start_step = flags.start_step

    stats = read_file(actor_log_path, upload_freq, start_step)
    if not flags_.disable_model:
        stats_m = read_file(model_log_path, upload_freq, start_step)
        stats = merge_stat(stats, stats_m)
    
    print("Uploading data with size %d for %s..." % (len(stats), flags_.xpid))
    print(f"Example of data uploaded {stats[-1]}")

    wlogger = util.Wandb(flags_)    
    for n, stat in enumerate(stats):        
        wlogger.wandb.log(stat, step=stat['real_step'])
        time.sleep(0.1)
        if n % 50 == 0: print("Uploading step: %d" % stat['real_step'])
    wlogger.wandb.finish(exit_code=0)