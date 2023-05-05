import os
import re
import argparse
import thinker.util as util

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Upload file to wandb")
    parser.add_argument("--load_checkpoint", default="",
                        help="Load checkpoint directory.")   
    flags = parser.parse_args() 
    flags_ = util.parse(["--load_checkpoint", flags.load_checkpoint])

    path = os.path.abspath(os.path.expanduser(flags.load_checkpoint))
    if os.path.islink(path):
        path = os.readlink(path)            

    actor_log_path = os.path.join(path, 'logs.csv')
    model_log_path = os.path.join(path, 'logs_model.csv')
    upload_freq = 50000

    stats_a = []
    with open(actor_log_path, 'r') as f:
        fields_ = f.readline()
        fields = fields_.strip().split(',')
        cur_real_step = 0
        while(True):
            line = f.readline()
            if not line: break
            out = parse_line(fields, line)   
            if out['real_step'] > cur_real_step:
                stats_a.append(out)
                cur_real_step += upload_freq

    if not flags_.disable_model and flags._train_model:
        stats_m = []
        with open(model_log_path, 'r') as f:
            fields_ = f.readline()
            fields = fields_.strip().split(',')
            cur_real_step = 0
            while(True):
                line = f.readline()
                if not line: break
                out = parse_line(fields, line)
                if out['real_step'] > cur_real_step:
                    stats_m.append(out)
                    cur_real_step += upload_freq      
        l = min(len(stats_a), len(stats_m))
        stats_a = stats_a[:l]
        stats_m = stats_m[:l]
        stats = []
        for n in range(l):
            stat = stats_m[n]
            stat.update(stats_a[n])
            stats.append(stat)
            print(stat)    
    else:
        stats = stats_a

    print("Uploading data with size %d for %s..." % (len(stats), flags_.xpid))
    print(f"Example of data uploaded {stats[-1]}")

    wlogger = util.Wandb(flags_)
    for stat in stats:
        wlogger.wandb.log(stat, step=stat['real_step'])
    wlogger.wandb.finish(exit_code=0)
