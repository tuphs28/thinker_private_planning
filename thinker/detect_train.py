import os
from torch.utils.data import Dataset, DataLoader
import yaml
import argparse
import re

import math
import torch
from torch import nn
from torch.nn import functional as F
from thinker.model_net import BaseNet, FrameEncoder
from thinker.actor_net import ShallowAFrameEncoder
from thinker import util
from thinker.core.file_writer import FileWriter

class CustomDataset(Dataset):
    def __init__(self, datadir, transform=None, data_n=None, prefix="data"):
        self.datadir = datadir
        self.transform = transform
        self.data = []        
        self.samples_per_file = None   
        self.data_n = data_n
        self.prefix = prefix
        self._preload_data(datadir)  # Preload data        

    def _preload_data(self, datadir):
        # Preload all .pt files
        file_list = [f for f in os.listdir(datadir) if f.endswith('.pt') and f.startswith(self.prefix)]
        file_list = sorted(file_list, key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0)
        for file_name in file_list:
            print(f"Starting to preload {file_name}")
            xs, y = torch.load(os.path.join(datadir, file_name))
            if self.samples_per_file is None:  # Set samples_per_file based on the first file
                self.t = xs['env_state'].shape[0]
                self.b = xs['env_state'].shape[2]
                self.samples_per_file = self.t * self.b
            xs.pop('step_status')
            xs.pop('done')
            # Flatten data across t and b dimensions for easier indexing
            for t_idx in range(self.t):
                for b_idx in range(self.b):
                    flattened_xs = {k: v[t_idx, :, b_idx] for k, v in xs.items()}
                    flattened_y = y[t_idx, b_idx]
                    self.data.append((flattened_xs, flattened_y))
                    if self.data_n is not None and len(self.data) >= self.data_n: return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        xs, y = self.data[idx]
        if self.transform:
            # Apply transform if necessary. Note: You might need to adjust this part
            # based on what your transform expects and can handle
            xs = {k: self.transform(v) for k, v in xs.items()}            
        return xs, y

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:,:x.size(1)]
        return x

class DetectFrameEncoder(nn.Module):
    def __init__(
        self,
        input_shape,     
        dim_rep_actions,   
        out_size=128,
        stride=2,
    ):  
        super(DetectFrameEncoder, self).__init__()
        self.out_size = out_size
        self.encoder = FrameEncoder(prefix="se",
                                    actions_ch=dim_rep_actions,
                                    input_shape=input_shape,                             
                                    size_nn=1,             
                                    downscale_c=2,    
                                    concat_action=False)
        
        self.conv = []
        in_ch =  self.encoder.out_shape[0]
        for ch in [64]:
            self.conv.append(nn.ReLU())
            self.conv.append(nn.Conv2d(in_channels=in_ch,
                                       out_channels=ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,))
            in_ch = ch
        self.conv = nn.Sequential(*self.conv)
        conv_out_size = in_ch * self.encoder.out_shape[1] * self.encoder.out_shape[2]
        self.fc = nn.Sequential(nn.Linear(conv_out_size, self.out_size))       

    def forward(self, x, action):
        # x in shape of (B, C, H, W)
        out, _ = self.encoder(x, done=None, actions=action, state={})
        out = self.conv(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)
        return out                                
        
class DetectNet(BaseNet):
    def __init__(
        self,
        env_state_shape,
        tree_rep_shape,
        dim_actions,
        num_actions,
        detect_ab=(0,0),
        clone=False,
        tran_layer_n=3,
    ):    
        super(DetectNet, self).__init__()
        
        self.env_state_shape = env_state_shape # in (C, H, W) 
        self.tree_rep_shape = tree_rep_shape # in (C,) 
        self.dim_actions = dim_actions
        self.num_actions = num_actions
        self.dim_rep_actions = self.dim_actions if self.dim_actions > 1 else self.num_actions

        self.detect_ab = detect_ab
        self.clone = clone

        self.enc_out_size = 128
        tran_nhead = 8
        reminder = tran_nhead - ((self.enc_out_size + tree_rep_shape[0] + self.dim_rep_actions + 1) % tran_nhead)
        self.enc_out_size += reminder
        #self.true_x_encoder = ShallowAFrameEncoder(input_shape=env_state_shape, out_size=self.enc_out_size)
        #self.pred_x_encoder = ShallowAFrameEncoder(input_shape=env_state_shape, out_size=self.enc_out_size)
        self.true_x_encoder = DetectFrameEncoder(input_shape=env_state_shape, dim_rep_actions=self.dim_rep_actions, out_size=self.enc_out_size)
        self.pred_x_encoder = DetectFrameEncoder(input_shape=env_state_shape, dim_rep_actions=self.dim_rep_actions, out_size=self.enc_out_size)
        #self.pred_x_encoder = self.true_x_encoder

        self.embed_size = self.enc_out_size + tree_rep_shape[0] + num_actions + 1
        self.pos_encoder = PositionalEncoding(self.embed_size)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_size, 
                                                   nhead=tran_nhead, 
                                                   dim_feedforward=512,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, tran_layer_n)
        self.classifier = nn.Linear(self.embed_size, 1)

        self.beta = nn.Parameter(torch.tensor(0.5), requires_grad=False) # portion of negative class

    def forward(self, env_state, tree_rep, action, reset):
        """
        Forward pass of detection nn
        Args:
            env_state: float Tensor in shape of (B, rec_t, C, H, W); true and predicted frame
            tree_rep: float Tensor in shape of (B, rec_t, C); model output
            action: uint Tensor in shape of (B, rec_t, dim_actions); action (real / imaginary)
            reset: bool Tensor in shape of  (B, rec_t); reset action
        Return:
            logit: float Tensor in shape of (B); logit of classifier output
            p: float Tensor in shape of (B); prob of classifier output
        """
        B, rec_t = env_state.shape[:2]
        if self.detect_ab[0] in [1, 3] or self.detect_ab[1] in [1, 3]:
            if self.clone: env_state = env_state.clone()                
            if self.detect_ab[0] in [1, 3]:
                env_state[:, 0] = 0.
            if self.detect_ab[1] in [1, 3]:
                env_state[:, 1:] = 0.
        if self.detect_ab[0] in [2, 3] or self.detect_ab[1] in [2, 3]:
            if self.clone: tree_rep = tree_rep.clone()
            if self.detect_ab[0] in [2, 3]:
                tree_rep[:, 0] = 0.
            if self.detect_ab[1] in [2, 3]:
                tree_rep[:, 1:] = 0.
        
        action = util.encode_action(action, self.dim_actions, self.num_actions)        
        true_proc_x = self.true_x_encoder(env_state[:,0], action[:,0])
        pred_proc_x = self.pred_x_encoder(
            torch.flatten(env_state[:,1:], 0, 1),
            torch.flatten(action[:,1:], 0, 1)
                                        )
        true_proc_x = true_proc_x.view(B, self.enc_out_size).unsqueeze(1) # (B, 1, C)
        pred_proc_x = pred_proc_x.view(B, rec_t - 1, self.enc_out_size)  # (B, rec_t - 1, C)
        proc_x = torch.concat([true_proc_x, pred_proc_x], dim=1) # (B, rec_t, C)
        
        embed = [proc_x, tree_rep, action, reset.unsqueeze(-1)]
        embed = torch.concat(embed, dim=2) # (B, rec_t, embed_size)
        embed_pos = self.pos_encoder(embed)
        out = self.transformer_encoder(embed_pos)
        logit = self.classifier(out[:, -1, :]).view(B)
        return logit, torch.sigmoid(logit)

def transform_data(xs, device, flags):
    xs_ = {}

    env_state = xs["env_state"]
    if flags.rescale:
        env_state = env_state.float() / 255
    xs_["env_state"] = env_state.to(device)

    if "tree_rep" in xs: xs_["tree_rep"] = xs["tree_rep"].to(device)

    action = xs["pri_action"]
    if not flags.tuple_actions:
        action = action.unsqueeze(-1)
    xs_["action"] = action.to(device)

    if "reset_action" in xs: xs_["reset"] = xs["reset_action"].to(device)
    return xs_

def evaluate_detect(target_y, pred_y):
    # Binarize the predictions
    pred_y_binarized = (pred_y > 0.5).float()
    target_y = target_y.float()

    # Compute the accuracy
    acc = torch.mean((pred_y_binarized == target_y).float()).item()
    
    # Compute the recall
    true_positives = (pred_y_binarized * target_y).sum().float()
    possible_positives = target_y.sum().float()
    rec = (true_positives / (possible_positives + 1e-6)).item()
    
    # Compute the precision
    predicted_positives = pred_y_binarized.sum().float()
    prec = (true_positives / (predicted_positives + 1e-6)).item()
    
    # Compute the F1 score
    f1 = 2 * (prec * rec) / (prec + rec + 1e-6)   

    neg_p = 1 - torch.mean(target_y.float()).item()

    return {
        "acc": acc,
        "rec": rec,
        "prec": prec,
        "f1": f1,
        "neg_p": neg_p,
        }

def train_epoch(detect_net, dataloader, optimizer, device, flags, train=True):
    if train:
        detect_net.train()
    else:
        detect_net.eval()     
    running_train_eval = {}   
    with torch.set_grad_enabled(train):
        step = 0
        for xs, target_y in dataloader:
            xs = transform_data(xs, device, flags)
            target_y = target_y.to(device)
            
            logit, pred_y = detect_net(**xs)
            n_mean_y = torch.mean((~target_y).float()).item()
            detect_net.beta.data = 0.99 * detect_net.beta.data + (1 - 0.99) * n_mean_y
            detect_net.beta.data.clamp_(0.05, 0.95)
            weights = torch.where(target_y == 1, detect_net.beta.data, 1-detect_net.beta.data)
            loss = F.binary_cross_entropy_with_logits(logit, target_y.float(), weight=weights)
            train_eval = evaluate_detect(target_y, pred_y)
            train_eval["loss"] = loss.item()

            if train:
                optimizer.zero_grad()  # Zero the gradients
                loss.backward()  # Backward pass: compute gradient of the loss with respect to model parameters
                optimizer.step()  # Perform a single optimization step (parameter update)
            
            for key in train_eval.keys():
                if key not in running_train_eval: 
                    running_train_eval[key] = train_eval[key]
                else:
                    running_train_eval[key] += train_eval[key]
            step += 1
    return {key: val / step for (key, val) in running_train_eval.items()}

def detect_train(flags):

    if not flags.ckp:
        flags.datadir = os.path.abspath(os.path.expanduser(flags.datadir))
        # create ckp dir
        xpid_n = 0
        while (True):
            xpid_ = flags.txpid if xpid_n == 0 else flags.txpid + f"_{xpid_n}"
            ckpdir = os.path.join(flags.datadir, xpid_)
            xpid_n += 1
            if not os.path.exists(ckpdir):
                os.mkdir(ckpdir) 
                flags.txpid = xpid_
                break    
    else:
        ckpdir = os.path.join(flags.datadir, flags.txpid)
    flags.tckpdir = ckpdir
    flags.tckp_path = os.path.join(ckpdir, "ckp_detect.tar")
    print(f"Checkpoint path: {flags.tckp_path}")

    # load data
    dataset = CustomDataset(datadir=flags.datadir, transform=None, data_n=flags.data_n)
    dataloader = DataLoader(dataset, batch_size=flags.batch_size, shuffle=True)

    val_dataset = CustomDataset(datadir=flags.datadir, transform=None, data_n=2000, prefix="val")
    val_dataloader = DataLoader(val_dataset, batch_size=flags.batch_size, shuffle=True)

    # load setting
    yaml_file_path = os.path.join(flags.datadir, 'config_detect.yaml')
    with open(yaml_file_path, 'r') as file:
        flags_data = yaml.safe_load(file)
    flags_data = argparse.Namespace(**flags_data)
    flags = argparse.Namespace(**{**vars(flags), **vars(flags_data)}) # merge the two flags
    flags.full_xpid = flags.dxpid + "_" + flags.txpid

    plogger = FileWriter(
        xpid=flags.txpid,
        xp_args=flags.__dict__,
        rootdir=flags.datadir,
        overwrite=not flags.ckp,
    )

    if flags.use_wandb: wlogger = util.Wandb(flags)

    # initalize net
    device = torch.device("cuda")
    detect_net = DetectNet(
        env_state_shape = flags_data.env_state_shape,
        tree_rep_shape = flags_data.tree_rep_shape,
        dim_actions = flags_data.dim_actions,
        num_actions = flags_data.num_actions,
    )

    # load optimizer
    optimizer = torch.optim.Adam(
        detect_net.parameters(), lr=flags.learning_rate, 
    )

    if flags.ckp:
        checkpoint = torch.load(flags.ckp_path, torch.device("cpu"))
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        detect_net.load_state_dict(checkpoint["net_state_dict"])
        epoch = checkpoint["epoch"]
        del checkpoint
    else:
        epoch = 0

    detect_net = detect_net.to(device)
    util.optimizer_to(optimizer, device)

    print("Detect network size: %d"
            % sum(p.numel() for p in detect_net.parameters())
        )

    while (epoch < flags.num_epochs):
        train_stat = train_epoch(detect_net, dataloader, optimizer, device, flags, train=True)
        val_stat = train_epoch(detect_net, val_dataloader, None, device, flags, train=False)
        stat = {**train_stat, **{'val/' + key: value for key, value in val_stat.items()}}
        stat["epoch"] = epoch
        plogger.log(stat)
        if flags.use_wandb: wlogger.wandb.log(stat, step=stat['epoch'])

        print_str = f'Epoch {epoch+1}/{flags.num_epochs},'
        for key in stat.keys(): 
            if 'val/' + key in stat.keys():
                print_str += f" {key}:{stat[key]:.4f} ({stat['val/'+key]:.4f})"
        print(print_str)    

        epoch += 1    
        if epoch % 5 == 0 or epoch >= flags.num_epochs:
            # save checkpoint
            d = {
                "epoch": epoch,
                "flags": flags,
                "optimizer_state_dict": optimizer.state_dict(),
                "net_state_dict": detect_net.state_dict(),
            }
            torch.save(d, flags.ckp_path)
            print(f"Checkpoint saved to {flags.ckp_path}")

        if epoch % 10 == 0 or epoch >= flags.num_epochs:
            wlogger.wandb.save(
                os.path.join(flags.tckpdir, "*"), flags.tckpdir
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Thinker detection network training")
    parser.add_argument("--dxpid", default="", help="Data file name")
    parser.add_argument("--dproject", default="detect", help="Data project name.")
    parser.add_argument("--datadir", default="../data/__dproject__/__dxpid__/", help="Data directory.")    
    parser.add_argument("--txpid", default="test", help="training xpid of the run.")  
    parser.add_argument("--project", default="detect_post", help="Project of the run.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size in training.")
    parser.add_argument("--learning_rate", default=0.0001, type=float, help="Learning rate.")
    parser.add_argument("--num_epochs", default=50, type=int, help="Number of epoch.")
    parser.add_argument("--data_n", default=50000, type=int, help="Training data size.")
    parser.add_argument("--ckp", action="store_true", help="Enable loading from checkpoint.")
    parser.add_argument("--use_wandb", action="store_true", help="Enable logging to wandb.")

    flags = parser.parse_args()    
    flags.datadir = flags.datadir.replace("__dproject__", flags.dproject)
    flags.datadir = flags.datadir.replace("__dxpid__", flags.dxpid)

    detect_train(flags)