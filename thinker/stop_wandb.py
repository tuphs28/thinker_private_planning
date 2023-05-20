import argparse
import os
import wandb


def parse():
    parser = argparse.ArgumentParser(description="Stop wandb run")
    parser.add_argument("--xpid", default=None, help="Experiment id (default: None).")
    parser.add_argument("--exit", type=int, default=1, help="exit code")
    return parser.parse_args()


if __name__ == "__main__":
    flags = parse()
    subnames = ["", "_model"]
    for subname in subnames:
        exp_name = flags.xpid + subname
        wlogger = wandb.wandb.init(
            project="thinker",
            entity=os.getenv("WANDB_USER", "stephen-chung"),
            reinit=True,
            resume="allow",
            id=exp_name,
            name=exp_name,
        )
        wlogger.finish(exit_code=flags.exit)
