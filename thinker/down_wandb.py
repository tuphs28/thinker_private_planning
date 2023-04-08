import wandb
import argparse
import os
import re

parser = argparse.ArgumentParser(description='Download artifacts from W&B run.')
parser.add_argument('--project', type=str, default='thinker', help='Name of the project.')
parser.add_argument('--run', type=str, required=True, help='ID of the run.')
parser.add_argument('--output_dir', type=str, default='/media/sc/datadisk/data/thinker/logs/', help='Output directory to store downloaded files.')

args = parser.parse_args()

# Initialize W&B run and project
wandb.init(project=args.project, id=args.run)

# Download all files from the run
run = wandb.Api().run(f"{args.project}/{args.run}")

m = re.match(r'^v\d+', args.run)
output_dir = args.output_dir
if m: output_dir = os.path.join(output_dir, m[0])
output_dir = os.path.join(output_dir, args.run)

for file in run.files():
    file.download(root=output_dir, replace=True)
    print(f"Downloaded file {file.name} to {os.path.join(output_dir, file.name)}")

print(f"File downloaded to {output_dir}")