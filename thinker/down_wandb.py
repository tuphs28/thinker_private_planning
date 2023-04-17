import wandb
import argparse
import os
import re

parser = argparse.ArgumentParser(description='Download artifacts from W&B run.')
parser.add_argument('--project', type=str, default='thinker', help='Name of the project.')
parser.add_argument('--run', type=str, required=True, help='ID of the run.')
parser.add_argument('--output_path', type=str, default='/media/sc/datadisk/data/thinker/logs/', help='Output directory to store downloaded files.')
parser.add_argument('--skip_download', action="store_true")

args = parser.parse_args()
m = re.match(r'^v\d+', args.run)
output_path = args.output_path
if m: output_path = os.path.join(output_path, m[0])
output_path = os.path.join(output_path, args.run)

if not args.skip_download:
    # Initialize W&B run and project
    wandb.init(project=args.project, id=args.run)

    # Download all files from the run
    run = wandb.Api().run(f"{args.project}/{args.run}")
    for file in run.files():
        if file.name[-3:] == "gif": continue
        file.download(root=output_path, replace=True)
        print(f"Downloaded file {file.name} to {os.path.join(output_path, file.name)}")

print(f"File downloaded to {output_path}")