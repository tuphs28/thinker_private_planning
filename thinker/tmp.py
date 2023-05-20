import torch

actor = torch.load("/home/scuk/logs/tmp/v11_defender_0.7b6/ckp_actor.tar")
print(actor["real_step"])
model = torch.load("/home/scuk/logs/tmp/v11_defender_0.7b6/ckp_model.tar")
print(model["real_step"])
