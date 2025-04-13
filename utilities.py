import torch
import os

def save_checkpoint(model, save_path, cuda):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    torch.save(model.cpu().state_dict(), save_path)
    if cuda:
        model.cuda()

def load_checkpoint(model, checkpoint_path, cuda):
    if not os.path.exists(checkpoint_path):
        return

    model.load_state_dict(torch.load(checkpoint_path))
    if cuda:
        model.cuda()

