import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
import time

from tensorboardX import SummaryWriter

from datasets import CPDataset, CPDataLoader
from models.gmm import GMM
from models.vgg import VGGLoss
from models.unet import UnetGenerator
from utilities import load_checkpoint, save_checkpoint
from visualization import board_add_images

import matplotlib.pyplot as plt
import numpy as np


def visualize_grid_sampling(input_images, warped_cloth_images, grids):
    batch_size, _, _, _ = input_images.shape
    plt.figure(figsize=(15, 8))

    for i in range(batch_size):
        # Original input image
        input_img = (input_images[i].cpu().detach().numpy() * 0.5) + 0.5  # De-normalize
        plt.subplot(3, batch_size, i + 1)
        plt.imshow(input_img.transpose(1, 2, 0))
        plt.axis("off")
        plt.title("Input Image")

        # Warped cloth image
        warped_cloth_img = (warped_cloth_images[i].cpu().detach().numpy() * 0.5) + 0.5  # De-normalize
        plt.subplot(3, batch_size, i + 1 + batch_size)
        plt.imshow(warped_cloth_img.transpose(1, 2, 0))
        plt.axis("off")
        plt.title("Warped Cloth")

        # Corresponding grid visualization
        grid = grids[i].cpu().detach().numpy()
        plt.subplot(3, batch_size, i + 1 + 2 * batch_size)
        plt.imshow(np.linalg.norm(grid, axis=2), cmap="viridis", interpolation="none")
        plt.colorbar()
        plt.axis("off")
        plt.title("Grid")

    plt.show()


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "GMM")
    parser.add_argument("--gpu_ids", default = "")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    
    parser.add_argument("--dataroot", default = "data")
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = "GMM")
    parser.add_argument("--data_list", default = "train_pairs.txt")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default = 20)
    parser.add_argument("--save_count", type=int, default = 100)
    parser.add_argument("--keep_step", type=int, default = 100_000)
    parser.add_argument("--decay_step", type=int, default = 100_000)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument("--use_cuda", action=argparse.BooleanOptionalAction, default = False)

    opt = parser.parse_args()
    return opt


def train_gmm(opt, train_loader, model, board):
    if opt.use_cuda:
        model.cuda()
    model.train()

    # criterion
    criterionL1 = nn.L1Loss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    # scheduler = torch.optim.\
    #             lr_scheduler.\
    #             LambdaLR(optimizer, lr_lambda= lambda step: 1.0 - max(0, step - opt.keep_step) / float(opt.decay_step + 1))

    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        if opt.use_cuda:
            im          = inputs['image'].cuda()
            im_pose     = inputs['pose_image'].cuda()
            im_h        = inputs['head'].cuda()
            shape       = inputs['shape'].cuda()
            agnostic    = inputs['agnostic'].cuda()
            c           = inputs['cloth'].cuda()
            im_c        = inputs['parse_cloth'].cuda()
            im_g        = inputs['grid_image'].cuda()
        else:
            im          = inputs['image']
            im_pose     = inputs['pose_image']
            im_h        = inputs['head']
            shape       = inputs['shape']
            agnostic    = inputs['agnostic']
            c           = inputs['cloth']
            im_c        = inputs['parse_cloth']
            im_g        = inputs['grid_image']

        # grid, theta  = model(agnostic, c)
        grid, _  = model(agnostic, c)
        # warped_mask  = F.grid_sample(cm, grid, padding_mode='zeros', align_corners=False)
        warped_cloth = F.grid_sample(c, grid, padding_mode='border', align_corners=False)
        warped_grid  = F.grid_sample(im_g, grid, padding_mode='zeros', align_corners=False)
        # visualize_grid_sampling(im, warped_cloth, grid)

        visuals = [
            [im_h, shape, im_pose],
            [c, warped_cloth, im_c],
            [warped_grid, (warped_cloth + im) * 0.5, im]
        ]

        loss = criterionL1(warped_cloth, im_c)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step + 1)
            board.add_scalar('metric', loss.item(), step + 1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %4f' % (step + 1, t, loss.item()), flush=True)

        if (step + 1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, f'step_{step + 1}.pth'), opt.use_cuda)


def train_tom(opt, train_loader, model, board):
    if opt.use_cuda:
        model.cuda()
    model.train()

    # criterion
    criterionL1 = nn.L1Loss()
    criterionVGG = VGGLoss(use_cuda=opt.use_cuda)
    criterionMask = nn.L1Loss()

    # optimzer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    for step in range(opt.keep_step + opt.decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        im_pose = inputs['pose_image']
        im_h = inputs['head']
        shape = inputs['shape']
        print("DEBUG: Type of inputs:", type(inputs)) 

        if opt.use_cuda:
            im = inputs['image'].cuda()
            agnostic = inputs['agnostic'].cuda()
            c = inputs['cloth'].cuda()
            cm = inputs['cloth_mask'].cuda()
        else:
            im = inputs['image']
            agnostic = inputs['agnostic']
            c = inputs['cloth']
            cm = inputs['cloth_mask']

        outputs = model(torch.cat([agnostic, c], 1))
        p_rendered, m_composite = torch.split(outputs, 3, 1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        p_tryon = c * m_composite + p_rendered * (1 - m_composite)

        visuals = [
            [im_h, shape, im_pose],
            [c, cm * 2 - 1, m_composite * 2 - 1],
            [p_rendered, p_tryon, im]
        ]

        loss_l1 = criterionL1(p_tryon, im)
        loss_vgg = criterionVGG(p_tryon, im)
        loss_mask = criterionMask(m_composite, cm)
        loss = loss_l1 + loss_vgg + loss_mask
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % opt.display_count == 0:
            board_add_images(board, 'combine', visuals, step + 1)
            board.add_scalar('metric', loss.item(), step + 1)
            board.add_scalar('L1', loss_l1.item(), step + 1)
            board.add_scalar('VGG', loss_vgg.item(), step + 1)
            board.add_scalar('MaskL1', loss_mask.item(), step + 1)

            t = time.time() - iter_start_time
            print(
                'step: %8d, time: %.3f, loss: %.4f, l1: %.4f, vgg: %.4f, mask: %.4f'
                % (
                    step + 1,
                    t,
                    loss.item(),
                    loss_l1.item(),
                    loss_vgg.item(),
                    loss_mask.item(),
                ),
                flush=True
            )

        if (step + 1) % opt.save_count == 0:
            save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'step_%06d.pth' % (step + 1)), opt.use_cuda)


def main():
    opt = get_opt()
    print(opt)
    print('Start to train stage: %s, name: %s!' % (opt.stage, opt.name))

    # create dataset
    train_dataset = CPDataset(opt)

    # create dataloader
    train_loader = CPDataLoader(opt, train_dataset)

    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)

    board = SummaryWriter(log_dir = os.path.join(opt.tensorboard_dir, opt.name))

    # create model, train and save the final checkpoint
    if opt.stage == 'GMM':
        model = GMM(opt)

        if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint, opt.use_cuda)

        start_time = time.time()
        train_gmm(opt, train_loader, model, board)
        end_time = time.time()
        print(f'GMM training took {(end_time - start_time) / 60} minutes')
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'gmm_final.pth'), opt.use_cuda)

    elif opt.stage == 'TOM':
        model = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)

        if not opt.checkpoint == '' and os.path.exists(opt.checkpoint):
            load_checkpoint(model, opt.checkpoint, opt.use_cuda)

        start_time = time.time()
        train_tom(opt, train_loader, model, board)
        end_time = time.time()
        print(f'TOM training took {(end_time - start_time) / 60} minutes')
        save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'tom_final.pth'), opt.use_cuda)

    else:
        raise NotImplementedError(f'Model [{opt.stage}] is not implemented')

    print(f'Finished training {opt.stage}, named: {opt.name}!')


if __name__ =='__main__':
    main()

