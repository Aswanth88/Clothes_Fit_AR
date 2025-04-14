from flask import Flask, request, jsonify, send_file
from PIL import Image
import io
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from models.gmm import GMM
from models.unet import UnetGenerator
import os
from flask_cors import CORS
import sys
from utilities import load_checkpoint
from customimages import CPDataset
from collections import namedtuple
import numpy as np  # Required for head mask blending


app = Flask(__name__)
CORS(app)

# Load model once
def get_model_opt(use_cuda=False):
    options = namedtuple('options', ['fine_width', 'fine_height', 'radius', 'grid_size', 'use_cuda'])
    return options(fine_width=192, fine_height=256, radius=5, grid_size=5, use_cuda=use_cuda)

# Replace argparse with a dictionary of options for Gunicorn
def get_default_options():
    return {
        'dataroot': 'data',
        'datamode': 'test',
        'data_list': 'test_pairs.txt',
        'result_dir': 'predictions',
        'gpu_ids': '',
        'use_cuda': False,
        'fine_width': 192,
        'fine_height': 256,
        'radius': 3,
        'stage': 'GMM'
    }

# Health check endpoint
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'status': 'Virtual Try-On API is running!'}), 200

@app.route('/', methods=['POST'])
def tryon():
    if 'person' not in request.files or 'cloth' not in request.files:
        return jsonify({'error': 'Missing files'}), 400
    
    opt =  get_default_options()
    model_opts = get_model_opt(use_cuda=False)

    pretrained_gmm_path = os.path.join("checkpoints", "train_gmm_200K", "gmm_final.pth")
    pretrained_tom_path = os.path.join("checkpoints", "train_tom_200K", "tom_final.pth")


    person_img = Image.open(request.files['person']).convert("RGB").resize((192, 256))
    cloth_img = Image.open(request.files['cloth']).convert("RGB").resize((192, 256))

    data_loader = CPDataset(opt, person_img, cloth_img)
    inputs = data_loader[0]

    # Prepare inputs
    if opt['use_cuda']:
        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        im_g = inputs['grid_image'].cuda()
    else:
        agnostic = inputs['agnostic']
        c = inputs['cloth']
        cm = inputs['cloth_mask']
        im_g = inputs['grid_image']

    im = inputs['image']

    agnostic = agnostic.unsqueeze(0)
    c = c.unsqueeze(0)
    cm = cm.unsqueeze(0)
    im_g = im_g.unsqueeze(0)

 # GMM predictions
    model = GMM(model_opts)
    load_checkpoint(model, pretrained_gmm_path, opt['use_cuda'])

    # Inference - GMM
    with torch.no_grad():
        grid, _ = model(agnostic, c)
        warped_cloth = F.grid_sample(c, grid, padding_mode='border', align_corners=False)
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros', align_corners=True)
        warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros', align_corners=False)

        # TOM predictions
    model_tom = UnetGenerator(25, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
    load_checkpoint(model_tom, pretrained_tom_path, opt['use_cuda'])

    with torch.no_grad():
        outputs = model_tom(torch.cat([agnostic, warped_cloth], 1))
        p_rendered, m_composite = torch.split(outputs, 3, 1)
        p_rendered = F.tanh(p_rendered)
        m_composite = F.sigmoid(m_composite)
        p_tryon = warped_cloth * m_composite + p_rendered * (1 - m_composite)

    gmm_overlay = warped_cloth + im
    gmm_overlay = gmm_overlay / torch.max(gmm_overlay)
    result_img = (p_tryon.squeeze(0).cpu().permute(1, 2, 0) * 0.5) + 0.5
    result_img = result_img.numpy()

    # Head preservation
    height = result_img.shape[0]
    head_height = int(height * 0.20)
    transition_height = int(height * 0.10)
    head_mask = np.zeros_like(result_img)
    for i in range(head_height + transition_height):
        if i < head_height:
            head_mask[i,:] = 1
        else:
            alpha = 1 - (i - head_height)/transition_height
            head_mask[i,:] = alpha

    person_np = np.array(person_img) / 255.0
    final_img = result_img * (1 - head_mask) + person_np * head_mask
    final_img = Image.fromarray((final_img * 255).astype(np.uint8))

    # Save to buffer
    buf = io.BytesIO()
    final_img.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
