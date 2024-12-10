import os.path as osp
import glob
from PIL import Image
import numpy as np
import torch
import RRDBNet_arch as arch

model_path = 'models/RRDB_ESRGAN_x4.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # if you want to run on CPU, change 'cuda' -> cpu

test_img_folder = 'LR/*'

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

print(f'Model path {model_path}. \nTesting...')

idx = 0
for path in glob.glob(test_img_folder):
    idx += 1
    base = osp.splitext(osp.basename(path))[0]
    print(idx, base)
    
    # Read images using Pillow
    img = Image.open(path).convert('RGB')
    img = np.array(img) / 255.0  # Normalize image to range [0, 1]
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    
    # Convert output back to an image
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # Convert to HWC format
    output = (output * 255.0).round().astype('uint8')  # Denormalize to [0, 255]
    output_img = Image.fromarray(output)
    
    # Save image
    output_img.save(f'results/{base}_rlt.png')
