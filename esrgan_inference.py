import os
import os.path as osp
import glob
import cv2
import numpy as np
import torch
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from tqdm import tqdm

import RRDBNet_arch as arch

# Settings
model_path = 'models/RRDB_ESRGAN_x4_fine_tuned.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_img_folder = 'LR/*'
hr_img_folder = 'HR_reference/'
results_folder = 'results'
csv_output_path = 'results/psnr_only_metrics.csv'

os.makedirs(results_folder, exist_ok=True)

# Load model
model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
model.eval()
model = model.to(device)
print(f'Model loaded from {model_path}\nTesting...')

# Results container
results = []

# Inference and PSNR evaluation
for path in tqdm(glob.glob(test_img_folder)):
    base = osp.splitext(osp.basename(path))[0]

    # Read LR image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float().unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(img).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output_img = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output_bgr = (output_img * 255.0).round().astype(np.uint8)
    cv2.imwrite(f'{results_folder}/{base}_rlt.png', output_bgr)

    # Read corresponding HR image
    hr_path = osp.join(hr_img_folder, f'{base}.png')
    if not osp.exists(hr_path):
        print(f'⚠️ HR image not found for: {base}. Skipping PSNR.')
        continue

    hr_img = cv2.imread(hr_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0

    # Resize output to match HR shape if needed
    if hr_img.shape != output_img.shape:
        output_img = cv2.resize(output_img, (hr_img.shape[1], hr_img.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Calculate PSNR
    psnr = compare_psnr(hr_img, output_img, data_range=1.0)

    results.append({
        'Image': base,
        'PSNR': psnr
    })

# Save results
df_results = pd.DataFrame(results)
df_results.to_csv(csv_output_path, index=False)

# Print average PSNR
print("\nAverage PSNR:")
print(df_results['PSNR'].mean())
