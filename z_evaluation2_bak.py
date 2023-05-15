import os, torch, time
from tqdm import tqdm
from z_utils import gpu_inference_time, model_complexity,param_counting, tensor2uint, imsave, create_logger
from z_dataloader_bak import get_data_loaders
from z_model import CE_LEV
import pyiqa 
from datetime import datetime
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

## params
# data_dir="/ssd/zzh/dataset/GoPro/GOPRO_Large_all/mini_test/"
# data_dir="/ssd/zzh/dataset/GoPro/GOPRO_Large_all/test/"
# data_dir="/ssd/zzh/dataset/WAIC_TSR_GT/easy/"  # WAIC_TSR dataset | bicycle/
data_dir="/ssd/zzh/dataset/WAIC_TSR_GT/hard/"  # WAIC_TSR dataset
outputs_dir = 'result/'
ckp_paths = ['models/center_v3.pth', 'models/F35_N8.pth', 'models/F26_N9_from_F35_N8.pth', 'models/F17_N9_from_F26_N9_from_F35_N8.pth']
frame_num = 7
ce_code = [1] * 7 # all one sequence
sigma_range=0
batch_size = 1
save_img = False
device = "cuda:0"

## dataset
dataloader = get_data_loaders(data_dir, frame_num, batch_size, patch_size=None, tform_op=None, sigma_range=0, shuffle=False, validation_split=0, status='test', num_workers=8, pin_memory=True, prefetch_factor=1, all2CPU=True)

## load model
model = CE_LEV(sigma_range=sigma_range, ce_code_n=frame_num, frame_n=frame_num, ce_code_init=ce_code)
model.LEV.load_state_dicts(ckp_paths=ckp_paths)
model.to(device)

## init
datestr = datetime.now().strftime("%y%m%d_%H-%M-%S")
outputs_dir = outputs_dir+data_dir.split('/')[-2]+'/'+datestr+'/'

if not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir)
if save_img:
    os.makedirs(outputs_dir+'/output')
    os.makedirs(outputs_dir+'/target')
    os.makedirs(outputs_dir+'/input')
    
logger = create_logger(outputs_dir+'test.log')   

met_psnr = pyiqa.create_metric(metric_name='psnr')
met_ssim = pyiqa.create_metric(metric_name='ssim')
met_lpips = pyiqa.create_metric(metric_name='lpips')
metrics = {"met_psnr":met_psnr, "met_ssim":met_ssim, "met_lpips":met_lpips}

## eval
model.eval()
# calc MACs & Param. Num
model_complexity(model=model, input_shape=(7, 3, 200, 320), logger=logger)
# param_counting(model, logger=logger)

# run
total_loss = 0.0
total_metrics = {"met_psnr":0, "met_ssim":0, "met_lpips":0}
time_start = time.time()
with torch.no_grad():
    for i, vid in enumerate(tqdm(dataloader, desc='Testing')):
        # move vid to gpu, convert to 0-1 float
        vid = vid.to(device).float()/255 
        N, M, C, Hx, Wx = vid.shape

        
        # direct inference
        # output, data, data_noisy = model(vid)
        
        # pad & crop
        sf = 20
        HX, WX = int((Hx+sf-1)/sf)*sf, int((Wx+sf-1)/sf) * \
            sf  # pad to a multiple of scale_factor (sf)
        pad_h, pad_w = HX-Hx, WX-Wx
        vid_pad = F.pad(vid, [0, pad_w, 0, pad_h])
        output, data, data_noisy = model(vid_pad)
        output = output[:, :, :, :Hx, :Wx]

        

        # clamp to 0-1
        output = torch.clamp(output, 0, 1)

        # save some sample images
        if save_img:
            scale_fc = 1
            for k, (in_img, out_img, gt_img) in enumerate(zip(data, output, vid)):
                in_img = tensor2uint(in_img*scale_fc)
                imsave(
                    in_img, f'{outputs_dir}input/ce-blur#{i*N+k+1:04d}.jpg')
                for j in range(frame_num):
                    out_img_j = tensor2uint(out_img[j])
                    gt_img_j = tensor2uint(gt_img[j])
                    imsave(
                        out_img_j, f'{outputs_dir}output/out-frame#{i*N+k+1:04d}-{j+1:04d}.jpg')
                    imsave(
                        gt_img_j, f'{outputs_dir}target/gt-frame#{i*N+k+1:04d}-{j+1:04d}.jpg')
            # break  # save one image per batch

        # computing loss, metrics on test set
        output_all = torch.flatten(output, end_dim=1)
        target_all = torch.flatten(vid, end_dim=1)
        # loss = criterion(output_all, target_all)
        batch_size = data.shape[0]
        # total_loss += loss.item() * batch_size
        for met_name, met_fc in metrics.items():
            total_metrics[met_name] += torch.mean(met_fc(output_all, target_all)) * batch_size
time_end = time.time()
time_cost = time_end-time_start
n_samples = len(dataloader.sampler)
log = {#'loss': total_loss / n_samples,
        'time/sample': time_cost/n_samples,
        'ce_code': ce_code}
log.update({met_name: round(total_metrics[met_name].item() / n_samples,4) for  met_name in metrics.keys()})

logger.info(log)

