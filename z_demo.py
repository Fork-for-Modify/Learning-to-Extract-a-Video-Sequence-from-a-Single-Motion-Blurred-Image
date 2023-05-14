from __future__ import print_function
import os, torch
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms
import utils
from z_model import LEV
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# param
input = "input.png"
cuda = True
ckp_paths = ['models/center_v3.pth', 'models/F35_N8.pth', 'models/F26_N9_from_F35_N8.pth', 'models/F17_N9_from_F26_N9_from_F35_N8.pth']
frame_num = 7

## load model
model = LEV()
model.load_state_dicts(ckp_paths=ckp_paths)
if cuda:
    model.cuda()
model.eval()


## run

inputFile = input
input = utils.load_image(inputFile)
width, height= input.size
input = input.crop((0,0, width-width%20, height-height%20))
input_transform = transforms.Compose([
    transforms.ToTensor(),
])
input = input_transform(input)
input = input.unsqueeze(0)
with torch.no_grad():
    if cuda:
        input = input.cuda()
    input = Variable(input)
    output = model(input)

for k in tqdm(range(frame_num)):
    output_data = output[:,k,...].data[0]*255
    utils.save_image('demo_result/'+inputFile[:-4] + f'-esti{k}' + inputFile[-4:], output_data)             