from __future__ import print_function
import os
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms
import utils
from model import centerEsti
from model import F26_N9
from model import F17_N9
from model import F35_N8

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# param
input = "input.png"
cuda = True

# load model
model1 = centerEsti()
model2 = F35_N8()
model3 = F26_N9()
model4 = F17_N9()


utils.load_state_dict(model1, 'models/center_v3.pth')
print('model1 loaded!')
utils.load_state_dict(model2, 'models/F35_N8.pth')
print('model2 loaded!')
utils.load_state_dict(model3, 'models/F26_N9_from_F35_N8.pth')
print('model3 loaded!')
utils.load_state_dict(model4, 'models/F17_N9_from_F26_N9_from_F35_N8.pth')
print('model4 loaded!')

if cuda:
    model1.cuda()
    model2.cuda()
    model3.cuda()
    model4.cuda()

model1.eval()
model2.eval()
model3.eval()
model4.eval()

inputFile = input
input = utils.load_image(inputFile)
width, height= input.size
input = input.crop((0,0, width-width%20, height-height%20))
input_transform = transforms.Compose([
    transforms.ToTensor(),
])
input = input_transform(input)
input = input.unsqueeze(0)
if cuda:
    input = input.cuda()
input = Variable(input, volatile=True)
output4 = model1(input)
output3_5 = model2(input, output4)
output2_6 = model3(input, output3_5[0], output4, output3_5[1])
output1_7 = model4(input, output2_6[0], output3_5[0], output3_5[1], output2_6[1])
if cuda:
    output1 = output1_7[0].cpu()
    output2 = output2_6[0].cpu()
    output3 = output3_5[0].cpu()
    output4 = output4.cpu()
    output5 = output3_5[1].cpu()
    output6 = output2_6[1].cpu()
    output7 = output1_7[1].cpu()
else:
    output1 = output1_7[0]
    output2 = output2_6[0]
    output3 = output3_5[0]
    output4 = output4
    output5 = output3_5[1]
    output6 = output2_6[1]
    output7 = output1_7[1]
output_data = output1.data[0]*255
utils.save_image('result/'+inputFile[:-4] + '-esti1' + inputFile[-4:], output_data)                
output_data = output2.data[0]*255
utils.save_image('result/'+inputFile[:-4] + '-esti2' + inputFile[-4:], output_data)                
output_data = output3.data[0]*255
utils.save_image('result/'+inputFile[:-4] + '-esti3' + inputFile[-4:], output_data)
output_data = output4.data[0]*255
utils.save_image('result/'+inputFile[:-4] + '-esti4' + inputFile[-4:], output_data)
output_data = output5.data[0]*255
utils.save_image('result/'+inputFile[:-4] + '-esti5' + inputFile[-4:], output_data)
output_data = output6.data[0]*255
utils.save_image('result/'+inputFile[:-4] + '-esti6' + inputFile[-4:], output_data)
output_data = output7.data[0]*255
utils.save_image('result/'+inputFile[:-4] + '-esti7' + inputFile[-4:], output_data)
