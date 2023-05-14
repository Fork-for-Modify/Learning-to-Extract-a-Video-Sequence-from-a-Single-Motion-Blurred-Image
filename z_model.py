from model import centerEsti,F26_N9,F17_N9,F35_N8
import torch.nn as nn
from utils import load_state_dict
import torch
from z_modules import CEBlurNet

 

class LEV(nn.Module):
    '''
    coded exposure blur decomposition network
    '''
    def __init__(self, frame_n=7):
        super(LEV, self).__init__()
        self.model1 = centerEsti()
        self.model2 = F35_N8()
        self.model3 = F26_N9()
        self.model4 = F17_N9()
        self.frame_n  =frame_n
    
    def load_state_dicts(self, ckp_paths):
        load_state_dict(self.model1, ckp_paths[0])
        load_state_dict(self.model2, ckp_paths[1])
        load_state_dict(self.model3, ckp_paths[2])
        load_state_dict(self.model4, ckp_paths[3])
    
    def forward(self, input):
        n,c,h,w = input.shape
        output = torch.zeros((self.frame_n,n,c,h,w))
        output4 = self.model1(input)
        output3_5 = self.model2(input, output4)
        output2_6 = self.model3(input, output3_5[0], output4, output3_5[1])
        output1_7 = self.model4(input, output2_6[0], output3_5[0], output3_5[1], output2_6[1])

        output[0] = output1_7[0]
        output[1] = output2_6[0]
        output[2] = output3_5[0]
        output[3] = output4
        output[4] = output3_5[1]
        output[5] = output2_6[1]
        output[6] = output1_7[1]

        output = torch.permute(output,(1,0,2,3,4))
        return output


class CE_LEV(nn.Module):
    def __init__(self, sigma_range=0, ce_code_n=8, frame_n=8, ce_code_init=None):
        super(CE_LEV, self).__init__()
        assert ce_code_init is not None, 'please assign ce_code' 
        self.BlurNet = CEBlurNet(
            sigma_range=sigma_range,  ce_code_n=ce_code_n, frame_n=frame_n, ce_code_init=ce_code_init)
        self.LEV = LEV(frame_n=frame_n)

    def forward(self, frames):
        ce_blur_img_noisy, time_idx, ce_code, ce_blur_img = self.BlurNet(frames)
        output = self.LEV(ce_blur_img_noisy)

        return output, ce_blur_img, ce_blur_img_noisy