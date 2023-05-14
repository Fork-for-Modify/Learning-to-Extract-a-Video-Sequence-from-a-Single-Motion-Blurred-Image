import torch,os
from PIL import Image
from torch.autograd import Variable


def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img


def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.data.new(batch.data.size())
    std = batch.data.new(batch.data.size())
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225
    # batch = torch.div(batch, 255.0)
    out =batch - Variable(mean)
    out = out / Variable(std)
    return out

def load_state_dict(network, ckp_path, state_dirc_key ='state_dict_G', verbose=False):
    try:    # try directly load
        network.load_state_dict(torch.load(ckp_path))
    except:     # if excetion, delete running_mean & running var, then load again
        state_dict = torch.load(ckp_path).get(state_dirc_key)
        for k in list(state_dict.keys()):
            if (k.find('running_mean')>0) or (k.find('running_var')>0):
                del state_dict[k]
                if verbose:
                    print('\n'.join(map(str,sorted(state_dict.keys()))))
            
        network.load_state_dict(state_dict)