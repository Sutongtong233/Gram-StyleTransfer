from urllib.parse import urlparse
import torch
from torch.hub import _download_url_to_file
import re
import os
import glob

def download_model(url, dst_path):
    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    
    HASH_REGEX = re.compile(r'-([a-f0-9]*)\.')
    hash_prefix = HASH_REGEX.search(filename).group(1)
    
    _download_url_to_file(url, os.path.join(dst_path, filename), hash_prefix, True)
    return filename

def load_model(model_name, model_dir):
    model  = eval('models.%s(init_weights=False)' % model_name)
    path_format = os.path.join(model_dir, '%s-[a-z0-9]*.pth' % model_name)
    model_path = glob.glob(path_format)[0]
    model.load_state_dict(torch.load(model_path))
    return model


model_urls = {
    # 'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    # 'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    # 'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    # 'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    # 'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    # 'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    # 'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

if __name__ == "__main__":
    path = '/home/tongtong/python_project/CV/Pretrain-models'
    if not (os.path.exists(path)):
        os.makedirs(path)
    for url in model_urls.values():
        download_model(url, path)
