from matplotlib import artist
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.models as models
from torch import optim
from model import *
from PIL import Image

def load_image(img_path, max_size=400, shape=None):
    ''' Load in and transform an image, making sure the image
    is <= 400 pixels in the x-y dims.'''
    image = Image.open(img_path).convert('RGB')
    # large images will slow down processing
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    if shape is not None:
        size = shape
    in_transform = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
    (0.229, 0.224, 0.225))])
    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    return image
# helper function for un-normalizing an image
# and converting it from a Tensor image to a NumPy image for display

def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image

def get_features(image, model, layers=None):
    """ Run an image forward through a model and get the features for
    a set of layers. Default layers are for VGGNet matching Gatys et al (2016)
    """
    '''
    vgg16: features, avg_pool, classifier
    '''
    ## Need the layers for the content and style representations of an image

    if layers is None:
        layers = {'0': 'conv1_1',
	'5': 'conv2_1',
	'10': 'conv3_1',
	'19': 'conv4_1',
	'21': 'conv4_2',  ## content representation is output of this layer
	'28': 'conv5_1'
    }

    features = {}
    x = image
    # model._modules is a dictionary holding each module in the model
    # 31 layers, 19 Conv2D
    for name, layer in model._modules.items(): # TODO: 少了个flatten层？
        if name == 'features':
            extract_ls = [0, 5, 10, 19, 21, 28]
            for i in range(31):
                x = layer[i](x)
                if i in extract_ls:
                    features[layers[str(i)]] = x
        elif name == 'classifier':
            x = torch.flatten(x)
            x = layer(x)
        else:
            x = layer(x)

    return features
    # get content and style features only once before training

def gram_matrix(tensor):
	""" Calculate the Gram Matrix of a given tensor
	Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
	"""
	# get the batch_size, depth, height, and width of the Tensor
	_, d, h, w = tensor.size()
	# reshape so we're multiplying the features for each channel
	tensor = tensor.view(d, h * w)
	# calculate the gram matrix
	gram = torch.mm(tensor, tensor.t())
	return gram



if __name__ == "__main__":
    device = 'cpu'
    vgg16 = models.vgg16(pretrained=False)
    vgg16.to(device)
    layers = {'0': 'conv1_1',
	'5': 'conv2_1',
	'10': 'conv3_1',
	'19': 'conv4_1',
	'21': 'conv4_2',  ## content representation is output of this layer
	'28': 'conv5_1'
    }
    style_weights = {
        'conv1_1': 1,
        'conv2_1': 1,
        'conv3_1': 1,
        'conv4_1': 1,
        'conv4_2': 1,
        'conv5_1': 1,
    }
    # loss weight hyperparams
    content_weight = 1
    style_weight = 1
    content_dic = '/home/tongtong/python_project/CV/StyleTransfer/images/content/'
    style_dic = '/home/tongtong/python_project/CV/StyleTransfer/images/reference/'
    result_dic = '/home/tongtong/python_project/CV/StyleTransfer/images/result/'
    content_name = '1'
    artist_name = 'van-gogh'
    index = '1'
    # load in content and style image, using shape parameter to make both content and style of same shape to make processing easier
    content = load_image(content_dic + content_name + ".jpg", shape=[400,400]).to(device)
    style = load_image(style_dic + artist_name + '/' + index + ".jpg", shape=[400,400]).to(device)
    target = load_image(content_dic + content_name + ".jpg", shape=[400,400]).to(device)
    target.requires_grad = True

    content_features = get_features(content, vgg16)
    style_features = get_features(style, vgg16)

    # for displaying the target image, intermittently
    show_every = 100
    # iteration hyperparameters
    optimizer = optim.Adam([target], lr=0.003)
    steps = 2000 # decide how many iterations to update your image (5000)
    for ii in range(1, steps+1):
        # get the features from your target image
        target_features = get_features(target, vgg16)
        # the content loss
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)  # TODO: choice here
        # the style loss
        # initialize the style loss to 0
        style_loss = 0
        # then add to it for each layer's gram matrix loss
        for layer in layers.values():
        # get the "target" style representation for the layer
            target_feature = target_features[layer]
            style_feature = style_features[layer]
            target_gram = gram_matrix(target_feature)
            style_gram = gram_matrix(style_feature)
            _, d, h, w = target_feature.shape
            # the style loss for one layer, weighted appropriately
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
            # add to the style loss
            style_loss += layer_style_loss / (d * h * w)
        # calculate the *total* loss
        total_loss = content_weight * content_loss + style_weight * style_loss
        # update your target image
        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()
    # display intermediate images and print the loss
        if  ii % 10 == 0:
            print('Total loss: ', total_loss.item())
        if  ii % show_every == 0:
            plt.imshow(im_convert(target))
            plt.savefig(result_dic + content_name + '-' + artist_name + '-' + str(ii) + '.jpg')
            print(1)

