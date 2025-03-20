#Code for Neural Style Transfer with and without Color Preservation using MODNet

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights
from rembg import remove
import copy
import cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

# setting image size high due to availability of GPU
#imsize = 512 if torch.cuda.is_available() else 128  # use small size if no GPU
imsize = 1024 if torch.cuda.is_available() else 128  # use small size if no GPU

# transform the image into a torch tensor
loader = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor()])

# function to define the image loader
def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

# function to define the image loader for color preservation
# not utilized in this code
def image_loader_2(image):
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

#################################
# set the content image and style images from local machine
style = ""
content = ""
#################################

style_img = image_loader(style)
content_img = image_loader(content)

# use MODNet's rembg to remove background from content image
input_path = content
inp = Image.open(input_path)
content_with_no_background = remove(inp)

#content_with_no_background.save("NO_BG.png")

# save the alpha channel
alpha_channel = content_with_no_background.getchannel("A")  # Extract the alpha channel
alpha_channel.save("ALPHA_CHANNEL.png")  # Optional debug save

# separate RGB and alpha channels
rgb_content = content_with_no_background.convert("RGB")
#alpha_channel = content_with_no_background.split()[-1]  # Extract the alpha channel

# convert RGB content image to tensor
rgb_content_img = image_loader_2(rgb_content)

# store the original size of the content image
original_size = rgb_content_img.shape[2:]
#print(style_img.size())
#print(rgb_content_img.size())

# resize style image and luminance content image if their sizes do not match
if style_img.size() != rgb_content_img.size():
    style_img = transforms.functional.resize(style_img.squeeze(0), original_size).unsqueeze(0)
    #rgb_content_img = transforms.functional.resize(rgb_content_img, (imsize, imsize))

# resize luminance content image and the original image if their sizes do not match
if rgb_content_img.shape[2:] != original_size:
    rgb_content_img = transforms.functional.resize(rgb_content_img.squeeze(0), original_size).unsqueeze(0)
    content_img = transforms.functional.resize(content_img.squeeze(0), original_size).unsqueeze(0)

# print to check size
#print(style_img.size())
#print(rgb_content_img.size())

# assert if sizes still do not match
assert style_img.size() == rgb_content_img.size(), \
    "we need to import style and content images of the same size"

assert style_img.size() == content_img.size(), \
            "we need to import style and content images of the same size"

# reconvert into PIL image
unloader = transforms.ToPILImage()

plt.ion()

# function to show the image
# not utilized in the final code due to running it on GPU server
def imshow(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    #plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


#plt.figure()
#imshow(style_img, title='Style Image')

#plt.figure()
#imshow(content_img, title='Content Image')

# class to compute content loss
class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we normalize the values of the gram matrix
    # by dividing by the number of element in each feature maps
    # as shown in the ppt slides/final report
    return G.div(a * b * c * d)

# class to compute style loss
class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

# use pretrained VGG-19
cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

# create a module to normalize input image so we can easily put it in a nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize ``img``
        return (img - self.mean) / self.std

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# function to get the losses
def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    # normalization module
    normalization = Normalization(normalization_mean, normalization_std)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    # increment every time we see a conv
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        # add content loss:
        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        # add style loss:
        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


input_img = content_img.clone()

# add the original input image to the figure:
plt.figure()
#imshow(input_img, title='Input Image')

# using L-BFGS optimizer
def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img], lr = 0.1)
    return optimizer

# function to run the style transfer
def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, style_weight, content_weight,
                       num_steps):

    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)

    input_img.requires_grad_(True)
    model.eval()
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        # correct the values of updated input image
        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img

#################################
# set the hyperparameters
background_style_weight = 0
background_content_weight = 0
mask_style_weight= 0
mask_content_weight= 0
###############################

# remove batch dimension and move to CPU
rgb_content_img_pil = transforms.ToPILImage()(rgb_content_img.squeeze(0).cpu())
#rgb_content_img_pil.save("1RGB_CONTENT_DEBUG.png")
#assert 1==0

# run style transfer for the human image with less style more content (2000 epochs)
styled_rgb_content_img = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            rgb_content_img, style_img, rgb_content_img, mask_style_weight, mask_content_weight, 2000)

# run style transfer for the background image with more style less content (2000 epochs)
output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                    content_img, style_img, input_img, background_style_weight, background_content_weight, 2000)

#print("rgb_content_img shape:", rgb_content_img.shape)
#print("content_img shape:", content_img.shape)

# convert back to PIL image for recombining
styled_rgb_content = transforms.ToPILImage()(styled_rgb_content_img.squeeze(0).cpu()).convert("RGBA")
styled_output = transforms.ToPILImage()(output.squeeze(0).cpu()).convert("RGBA")

# resize alpha channel to match the styled image
alpha_resized = alpha_channel.resize(styled_rgb_content.size, Image.ANTIALIAS)

# add the alpha channel back
styled_rgb_content.putalpha(alpha_resized)
#styled_rgb_content.save("STYLED_NO_BACKGROUND.png")

# blend the foreground (`styled_rgb_content`) with the background (`styled_output`)
final_combined_image = Image.alpha_composite(styled_output, styled_rgb_content)

# Save the final combined image with color tranfser
final_combined_image.save("FINAL_COMBINED_IMAGE.png")
print("Final combined image saved as 'FINAL_COMBINED_IMAGE.png'")

# function to merge luminance with chrominance for color preservation
def merge_luminance_with_color(original_image, stylized_luminance):

    # extract U and V channels (color information) of the original image
    original_image = np.array(original_image)
    yuv_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2YUV)
    chrominance = yuv_image[:, :, 1:]

    # convert stylized luminance to NumPy array and ensure it's 2D
    stylized_luminance = stylized_luminance.squeeze().cpu().numpy()
    stylized_luminance = 0.299 * stylized_luminance[0, :, :] + 0.587 * stylized_luminance[1, :, :] + 0.114 * stylized_luminance[2, :, :]

    # check dimensions of stylized_luminance
    if len(stylized_luminance.shape) != 2:
        raise ValueError("Stylized luminance should be a 2D array, but got shape: {}".format(stylized_luminance.shape))

    # resize stylized luminance to match original image dimensions for added safety
    stylized_luminance_resized = cv2.resize(
        stylized_luminance,
        (original_image.shape[1], original_image.shape[0]),
        interpolation=cv2.INTER_CUBIC
    )

    # normalize stylized luminance to [0, 1]
    stylized_luminance_resized = (stylized_luminance_resized - stylized_luminance_resized.min()) / \
        (stylized_luminance_resized.max() - stylized_luminance_resized.min())

    # merge the resized stylized luminance with original chrominance
    # create an empty YUV image with the same shape
    stylized_yuv = np.zeros_like(yuv_image)
    # replace Y channel
    stylized_yuv[:, :, 0] = np.clip(stylized_luminance_resized * 255, 0, 255).astype(np.uint8)
    # retain original U and V channels (color information)
    stylized_yuv[:, :, 1:] = chrominance

    # debugging
    #print("Stylized YUV sample:", stylized_yuv[0, 0, :])

    # convert back to RGB color space
    stylized_rgb = cv2.cvtColor(stylized_yuv, cv2.COLOR_YUV2RGB)

    # debugging to check sizes
    print("Original chrominance shape:", chrominance.shape)
    print("Stylized luminance shape:", stylized_luminance_resized.shape)
    print("Final YUV shape:", stylized_yuv.shape)

    return Image.fromarray(stylized_rgb)

# reload the stylized image
combined_image_loaded = Image.open("FINAL_COMBINED_IMAGE.png")

# resize the combined image back to the original content image size
combined_image_resized = combined_image_loaded.resize(original_size, Image.ANTIALIAS)

# convert the combined image into a tensor
combined_image_tensor = transforms.ToTensor()(combined_image_resized).unsqueeze(0).to(device)

# convert the resized tensor back to a NumPy array for merging
stylized_luminance = combined_image_tensor.squeeze(0).detach()
print("Stylized luminance shape:", stylized_luminance.shape)

# merge the stylized luminance with the combined image's color
output_image = merge_luminance_with_color(Image.open(content).convert('RGB'), stylized_luminance)

out_name = "FINAL_COMBINED_IMAGE_COLORED.jpg"
# Save the image
output_image.save(out_name)
print(f"Output image saved as '{out_name}'")