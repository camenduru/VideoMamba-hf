import shlex
import subprocess
import spaces
import torch

# install packages for mamba
def install():
    print("Install personal packages", flush=True)
    subprocess.run(shlex.split("pip install causal_conv1d-1.0.0-cp310-cp310-linux_x86_64.whl"))
    subprocess.run(shlex.split("pip install mamba_ssm-1.0.1-cp310-cp310-linux_x86_64.whl"))

install()

import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from decord import VideoReader
from decord import cpu
from videomamba_image import videomamba_image_tiny
from videomamba_video import videomamba_tiny
from kinetics_class_index import kinetics_classnames
from imagenet_class_index import imagenet_classnames
from transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop,
    Stack, ToTorchFormatTensor
)

import gradio as gr
from huggingface_hub import hf_hub_download


# Device on which to run the model
# Set to cuda to load on GPU
device = "cuda"
model_video_path = hf_hub_download(repo_id="OpenGVLab/VideoMamba", filename="videomamba_t16_k400_f16_res224.pth")
model_image_path = hf_hub_download(repo_id="OpenGVLab/VideoMamba", filename="videomamba_t16_in1k_res224.pth")
# Pick a pretrained model
model_video = videomamba_tiny(num_classes=400, num_frames=16)
video_sd = torch.load(model_video_path, map_location='cpu')
model_video.load_state_dict(video_sd)
model_image = videomamba_image_tiny()
image_sd = torch.load(model_image_path, map_location='cpu')
model_image.load_state_dict(image_sd['model'])
# Set to eval mode and move to desired device
model_video = model_video.to(device).eval()
model_image = model_image.to(device).eval()

# Create an id to label name mapping
kinetics_id_to_classname = {}
for k, v in kinetics_classnames.items():
    kinetics_id_to_classname[k] = v
imagenet_id_to_classname = {}
for k, v in imagenet_classnames.items():
    imagenet_id_to_classname[k] = v[1]


def get_index(num_frames, num_segments=8):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets


def load_video(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    num_frames = len(vr)
    frame_indices = get_index(num_frames, 16)

    # transform
    crop_size = 224
    scale_size = 224
    input_mean = [0.485, 0.456, 0.406]
    input_std = [0.229, 0.224, 0.225]

    transform = T.Compose([
        GroupScale(int(scale_size)),
        GroupCenterCrop(crop_size),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(input_mean, input_std)
    ])

    images_group = list()
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy())
        images_group.append(img)
    torch_imgs = transform(images_group)
    return torch_imgs


@spaces.GPU
def inference_video(video):
    vid = load_video(video)

    # The model expects inputs of shape: B x C x H x W
    TC, H, W = vid.shape
    inputs = vid.reshape(1, TC//3, 3, H, W).permute(0, 2, 1, 3, 4)

    with torch.no_grad():
        prediction = model_video(inputs.to(device))
        prediction = F.softmax(prediction, dim=1).flatten()

    return {kinetics_id_to_classname[str(i)]: float(prediction[i]) for i in range(400)}


@spaces.GPU
def inference_image(img):
    image = img
    image_transform = T.Compose(
    [
        T.Resize(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    )
    image = image_transform(image)

    # The model expects inputs of shape: B x C x H x W
    image = image.unsqueeze(0)

    with torch.no_grad():
        prediction = model_image(image.to(device))
        prediction = F.softmax(prediction, dim=1).flatten()

    return {imagenet_id_to_classname[str(i)]: float(prediction[i]) for i in range(1000)}


demo = gr.Blocks()
with demo:
    gr.Markdown(
        """
        # VideoMamba-Ti
        Gradio demo for <a href='https://github.com/OpenGVLab/VideoMamba' target='_blank'>VideoMamba</a>: To use it, simply upload your video, or click one of the examples to load them. Read more at the links below.
        """
    )

    with gr.Tab("Video"):
        # with gr.Box():
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    input_video = gr.Video(label='Input Video', height=360)
                with gr.Row():
                    submit_video_button = gr.Button('Submit')
            with gr.Column():
                    label_video = gr.Label(num_top_classes=5)
        with gr.Row():
            gr.Examples(examples=['./videos/hitting_baseball.mp4', './videos/hoverboarding.mp4', './videos/yoga.mp4'], inputs=input_video, outputs=label_video, fn=inference_video, cache_examples=True)

    with gr.Tab("Image"):
        # with gr.Box():
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    input_image = gr.Image(label='Input Image', type='pil', height=360)
                with gr.Row():
                    submit_image_button = gr.Button('Submit')
            with gr.Column():
                label_image = gr.Label(num_top_classes=5)
        with gr.Row():
            gr.Examples(examples=['./images/cat.png', './images/dog.png', './images/panda.png'], inputs=input_image, outputs=label_image, fn=inference_image, cache_examples=True)

    gr.Markdown(
        """
        <p style='text-align: center'><a href='https://arxiv.org/abs/2403.06977' target='_blank'>VideoMamba: State Space Model for Efficient Video Understanding</a> | <a href='https://github.com/OpenGVLab/VideoMamba' target='_blank'>Github Repo</a></p>
        """
    )

    submit_video_button.click(fn=inference_video, inputs=input_video, outputs=label_video)
    submit_image_button.click(fn=inference_image, inputs=input_image, outputs=label_image)

demo.queue(max_size=20).launch()
