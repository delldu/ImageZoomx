"""Image Weather Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 12月 14日 星期二 00:22:28 CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
import time
from tqdm import tqdm
import torch

import redos
import todos

from . import zoomx

import pdb


def get_model():
    """Create model."""

    device = todos.model.get_device()
    model = zoomx.ImageZoomxModel()

    cdir = os.path.dirname(__file__)
    model_path = "models/image_zoomx.pth"
    checkpoint = model_path if cdir == "" else cdir + "/" + model_path
    todos.model.load(model, checkpoint)

    model = model.to(device)
    model.eval()

    return model, device


def model_forward(model, device, input_tensor, zoom_times):
    H = int(zoom_times * input_tensor.size(2))
    W = int(zoom_times * input_tensor.size(3))

    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output_tensor = model(input_tensor, torch.Tensor([H, W]))
    return output_tensor


def image_client(name, input_files, zoom_times, output_dir):
    redo = redos.Redos(name)
    cmd = redos.image.Command()
    image_filenames = todos.data.load_files(input_files)
    for filename in image_filenames:
        output_file = f"{output_dir}/{os.path.basename(filename)}"
        context = cmd.zoomx(filename, zoom_times, output_file)
        redo.set_queue_task(context)
    print(f"Created {len(image_filenames)} tasks for {name}.")


def image_server(name, HOST="localhost", port=6379):
    # load model
    model, device = get_model()

    def image_service(input_file, output_file, targ):
        zoom_times = float(redos.taskarg_search(targ, "zoom_times"))
        print(f"  zoom in {input_file} with {zoom_times} times ...")
        try:
            input_tensor = todos.data.load_tensor(input_file)
            output_tensor = model_forward(model, device, input_tensor, zoom_times)
            todos.data.save_tensor(output_tensor, output_file)
            return True
        except:
            return False

    return redos.image.service(name, "image_zoomx", image_service, HOST, port)


def image_predict(input_files, zoom_times, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_model()

    def image_service(input_file, output_file, targ):
        zoom_times = float(redos.taskarg_search(targ, "zoom_times"))
        print(f"  zoom in {input_file} with {zoom_times} times ...")
        try:
            input_tensor = todos.data.load_tensor(input_file)
            output_tensor = model_forward(model, device, input_tensor, zoom_times)
            todos.data.save_tensor(output_tensor, output_file)
            return True
        except:
            return False

    # load files
    image_filenames = todos.data.load_files(input_files)

    # start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        output_file = f"{output_dir}/{os.path.basename(filename)}"
        # create taskarg for zoom_times
        targ = redos.taskarg_parse(
            f"image_zoomx(input_file={filename},zoom_times={zoom_times},output_file={output_file})"
        )
        image_service(filename, output_file, targ)


def video_service(input_file, output_file, targ):
    zoom_times = float(redos.taskarg_search(targ, "zoom_times"))

    # load video
    video = redos.video.Reader(input_file)
    if video.n_frames < 1:
        print(f"Read video {input_file} error.")
        return False

    # Create directory to store result
    output_dir = output_file[0 : output_file.rfind(".")]
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_model()

    print(f"  zoom in {input_file}, save to {output_file} ...")
    progress_bar = tqdm(total=video.n_frames)

    def zoom_video_frame(no, data):
        # print(f"frame: {no} -- {data.shape}")
        progress_bar.update(1)

        input_tensor = todos.data.frame_totensor(data)

        # convert tensor from 1x4xHxW to 1x3xHxW
        input_tensor = input_tensor[:, 0:3, :, :]
        output_tensor = model_forward(model, device, input_tensor, zoom_times)

        temp_output_file = "{}/{:06d}.png".format(output_dir, no)
        todos.data.save_tensor(output_tensor, temp_output_file)

    video.forward(callback=zoom_video_frame)

    redos.video.encode(output_dir, output_file)

    # delete temp files
    for i in range(video.n_frames):
        temp_output_file = "{}/{:06d}.png".format(output_dir, i)
        os.remove(temp_output_file)

    return True


def video_predict(input_file, zoom_times, output_file):
    # create taskarg for zoom_times
    targ = redos.taskarg_parse(
        f"video_zoomx(input_file={input_file},zoom_times={zoom_times},output_file={output_file})"
    )
    video_service(input_file, output_file, targ)


def video_client(name, input_file, zoom_times, output_file):
    cmd = redos.video.Command()
    context = cmd.zoomx(input_file, zoom_times, output_file)
    redo = redos.Redos(name)
    redo.set_queue_task(context)
    print(f"Created 1 video tasks for {name}.")


def video_server(name, HOST="localhost", port=6379):
    return redos.video.service(name, "video_zoomx", video_service, HOST, port)
