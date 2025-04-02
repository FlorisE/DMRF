import argparse
import json

from common import *
from scenes import *

import pyngp as ngp # noqa

from tqdm import tqdm

import cv2
from scipy.spatial.transform import Rotation as R
import time

import os


def parse_args():
    parser = argparse.ArgumentParser(description="Run neural graphics primitives testbed with additional configuration & output options")

    parser.add_argument("--mode", default="", const="nerf", nargs="?", choices=["nerf", "sdf", "image", "volume"], help="Mode can be 'nerf', 'sdf', 'image' or 'volume'. Inferred from the scene if unspecified.")
    parser.add_argument("--network", default="", help="Path to the network config. Uses the scene's default if unspecified.")

    parser.add_argument("--load_snapshot", default="/home/user/src/ClearFields_Dishwasher/data/dishwasher2/COLMAP/base.msgpack", help="Load this snapshot before training. recommended extension: .msgpack")
    # parser.add_argument("--load_snapshot", default="/home/user/src/ClearFields_Dishwasher/data/dishwasher/base.msgpack", help="Load this snapshot before training. recommended extension: .msgpack")

    parser.add_argument("--exposure", default=0.0, type=float, help="Controls the brightness of the image. Positive numbers increase brightness, negative numbers decrease it.")

    parser.add_argument("--video_spp", type=int, default=64, help="Number of samples per pixel. A larger number means less noise, but slower rendering.")

    parser.add_argument("--width", "--screenshot_w", type=int, default=640, help="Resolution width of GUI and screenshots.")
    parser.add_argument("--height", "--screenshot_h", type=int, default=480, help="Resolution height of GUI and screenshots.")

    parser.add_argument("--gui", action="store_true", help="Run the testbed GUI interactively.")

    parser.add_argument("--sharpen", default=0, help="Set amount of sharpening applied to NeRF training images. Range 0.0 to 1.0.")

    # 5.562886598 for original dishwasher, 0.526823178 for dishwasher 2
    parser.add_argument("--depth_scale", type=float, required=True)

    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--exp_id", type=str, required=True)

    parser.add_argument("--output_folder", type=str, required=True)

    return parser.parse_args()

def write_frame(testbed, output_folder):
    testbed.hybrid_render = 1
    testbed.render_mode = testbed.render_mode.MeshSegmentation
    frame = testbed.render(resolution[0], resolution[1], 1, True)
    timestamp = int(time.time())

    cv2.imwrite(f"{args.output_folder}/segmentation/{timestamp}.png", np.uint8(frame[:, :, :3] * 255))
    print('Wrote segmentation')

    testbed.render_mode = testbed.render_mode.MeshDepth
    frame = testbed.render(resolution[0], resolution[1], 1, True)
    depth_raw = frame[..., 0]
    depth_int = 1000 * depth_raw
    cv2.imwrite(f"{args.output_folder}/depth_mesh/{timestamp}.png", np.uint16(depth_int))
    print('Wrote depth')

    testbed.render_mode = testbed.render_mode.Shade
    frame = testbed.render(resolution[0], resolution[1], args.video_spp, True)
    write_image(f"{args.output_folder}/color/{timestamp}.jpg", np.clip(frame * 2**args.exposure, 0.0, 1.0), quality=100)

    testbed.hybrid_render = 0
    testbed.render_mode = testbed.render_mode.Depth
    frame = testbed.render(resolution[0], resolution[1], 8, True)
    depth_raw = frame[..., 0]
    depth_int = 1000 * depth_raw
    cv2.imwrite(f"{args.output_folder}/depth_scene/{timestamp}.png", np.uint16(depth_int))

    json.dump({'T': testbed.camera_matrix.tolist()}, open(f"{args.output_folder}/camera_matrix/{timestamp}.json", 'w'))


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(f'{args.output_folder}/segmentation', exist_ok=True)
    os.makedirs(f'{args.output_folder}/color', exist_ok=True)
    os.makedirs(f'{args.output_folder}/depth_mesh', exist_ok=True)
    os.makedirs(f'{args.output_folder}/depth_scene', exist_ok=True)
    os.makedirs(f'{args.output_folder}/depth_merged', exist_ok=True)
    os.makedirs(f'{args.output_folder}/camera_matrix', exist_ok=True)

    mode = ngp.TestbedMode.Nerf
    configs_dir = os.path.join(ROOT_DIR, "configs", "nerf")
    scenes = scenes_nerf

    network = os.path.join(configs_dir, "base.json")
    if not os.path.isabs(network):
        network = os.path.join(configs_dir, network)

    testbed = ngp.Testbed(mode)
    testbed.init_rt(config_path=f'scripts/{args.experiment}/{args.exp_id}/exp.json')

    testbed.nerf.sharpen = float(args.sharpen)
    testbed.exposure = args.exposure

    snapshot = args.load_snapshot
    if not os.path.exists(snapshot) and snapshot in scenes:
        snapshot = default_snapshot_filename(scenes[snapshot])
    print("Loading snapshot ", snapshot)
    testbed.load_snapshot(snapshot)

    testbed.shall_train = False
    testbed.nerf.render_with_lens_distortion = True

    # testbed.background_color = [0.843, 0.843, 0.843, 1.000]
    background_tint = np.random.random()
    testbed.background_color = [background_tint, background_tint, background_tint, 1.000]
    testbed.rt_set_shadow_decay(0.3)
    view_dir_1 = np.array([0.688,-0.528,-0.498])
    view_dir_2 = np.array([0.646,-0.483,-0.592])    
    testbed.look_at = [0.302,0.331,0.084]
    half_circle = 40    
    testbed.view_dir = view_dir_1
    # testbed.scale = 1.500
    testbed.scale = 1
    testbed.hybrid_render = 0
    testbed.depth_scale = args.depth_scale
    n_frames = 1
    resolution = [args.width or 1920, args.height or 1080]

    testbed.rt_depth = True

    if "tmp" not in os.listdir():
        os.makedirs("tmp")
    testbed.init_window(args.width, args.height, False)
    while testbed.frame():
        if testbed.want_repl():
            write_frame(testbed, args.output_folder)
            exit()
