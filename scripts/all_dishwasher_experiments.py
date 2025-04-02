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
import shutil

NERF_DIR = "/home/user/src/ClearFields_Training/10k/10k_NeRF"

def parse_args():
    parser = argparse.ArgumentParser(description="Run neural graphics primitives testbed with additional configuration & output options")

    parser.add_argument("--scene", "--training_data", default="", help="The scene to load. Can be the scene's name or a full path to the training data.")
    parser.add_argument("--mode", default="", const="nerf", nargs="?", choices=["nerf", "sdf", "image", "volume"], help="Mode can be 'nerf', 'sdf', 'image' or 'volume'. Inferred from the scene if unspecified.")
    parser.add_argument("--network", default="", help="Path to the network config. Uses the scene's default if unspecified.")

    parser.add_argument("--load_snapshot", default=f"{NERF_DIR}/transforms_out_optimized_base.msgpack", help="Load this snapshot before training. recommended extension: .msgpack")
    # parser.add_argument("--load_snapshot", default="/home/user/src/ClearFields_Dishwasher/data/dishwasher2/COLMAP/base.msgpack", help="Load this snapshot before training. recommended extension: .msgpack")
    # parser.add_argument("--load_snapshot", default="/home/user/src/ClearFields_Dishwasher/data/dishwasher/base.msgpack", help="Load this snapshot before training. recommended extension: .msgpack")
    parser.add_argument("--save_snapshot", default="", help="Save this snapshot after training. recommended extension: .msgpack")

    parser.add_argument("--exposure", default=0.0, type=float, help="Controls the brightness of the image. Positive numbers increase brightness, negative numbers decrease it.")

    parser.add_argument("--video_spp", type=int, default=64, help="Number of samples per pixel. A larger number means less noise, but slower rendering.")

    parser.add_argument("--save_mesh", default="", help="Output a marching-cubes based mesh from the NeRF or SDF model. Supports OBJ and PLY format.")
    parser.add_argument("--marching_cubes_res", default=256, type=int, help="Sets the resolution for the marching cubes grid.")

    parser.add_argument("--width", "--screenshot_w", type=int, default=640, help="Resolution width of GUI and screenshots.")
    parser.add_argument("--height", "--screenshot_h", type=int, default=480, help="Resolution height of GUI and screenshots.")

    parser.add_argument("--gui", action="store_true", help="Run the testbed GUI interactively.")
    parser.add_argument("--simulation", action="store_true", help="Run simulation.")
    parser.add_argument("--train", action="store_true", help="If the GUI is enabled, controls whether training starts immediately.")
    parser.add_argument("--second_window", action="store_true", help="Open a second window containing a copy of the main output.")

    parser.add_argument("--sharpen", default=0, help="Set amount of sharpening applied to NeRF training images. Range 0.0 to 1.0.")
    parser.add_argument("--start_idx", default=-1, type=int, help="Start index")
    parser.add_argument("--finish_idx", default=10000, type=int, help="Finish index")
    parser.add_argument("--num", default=0, type=int, help="Number of subsequent samples")
    parser.add_argument("--indices", nargs="+", type=int)

    # 5.562886598 for original dishwasher, 0.526823178 for dishwasher 2
    parser.add_argument("--depth_scale", type=float, required=True)

    parser.add_argument("--experiment", type=str, required=True)

    parser.add_argument("--output_folder", type=str, required=True)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    os.makedirs(f'{args.output_folder}/segmentation', exist_ok=True)
    os.makedirs(f'{args.output_folder}/color', exist_ok=True)
    os.makedirs(f'{args.output_folder}/nerf_color_training_frame', exist_ok=True)
    os.makedirs(f'{args.output_folder}/depth_mesh', exist_ok=True)
    os.makedirs(f'{args.output_folder}/depth_scene', exist_ok=True)
    os.makedirs(f'{args.output_folder}/depth_merged', exist_ok=True)
    # os.makedirs(f'{args.output_folder}/camera_matrix', exist_ok=True)

    if not args.indices:
        if args.start_idx == -1:
            segmentations = sorted(os.listdir(f'{args.output_folder}/segmentation'))
            start_idx = int(segmentations[-1][:4]) if len(segmentations) > 0 else 0
        else:
            start_idx = args.start_idx

        finish_idx = start_idx + args.num if args.num > 0 else args.finish_idx

        enumerator = range(start_idx, finish_idx)
    else:
        enumerator = args.indices

    for i in enumerator:
        mode = ngp.TestbedMode.Nerf
        configs_dir = os.path.join(ROOT_DIR, "configs", "nerf")
        scenes = scenes_nerf

        network = os.path.join(configs_dir, "base.json")
        if not os.path.isabs(network):
            network = os.path.join(configs_dir, network)

        testbed = ngp.Testbed(mode)
        testbed.init_rt(config_path=f'scripts/{args.experiment}/{i}/exp.json')
        print('test')

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
        testbed.hybrid_render = 1
        testbed.depth_scale = args.depth_scale
        n_frames = 1
        resolution = [args.width or 1920, args.height or 1080]

        testbed.rt_depth = True

        if "tmp" not in os.listdir():
            os.makedirs("tmp")
        
        # indices = np.concatenate((np.arange(250), np.arange(344, 529), np.arange(576, 959-85)))
        indices = np.arange(testbed.nerf.training.dataset.n_images)

        training_views = sorted(os.listdir(f"{NERF_DIR}/images"))

        while True:
            # view_idx = np.random.choice(indices)
            view_idx = 0
            testbed.set_camera_to_training_view(view_idx)

            # random_rotvec = R.random().as_rotvec() * 0.05
            # random_rotvec -= 0.025 * np.pi
            # rot = np.matmul(R.from_rotvec(random_rotvec).as_matrix(), testbed.camera_matrix[:3, :3])
            # T = testbed.camera_matrix.copy()
            # T[:3, :3] = rot
            # random_t = np.random.rand(3)
            # T[:3, 3] += ((random_t * 0.01) - 0.005)
            # testbed.camera_matrix = T

            testbed.render_mode = testbed.render_mode.MeshSegmentation
            frame = testbed.render(resolution[0], resolution[1], 1, True)

            if not frame[:, :, :3].any():
                print("No results, retrying")
                continue
            cv2.imwrite(f"{args.output_folder}/segmentation/{i:04d}.png", np.uint8(frame[:, :, :3] * 255))
            print('Wrote segmentation')

            testbed.hybrid_render = 1
            testbed.render_mode = testbed.render_mode.MeshDepth
            frame = testbed.render(resolution[0], resolution[1], 1, True)
            depth_raw = frame[..., 0]
            depth_int = 1000 * depth_raw
            if np.any(depth_int > 3000):
                print("Depth too far away. Retry.")
                continue
            cv2.imwrite(f"{args.output_folder}/depth_mesh/{i:04d}.png", np.uint16(depth_int))
            print('Wrote depth')

            testbed.hybrid_render = 1
            testbed.render_mode = testbed.render_mode.Shade
            frame = testbed.render(resolution[0], resolution[1], args.video_spp, True)
            write_image(f"{args.output_folder}/color/{i:04d}.jpg", np.clip(frame * 2**args.exposure, 0.0, 1.0), quality=100)

            relative_color_training_frame_path = testbed.nerf.training.dataset.paths[view_idx]
            path_parts = relative_color_training_frame_path.split('/') 
            shutil.copy(f"{NERF_DIR}/images/{path_parts[-1]}", f"{args.output_folder}/nerf_color_training_frame/{i:04d}.png")

            testbed.hybrid_render = 0
            testbed.render_mode = testbed.render_mode.Depth
            frame = testbed.render(resolution[0], resolution[1], 8, True)
            depth_raw = frame[..., 0]
            depth_int = 1000 * depth_raw
            cv2.imwrite(f"{args.output_folder}/depth_scene/{i:04d}.png", np.uint16(depth_int))

            # json.dump({'T': T.tolist()}, open(f"{args.output_folder}/camera_matrix/{i:04d}.json", 'w'))
            break
    print('Finished at', time.strftime("%m/%d/%Y, %H:%M:%S"))
