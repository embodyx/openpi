from openpi_client import image_tools
from openpi_client import websocket_client_policy
import numpy as np
from PIL import Image
import sys
from pathlib import Path
from typing import Dict, List, Union
from dataclasses import dataclass, field
import time
import draccus
import cv2



bridge_path = Path.home() / "CyberLimb" / "openvla"
sys.path.append(str(bridge_path))
print("bridge path exist: ", bridge_path.exists(), bridge_path)

from experiments.robot.bridge.bridgev2_utils import (
    get_next_task_label,
    get_preprocessed_image,
    get_widowx_env,
    refresh_obs,
    save_rollout_data,
    save_rollout_video,
)

from experiments.robot.robot_utils import (
    get_image_resize_size,
)

@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                               # Model family
    pretrained_checkpoint: Union[str, Path] = ""                # Pretrained checkpoint path
    load_in_8bit: bool = False                                  # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                                  # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = False                                   # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # WidowX environment-specific parameters
    #################################################################################################################
    host_ip: str = "localhost"
    port: int = 5556

    # Note: Setting initial orientation with a 30 degree offset, which makes the robot appear more natural
    init_ee_pos: List[float] = field(default_factory=lambda: [0.3, -0.09, 0.26])
    init_ee_quat: List[float] = field(default_factory=lambda: [0, -0.259, 0, -0.966])
    bounds: List[List[float]] = field(default_factory=lambda: [
            [0.05, -0.25, -0.02, -1.57, 0],
            [0.50, 0.30, 0.35, 1.57, 0],
        ]
    )

    camera_topics: List[Dict[str, str]] = field(default_factory=lambda: [{"name": "/camera/camera_stream/color/image_raw", "dtype": "rgb8"}])

    blocking: bool = False                                      # Whether to use blocking control
    max_episodes: int = 50                                      # Max number of episodes to run
    max_steps: int = 240                                         # Max number of timesteps per episode
    control_frequency: float = 0.5                               # WidowX control frequency (much slower for smooth motion)

    #################################################################################################################
    # Utils
    #################################################################################################################
    save_data: bool = True                                     # Whether to save rollout data (images, actions, etc.)

    # fmt: on

    # Cloud configuration
    # Set your cloud endpoint here
    CLOUD_URL = "http://0.0.0.0:8000/predict"  # or https://... if behind TLS


# VM IP where the server is running
VM_IP = "34.31.151.48"
PORT = 8000

# Outside of episode loop, initialize the policy client.
# Point to the host and port of the policy server (localhost and 8000 are the defaults).
client = websocket_client_policy.WebsocketClientPolicy(host=VM_IP, port=PORT)

img = Image.open("/home/nick/vote/example.png")

# img.show()

img_array_example = np.array(img)

# for step in range(1):
#     # Inside the episode loop, construct the observation.
#     # Resize images on the client side to minimize bandwidth / latency. Always return images in uint8 format.
#     # We provide utilities for resizing images + uint8 conversion so you match the training routines.
#     # The typical resize_size for pre-trained pi0 models is 224.
#     # Note that the proprioceptive `state` can be passed unnormalized, normalization will be handled on the server side.
#     observation = {
#         "observation/image": image_tools.convert_to_uint8(
#             image_tools.resize_with_pad(img_array, 224, 224)
#         ),
#         # "observation/wrist_image": image_tools.convert_to_uint8(
#         #     image_tools.resize_with_pad(wrist_img, 224, 224)
#         # ),
#         "observation/state": np.random.uniform(-1, 1, size=7),
#         "prompt": "pick up the eggplant",
#     }

#     # Call the policy server with the current observation.
#     # This returns an action chunk of shape (action_horizon, action_dim).
#     # Note that you typically only need to call the policy every N steps and execute steps
#     # from the predicted action chunk open-loop in the remaining steps.
#     action_chunk = client.infer(observation)["actions"]

#     # Execute the actions in the environment.
#     print(action_chunk)

def send_to_cloud_and_get_response(img_array, state, prompt):
    # if obs["full_image"] is not None:
    #     main_image = obs["full_image"]
    #     # Encode the image as JPEG
    #     ok, buf = cv2.imencode(".jpg", main_image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    #     if not ok:
    #         raise RuntimeError("JPEG encode failed")
    #     main_image_jpeg_bytes = buf.tobytes()
    # if obs["wrist_image"] is not None:
    #     wrist_image = obs["full_image"]
    #     # Encode the image as JPEG
    #     ok, buf = cv2.imencode(".jpg", wrist_image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    #     if not ok:
    #         raise RuntimeError("JPEG encode failed")
    #     wrist_image_jpeg_bytes = buf.tobytes()
    # # POST raw JPEG
    # r = requests.post(cfg.CLOUD_URL, data=main_image_jpeg_bytes, headers={"Content-Type": "image/jpeg"}, timeout=5)
    # r.raise_for_status()
    # return r.json()

    observation = {
        "observation/image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img_array_example, 224, 224)
        ),
        # "observation/wrist_image": image_tools.convert_to_uint8(
        #     image_tools.resize_with_pad(wrist_img, 224, 224)
        # ),
        "observation/state": np.random.uniform(-1, 1, size=7),
        "prompt": "pick up the eggplant",
    }


    # observation = {
    #     "observation/image": image_tools.convert_to_uint8(
    #         image_tools.resize_with_pad(img_array, 224, 224)
    #     ),
    #     # "observation/wrist_image": image_tools.convert_to_uint8(
    #     #     image_tools.resize_with_pad(wrist_img, 224, 224)
    #     # ),
    #     "observation/state": state,
    #     "prompt": prompt,
    # }

    # Call the policy server with the current observation.
    # This returns an action chunk of shape (action_horizon, action_dim).
    # Note that you typically only need to call the policy every N steps and execute steps
    # from the predicted action chunk open-loop in the remaining steps.
    action_chunk = client.infer(observation)["actions"]

    return action_chunk

@draccus.wrap()
def eval_model_in_bridge_env(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    assert not cfg.center_crop, "`center_crop` should be disabled for Bridge evaluations!"

    # Initialize the wrist cam
    wrist_cam = "/dev/v4l/by-id/usb-SONix_Technology_Co.__Ltd._Streaming_Camera_SN0001-video-index0" 
    cap = cv2.VideoCapture(wrist_cam, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)   # manual
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  

    # Initialize the WidowX environment
    env = get_widowx_env(cfg)

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)

    # Start evaluation
    task_label = ""
    episode_idx = 0
    while episode_idx < 1:
        # Get task description from user
        task_label = get_next_task_label(task_label)

        # Reset environment
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, _ = reset_result
        else:
            obs = reset_result

        # Setup
        t = 0
        step_duration = 1.0 / cfg.control_frequency
        replay_images = []
        replay_images_wrist = []
        episode_start_time = None  # Timer for episode elapsed time from first inference
        if cfg.save_data:
            rollout_images = []
            rollout_wrist_images = []
            rollout_states = []
            rollout_actions = []

        # Start episode
        print(f"Starting episode {episode_idx+1}...")
        print("Episode running... Press Ctrl-C to terminate episode early!")
        last_tstamp = time.time()
        while t < 1:
            try:
                curr_tstamp = time.time()
                if curr_tstamp > last_tstamp + step_duration:
                    print(f"t: {t}")
                    print(f"Previous step elapsed time (sec): {curr_tstamp - last_tstamp:.2f}")
                    last_tstamp = time.time()

                    # Refresh the camera image and proprioceptive state
                    obs = refresh_obs(obs, env)

                    # Also get the wrist camera image
                    if cap is not None:
                        ok, bgr = cap.read()
                        # cap.release()
                        if not ok or bgr is None:
                            raise RuntimeError("capture failed")
                        # Convert to rgb
                        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    
                        # Add the wrist camera to the observation
                        obs['wrist_image'] = rgb

                    # Save full (not preprocessed) image for replay video
                    replay_images.append(obs["full_image"])
                    replay_images_wrist.append(obs["wrist_image"])

                    # # # Get preprocessed image
                    # obs = get_preprocessed_image(obs, resize_size)
                    # if obs["full_image"] is not None:
                    #     preprocessed_main_image = obs["full_image"]
                    # if obs["wrist_image"] is not None:
                    #     preprocessed_wrist_image = obs["wrist_image"]

                    if cfg.save_data:
                        rollout_images.append(obs["full_image"])
                        rollout_wrist_images.append(obs["wrist_image"])
                        rollout_states.append(obs['proprio'])
                    
                    # Start episode timer right before first inference
                    if episode_start_time is None:
                        episode_start_time = time.time()
                        print("Starting episode timer for inference tracking...")

                    # Send to cloud for inference
                    action_chunk = send_to_cloud_and_get_response(obs["full_image"], obs['proprio'], task_label)
                    selected_action = np.array(action_chunk[:10])

                    if cfg.save_data:
                        rollout_actions.append(selected_action.tolist())

                    # # Execute action
                    # action[:6] *= 1
                    # print("action:", action.tolist())
                    # step_result = env.step(action)
                    # if len(step_result) == 4:
                    #     obs, _, _, _ = step_result
                    # else:
                    #     obs, _, _, _, _ = step_result
                    t += 1

                    # Print elapsed time since first inference
                    if episode_start_time is not None:
                        elapsed_time = time.time() - episode_start_time
                        print(f"Episode elapsed time since first inference: {elapsed_time:.2f} sec")

            except (KeyboardInterrupt, Exception) as e:
                if isinstance(e, KeyboardInterrupt):
                    print("\nCaught KeyboardInterrupt: Terminating episode early.")
                else:
                    print(f"\nCaught exception: {e}")
                break

        # Save a replay video of the episode
        save_rollout_video(replay_images, replay_images_wrist, episode_idx)

        # [If saving rollout data] Save rollout data
        if cfg.save_data:
            save_rollout_data(replay_images, rollout_images, rollout_states, rollout_actions, idx=episode_idx)

        # Continue to next episode automatically
        episode_idx += 1

if __name__ == "__main__":
    eval_model_in_bridge_env() 
