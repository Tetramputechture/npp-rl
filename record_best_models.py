import imageio
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from pathlib import Path


def create_env(render_mode: str = 'rgb_array') -> VecNormalize:
    """Create a vectorized environment for training or evaluation."""
    env = DummyVecEnv([lambda: BasicLevelNoGold(
        render_mode=render_mode, enable_frame_stack=False)])

    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    return env


def record_video(env: BasicLevelNoGold, policy: PPO, video_path, num_episodes=1):
    """Record a video of the trained agent playing."""
    images = []
    for _ in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            images.append(env.render())
            action, _ = policy.predict(state, deterministic=True)
            action = action.item()
            state, _, done, _ = env.step(action)

    # Save video
    imageio.mimsave(video_path, [np.array(img) for img in images], fps=30)


BEST_MODELS_PATH = Path('./training_logs/tune_logs/')
BASE_VIDEOS_PATH = Path('./videos/')

# Our best models are located in folders that match the pattern "best_model_..."
# the folder will contain a best_model.zip file.
# For each of these folders, we want to record a video of the agent playing.

env = create_env()

# Create a new directory for the videos
BASE_VIDEOS_PATH.mkdir(parents=True, exist_ok=True)

for folder in BEST_MODELS_PATH.iterdir():
    if folder.is_dir() and folder.name.startswith("best_model_"):
        # Check if the best_model.zip file exists
        if not (folder / "best_model.zip").exists():
            print(
                f"Skipping folder {folder} because best_model.zip does not exist")
            continue
        print(f"Processing folder: {folder}")
        # Load the model
        model = PPO.load(folder / "best_model", env=env)
        video_path = BASE_VIDEOS_PATH / folder.name / "video.mp4"
        # Record a video of the agent playing
        record_video(env, model, video_path)
