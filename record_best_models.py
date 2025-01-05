import imageio
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, VecMonitor, VecCheckNan
from nclone_environments.basic_level_no_gold.basic_level_no_gold import BasicLevelNoGold
from pathlib import Path


def create_env(render_mode: str = 'rgb_array') -> VecNormalize:
    """Create a vectorized environment for training or evaluation."""
    env = DummyVecEnv([lambda: BasicLevelNoGold(
        render_mode=render_mode, enable_frame_stack=False, enable_animation=True)])

    env = VecMonitor(env)
    env = VecCheckNan(env, raise_exception=True)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    return env


def action_to_string(action: int) -> str:
    """Convert action index to human-readable string."""
    action_names = {
        0: 'NOOP',
        1: 'Left',
        2: 'Right',
        3: 'Jump',
        4: 'Jump + Left',
        5: 'Jump + Right'
    }
    return action_names.get(action, 'Unknown')


def record_video(env: BasicLevelNoGold, policy: PPO, video_path, actions_path, num_episodes=1):
    """Record a video of the trained agent playing."""
    images = []
    actions = []
    for _ in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            images.append(env.render())
            actions, _ = policy.predict(state, deterministic=True)
            actions.append(actions[0])
            state, _, done, _ = env.step(actions)

    # Save actions as their string representation
    actions_path.write_text(
        ",".join([action_to_string(action) for action in actions]))

    # Save video
    imageio.mimsave(video_path, [np.array(img) for img in images], fps=30)


BEST_MODELS_PATH = Path('./training_logs/tune_logs/')
BASE_VIDEOS_PATH = Path('./videos/')
ACTIONS_PATH = Path('./actions/')

# Our best models are located in folders that match the pattern "best_model_..."
# the folder will contain a best_model.zip file.
# For each of these folders, we want to record a video of the agent playing.

if __name__ == "__main__":
    env = create_env()

    # Create a new directory for the videos
    BASE_VIDEOS_PATH.mkdir(parents=True, exist_ok=True)
    # Create a new directory for the actions
    ACTIONS_PATH.mkdir(parents=True, exist_ok=True)

    for folder, idx in enumerate(BEST_MODELS_PATH.iterdir()):
        if folder.is_dir() and folder.name.startswith("best_model_"):
            # Check if the best_model.zip file exists
            if not (folder / "best_model.zip").exists():
                print(
                    f"Skipping folder {folder} because best_model.zip does not exist")
                continue
            print(f"Processing folder: {folder}")
            # Load the model
            model = PPO.load(folder / "best_model")
            folder_name = f"model_{idx}"
            video_path = BASE_VIDEOS_PATH / folder_name / "video.mp4"
            actions_path = ACTIONS_PATH / folder_name / "actions.txt"
            # Create the folder if it doesn't exist
            video_path.parent.mkdir(parents=True, exist_ok=True)
            # Record a video of the agent playing
            record_video(env, model, video_path, actions_path)
