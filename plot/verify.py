import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

class CropObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        orig_shape = env.observation_space.shape
        if len(orig_shape) == 2:
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(orig_shape[0]//2, orig_shape[1]),
                dtype=np.uint8
            )
        else:
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(orig_shape[0]//2, orig_shape[1], orig_shape[2]),
                dtype=np.uint8
            )

    def observation(self, obs):
        height = obs.shape[0]
        if len(obs.shape) == 2:
            return obs[height//2:, :]
        else:
            return obs[height//2:, :, :]

def visualize_frame_processing():
    # Create environment with all wrappers
    env = gym.make("ALE/SpaceInvaders-v5", render_mode="rgb_array")
    
    # Store frames after each wrapper
    frames = {}
    
    # Original
    obs, _ = env.reset()
    frames['original'] = obs.copy()
    
    # Add wrappers one by one and capture frames
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    
    env = CropObservationWrapper(env)
    obs, _ = env.reset()
    frames['after_crop'] = obs.copy()
    
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    obs, _ = env.reset()
    frames['after_resize'] = obs.copy()
    
    env = gym.wrappers.GrayScaleObservation(env)
    obs, _ = env.reset()
    frames['after_grayscale'] = obs.copy()
    
    env = gym.wrappers.FrameStack(env, 4)
    obs, _ = env.reset()
    frames['final'] = obs[0].copy()  # Take first frame from stack
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Frame Processing Pipeline', fontsize=16)
    
    # Plot frames
    axes[0,0].imshow(frames['original'])
    axes[0,0].set_title(f'Original\n{frames["original"].shape}')
    
    
    axes[0,2].imshow(frames['after_crop'])
    axes[0,2].set_title(f'After Crop\n{frames["after_crop"].shape}')
    
    axes[1,0].imshow(frames['after_resize'])
    axes[1,0].set_title(f'After Resize\n{frames["after_resize"].shape}')
    
    axes[1,1].imshow(frames['after_grayscale'], cmap='gray')
    axes[1,1].set_title(f'After Grayscale\n{frames["after_grayscale"].shape}')
    
    axes[1,2].imshow(frames['final'], cmap='gray')
    axes[1,2].set_title(f'Final (First of 4 Frames)\n{frames["final"].shape}')
    
    # Turn off axes
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print shapes
    print("\nFrame shapes at each stage:")
    for name, frame in frames.items():
        print(f"{name}: {frame.shape}")
    
    return frames

if __name__ == "__main__":
    frames = visualize_frame_processing()