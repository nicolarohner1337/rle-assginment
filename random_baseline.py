import gymnasium as gym
import numpy as np
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from torch.utils.tensorboard import SummaryWriter
import time

def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)

        env.action_space.seed(seed)
        return env

    return thunk

def evaluate_random_agent(env_id, num_episodes=1000000, seed=1):
    run_name = f"SpaceInvadersrandom_baseline{seed}_{time.strftime('%Y%m%d%H%M%S')}"
    writer = SummaryWriter(f"runs/{run_name}")

    # Create environment
    envs = gym.vector.SyncVectorEnv([
        make_env(env_id, seed, 0, True, run_name)
    ])

    # Lists to store metrics
    episode_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        obs,  = envs.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            # Random action
            action = np.array([envs.single_action_space.sample()])

            # Step environment
            obs, reward, terminations, truncations, info = envs.step(action)
            done = terminations[0] or truncations[0]
            episode_reward += reward[0]
            episode_length += 1

        # Log episode results
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        writer.add_scalar("charts/episodic_return", episode_reward, episode)
        writer.add_scalar("charts/episodic_length", episode_length, episode)

    # Log summary statistics
    writer.add_scalar("eval/episodic_return_mean", np.mean(episode_rewards), 0)
    writer.add_scalar("eval/episodic_return_std", np.std(episode_rewards), 0)
    writer.add_scalar("eval/episodic_return_min", np.min(episode_rewards), 0)
    writer.add_scalar("eval/episodic_return_max", np.max(episode_rewards), 0)

    envs.close()
    writer.close()

    return episode_rewards, episode_lengths

if __name__ == "__main__":
    rewards, lengths = evaluate_random_agent(
        env_id="ALE/SpaceInvaders-v5",
        num_episodes=100,
        seed=1
    )

    print("\nBaseline Results:")
    print(f"Mean Return: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Min Return: {np.min(rewards):.2f}")
    print(f"Max Return: {np.max(rewards):.2f}")
    print(f"Mean Episode Length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f}")
