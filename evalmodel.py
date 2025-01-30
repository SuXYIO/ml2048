import argparse
import torch
import gymnasium as gym
import Game2048Env

def evaluate_model(env, model, num_episodes=100):
    total_rewards = []
    
    for _ in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state.flatten(), dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0
        
        while True:
            with torch.no_grad():
                action = model(state).max(1).indices.view(1, 1)
            
            observation, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            
            if terminated or truncated:
                break
            
            state = torch.tensor(observation.flatten(), dtype=torch.float32, device=device).unsqueeze(0)  # Update state
        
        total_rewards.append(total_reward)
    
    average_reward = sum(total_rewards) / num_episodes
    print(f'Average Reward over {num_episodes} episodes: {average_reward}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate trained model for 2048 game')
    parser.add_argument('path', type=str, help='path to trained model')
    parser.add_argument('episodes', type=int, help='number of episodes to eval')
    args = parser.parse_args()

    policy_net = torch.load(args.path)
    policy_net.eval()

    env = gym.make('Game2048Env/Game2048-v0')

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )

    evaluate_model(env, policy_net, num_episodes=args.episodes)
