'''
demonstrate trained network
-h for help
'''
import argparse
import torch
import gymnasium as gym
import Game2048Env

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate trained model for 2048 game')
    parser.add_argument('path', type=str, help='path to trained model')
    parser.add_argument(
        '-r',
        '--render-mode',
        type=str,
        default='human',
        choices=['human', 'ansi'],
        help='render mode for demo'
    )
    args = parser.parse_args()
    model = torch.load(args.path)
    model.eval()

    env = gym.make('Game2048Env/Game2048-v0', render_mode=args.render_mode)
    state, _ = env.reset()
    state = torch.tensor(state.flatten(), dtype=torch.float32, device=device).unsqueeze(0)

    while True:
        with torch.no_grad():
            action = model(state).max(1).indices.view(1, 1)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        if terminated or truncated:
            break
        state = torch.tensor(
            observation.flatten(),
            dtype=torch.float32,
            device=device
        ).unsqueeze(0)
        if args.render_mode == 'ansi':
            print(env.render())
