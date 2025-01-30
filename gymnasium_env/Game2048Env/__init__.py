from gymnasium.envs.registration import register

register(
    id="gymnasium_env/Game2048-v0",
    entry_point='gymnasium_env.envs:Game2048Env',
)
