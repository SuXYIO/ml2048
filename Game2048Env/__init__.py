from gymnasium.envs.registration import register

register(
    id="Game2048Env/Game2048-v0",
    entry_point='Game2048Env.envs:Game2048Env',
)
