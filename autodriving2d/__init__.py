from gymnasium.envs.registration import register

register(
    id="autodriving2d/CityDrive-v0",
    entry_point="autodriving2d.envs:CityDrive",
)