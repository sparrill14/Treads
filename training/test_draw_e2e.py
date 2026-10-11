"""A mutual bomb kill must terminate the Gym episode immediately."""

from runtime_codec import decode_multi_discrete_action
from model_contract import NEUTRAL_AIM_BIN
from treads_env import TreadsEnv


def main() -> None:
    env = TreadsEnv(level=111, max_episode_steps=1800, player_max_ammo=0, player_max_bombs=2)
    try:
        env.reset(seed=42)
        # Move beside the unarmed opponent and remain inside two delayed blasts.
        for tick in range(700):
            action = decode_multi_discrete_action(
                [3 if tick < 180 else 0, NEUTRAL_AIM_BIN, 0, 1 if tick in (180, 310) else 0],
                env._last_obs_raw,
            )
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                assert terminated and not truncated
                assert info["result"]["status"] == "draw", info
                assert not info["result"]["timeout"]
                assert tick < 700
                break
        else:
            raise AssertionError("Mutual kill did not end the episode")
    finally:
        env.close()
    print("Mutual bomb kill is terminal, with no timeout bootstrap")


if __name__ == "__main__":
    main()
