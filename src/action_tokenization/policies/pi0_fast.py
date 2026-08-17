"""LeRobot pi0-FAST policy — re-exports with no modifications.

Usage:
    from action_tokenization.policies import PI0FastPolicy, PI0FastConfig

    config = PI0FastConfig(...)
    policy = PI0FastPolicy(config)
    # or
    policy = PI0FastPolicy.from_pretrained("lerobot/pi0fast-base")
"""

from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig
from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy

__all__ = ["PI0FastConfig", "PI0FastPolicy"]
