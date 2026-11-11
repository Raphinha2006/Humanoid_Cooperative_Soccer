"""White-box reward probes on constructed states, not physical rollouts."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import torch

from soccer_sim.gpu_receive.model import goal, reward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    states = torch.zeros(4, 160, device="cpu")
    states[:, 2] = states[:, 77] = 0.75
    states[:, 0] = 5.0
    states[:, 6] = 1.0  # Sender faces receiver along -x.
    states[:, 78] = 1.0  # Receiver faces sender along +x.
    states[:, 150] = 0.35
    states[:, 152] = 0.11
    states[2, 78] = 0.0
    states[2, 81] = 1.0  # Receiver instead faces away from sender.
    states[2, 150] = -0.35  # Keep the ball at the same body-relative target.
    states[3, 153] = -1.0  # Incoming ball at the target, still moving.
    delta = torch.zeros(4, 12)
    with torch.no_grad():
        scores = reward(states, delta, delta).tolist()
        relative = goal(states)[:, :3].tolist()
    labels = ["front_stationary_contact_history", "same_state_no_contact_history",
              "facing_away_stationary_body_front", "front_incoming_1mps"]
    path = Path(inspect.getfile(reward))
    result = {
        "scope": "Constructed-state algebraic diagnostics only; no simulation or trained-policy evaluation",
        "reward_module_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "device": "cpu",
        "probes": [dict(name=name, instantaneous_reward=score, ball_receiver_frame=rel)
                   for name, score, rel in zip(labels, scores, relative)],
        "interpretation": [
            "Contact history is not an input: identical physical states with different histories receive equal reward.",
            "Facing away can receive the same reward when the ball is at the receiver-relative target.",
            "These probes expose missing task distinctions, not evidence of physical feasibility or actual policy exploitation.",
            "Compare complete discounted real trajectories before selecting new reward weights.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"probe_rewards": dict(zip(labels, scores)), "gpu_used": False}))


if __name__ == "__main__":
    main()
