"""YAML configuration, dotted overrides and basic physical validity checks."""

import math
from copy import deepcopy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/hsl_m_3v3.yaml"


def load_config(path=DEFAULT_CONFIG, overrides=()):
    path = Path(path).resolve()

    def read(current, seen):
        if current in seen:
            raise ValueError("Configuration inheritance cycle")
        data = yaml.safe_load(current.read_text())
        parent = data.pop("extends", None)
        if parent is None:
            return data
        base = read((current.parent / parent).resolve(), seen | {current})

        def merge(left, right):
            for key, value in right.items():
                if (
                    key in left
                    and isinstance(left[key], dict)
                    and isinstance(value, dict)
                ):
                    merge(left[key], value)
                else:
                    left[key] = value

        merge(base, data)
        return base

    cfg = read(path, set())
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Override must be dotted.key=value: {item}")
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node[part]
        if parts[-1] not in node:
            raise KeyError(f"Unknown parameter: {key}")
        node[parts[-1]] = yaml.safe_load(value)
    validate(cfg)
    return deepcopy(cfg)


def validate(c):
    def positive(v, key):
        if not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0:
            raise ValueError(f"{key} must be finite and positive")

    for group, keys in {
        "field": [
            "length",
            "width",
            "line_width",
            "runoff",
            "goal_area_depth",
            "goal_area_width",
            "penalty_area_depth",
            "penalty_area_width",
            "center_circle_radius",
        ],
        "goal": ["width", "height", "depth", "post_diameter"],
        "ball": ["radius", "mass", "inertia_factor"],
        "physics": ["timestep"],
        "robot": [
            "kp",
            "kd",
            "base_height",
            "mass_scale",
            "inertia_scale",
            "torque_scale",
        ],
    }.items():
        for key in keys:
            positive(c[group][key], f"{group}.{key}")
    f, g = c["field"], c["goal"]
    if f["width"] >= f["length"]:
        raise ValueError("field.length must exceed field.width")
    if not (g["width"] < f["goal_area_width"] <= f["penalty_area_width"] < f["width"]):
        raise ValueError("Expected goal < goal area <= penalty area < field width")
    if not (f["goal_area_depth"] < f["penalty_area_depth"] < f["length"] / 2):
        raise ValueError("Invalid area depths")
    if f["line_width"] >= min(f["goal_area_depth"], f["center_circle_radius"]):
        raise ValueError("line_width too large")
    if c["robot"]["base_mode"] not in ("fixed", "free"):
        raise ValueError("robot.base_mode must be fixed or free")
    if "pretrained" in c:
        p = c["pretrained"]
        for key in ("release_distance", "kp_scale", "kd_scale", "action_scale"):
            positive(p[key], "pretrained." + key)
        if len(p["command"]) != 3 or not all(math.isfinite(x) for x in p["command"]):
            raise ValueError("pretrained.command needs three finite numbers")
        damping = p["joint_damping"]
        if damping is not None and (not math.isfinite(damping) or damping < 0):
            raise ValueError("pretrained.joint_damping must be nonnegative or null")
    for link, values in c["robot"]["link_overrides"].items():
        if set(values) - {"mass", "diaginertia"}:
            raise ValueError(f"Unsupported inertial override for {link}")
        if "mass" in values:
            positive(values["mass"], link + ".mass")
        if "diaginertia" in values:
            inertia = values["diaginertia"]
            if len(inertia) != 3:
                raise ValueError("diaginertia needs three components")
            for v in inertia:
                positive(v, link + ".diaginertia")
            if max(inertia) > sum(inertia) - max(inertia):
                raise ValueError("Inertia violates triangle inequality")
    if c["physics"]["timestep"] > 0.005:
        raise ValueError("Use a timestep <= 0.005 for foot-ball contact")
    for name, contact in c["contacts"].items():
        fr = contact["friction"]
        if len(fr) != 5 or any(not math.isfinite(x) or x < 0 for x in fr):
            raise ValueError(
                f"contacts.{name}.friction requires five nonnegative numbers"
            )
        if contact["condim"] not in (1, 3, 4, 6):
            raise ValueError("condim must be 1, 3, 4 or 6")
        if len(contact["solref"]) != 2 or any(x <= 0 for x in contact["solref"]):
            raise ValueError("This baseline uses positive timeconst/dampratio solref")
    names = [r["name"] for r in c["robot"]["instances"]]
    if len(set(names)) != len(names):
        raise ValueError("Robot names must be unique")
    p = c["passing"]
    if p["sender"] == p["receiver"] or any(
        p[k] not in names for k in ("sender", "receiver")
    ):
        raise ValueError("Choose distinct existing sender/receiver")
