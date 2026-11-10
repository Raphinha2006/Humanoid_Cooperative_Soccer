"""Place upstream floating-base policy models on the same parameterised HSL field."""

import xml.etree.ElementTree as ET
from copy import deepcopy

import mujoco

from .pretrained import POLICIES
from .scene import build_scene, vec

MODEL_PATHS = {
    "booster_mjlab": POLICIES / "booster_mjlab/website/static/demo/k1_web.xml",
    "sonic_k1": POLICIES
    / "sonic_k1/gear_sonic/data/assets/robot_description/mjcf/k1.xml",
}


def build_policy_scene(config, kind, *, native_contacts=False):
    source_path = MODEL_PATHS[kind]
    source = ET.parse(source_path).getroot()
    rc = config["robot"]
    # Some upstream fixed hand-tip bodies infer inertia from geometry; scale those too.
    source_model = (
        mujoco.MjModel.from_xml_path(str(source_path))
        if rc["mass_scale"] != 1 or rc["inertia_scale"] != 1 or rc["link_overrides"]
        else None
    )
    root = ET.fromstring(build_scene(config, robots=False))
    src_default = source.find("default")
    if src_default is not None:
        root.remove(root.find("default"))
        root.append(deepcopy(src_default))
    meshdir = source.find("compiler").get("meshdir", "")
    for asset in source.find("asset"):
        if asset.tag in ("texture", "material") and (
            not asset.get("name") or asset.get("name") in ["texplane", "matplane"]
        ):
            continue
        asset = deepcopy(asset)
        if asset.get("file"):
            asset.set(
                "file",
                str((source_path.parent / meshdir / asset.get("file")).resolve()),
            )
        root.find("asset").append(asset)
    # The original field remains, including its independent ball-ground contact pair.
    world = root.find("worldbody")
    contact = root.find("contact")
    floor = source.find("worldbody/geom")
    if native_contacts and floor is not None:
        for attr in ["friction", "condim", "solref", "solimp", "solmix"]:
            if floor.get(attr) is not None:
                world.find("geom[@name='turf']").set(attr, floor.get(attr))
    actuators = ET.SubElement(root, "actuator")
    sensors = ET.SubElement(root, "sensor")
    source_robot = source.find("worldbody/body")
    for spec in config["robot"]["instances"]:
        prefix = spec["name"] + "/"
        body = deepcopy(source_robot)
        # Prefix named entities and internal references, leaving shared meshes/materials intact.
        mapping = {
            e.get("name"): prefix + e.get("name") for e in body.iter() if e.get("name")
        }
        for e in body.iter():
            if e.get("name"):
                e.set("name", mapping[e.get("name")])
        rc = config["robot"]
        known = {b.get("name")[len(prefix) :] for b in body.iter("body")}
        unknown = set(rc["link_overrides"]) - known
        if unknown:
            raise KeyError(f"Unknown upstream link overrides: {unknown}")
        for link in body.iter("body"):
            inertial = link.find("inertial")
            if inertial is None and source_model is not None:
                original_name = link.get("name")[len(prefix) :]
                original = source_model.body(original_name)
                if original.mass[0] > 0:
                    inertial = ET.SubElement(
                        link,
                        "inertial",
                        mass=str(original.mass[0]),
                        diaginertia=vec(original.inertia),
                        pos=vec(original.ipos),
                        quat=vec(original.iquat),
                    )
            if inertial is not None:
                inertial.set(
                    "mass", str(float(inertial.get("mass")) * rc["mass_scale"])
                )
                inertial.set(
                    "diaginertia",
                    vec(
                        [
                            float(v) * rc["mass_scale"] * rc["inertia_scale"]
                            for v in inertial.get("diaginertia").split()
                        ]
                    ),
                )
                for attr, value in (
                    rc["link_overrides"]
                    .get(link.get("name")[len(prefix) :], {})
                    .items()
                ):
                    inertial.set(
                        attr, vec(value) if isinstance(value, list) else str(value)
                    )
        damping = config["pretrained"]["joint_damping"]
        if damping is not None:
            for joint in body.iter("joint"):
                if joint.get("type") != "free":
                    joint.set("damping", str(damping))
        body.set("pos", vec([*spec["position"], config["robot"]["base_height"]]))
        world.append(body)
        for section, target in [
            ("actuator", actuators),
            ("sensor", sensors),
            ("contact", contact),
        ]:
            src = source.find(section)
            if src is None:
                continue
            for item in src:
                item = deepcopy(item)
                for attr, value in list(item.attrib.items()):
                    if attr == "name":
                        item.set(attr, prefix + value)
                    elif (
                        attr
                        in [
                            "joint",
                            "joint1",
                            "joint2",
                            "body",
                            "body1",
                            "body2",
                            "site",
                            "site1",
                            "site2",
                            "objname",
                            "geom1",
                            "geom2",
                        ]
                        and value in mapping
                    ):
                        item.set(attr, mapping[value])
                target.append(item)
        # Give every unnamed collision geometry a stable ID; preserve native foot-ground dynamics.
        for index, g in enumerate(body.iter("geom")):
            if not g.get("name"):
                g.set("name", prefix + f"geom_{index}")
            if g.get("name") == prefix + "trunk_collision" or (
                g.get("mesh") == "Trunk" and g.get("contype") == "0"
            ):
                g.set(
                    "rgba",
                    ".12 .48 .95 1" if spec["team"] == "blue" else ".94 .19 .17 1",
                )
            if g.get("contype") == "0" and g.get("conaffinity") == "0":
                continue
            if ("foot" in g.get("name", "").lower() and g.get("type") == "mesh") or (
                g.get("type") == "box" and g.get("size") == "0.09 0.035 0.018"
            ):
                for kind2, g2 in [("ball_foot", "ball_geom")] + (
                    [] if native_contacts else [("foot_turf", "turf")]
                ):
                    d = config["contacts"][kind2]
                    ET.SubElement(
                        contact,
                        "pair",
                        name=prefix + kind2 + f"_{index}",
                        geom1=g.get("name"),
                        geom2=g2,
                        condim=str(d["condim"]),
                        friction=vec(d["friction"]),
                        solref=vec(d["solref"]),
                        solimp=vec(d["solimp"]),
                    )
    return ET.tostring(root, encoding="unicode")
