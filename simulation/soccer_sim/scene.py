"""Construct MJCF with outer-edge field dimensions and shared official meshes."""

import math
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

from .config import ROOT


def vec(values):
    return " ".join(f"{float(x):.10g}" for x in values)


def build_scene(c, *, robots=True):
    root = ET.Element("mujoco", model=c["profile"])
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    p = c["physics"]
    ET.SubElement(
        root,
        "option",
        timestep=str(p["timestep"]),
        gravity=vec(p["gravity"]),
        integrator=p["integrator"],
        solver=p["solver"],
        iterations=str(p["iterations"]),
        tolerance=str(p["tolerance"]),
        cone=p["cone"],
    )
    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "global",
        offwidth=str(c["render"]["width"]),
        offheight=str(c["render"]["height"]),
    )
    ET.SubElement(visual, "quality", shadowsize="2048")
    default = ET.SubElement(root, "default")
    ET.SubElement(
        default, "geom", friction=".6 .002 .0001", solref=".012 1", solimp=".9 .95 .001"
    )
    asset, world = ET.SubElement(root, "asset"), ET.SubElement(root, "worldbody")
    ET.SubElement(
        asset,
        "texture",
        type="skybox",
        builtin="gradient",
        rgb1=".15 .23 .32",
        rgb2=".03 .05 .08",
        width="512",
        height="512",
    )
    ET.SubElement(
        world,
        "light",
        pos="0 0 8",
        dir="0 0 -1",
        directional="true",
        diffuse=".8 .8 .8",
    )
    ET.SubElement(
        world,
        "light",
        pos="-4 -5 4",
        dir="1 1 -1",
        diffuse=".45 .45 .45",
        castshadow="false",
    )
    f, g = c["field"], c["goal"]
    L, W, w = f["length"], f["width"], f["line_width"]
    ET.SubElement(
        world,
        "geom",
        name="turf",
        type="plane",
        size=vec([L / 2 + f["runoff"], W / 2 + f["runoff"], 0.1]),
        rgba=".075 .30 .14 1",
        contype="1",
        conaffinity="65535",
    )
    # Stripes are visual only; the entire effective grass contact plane is z=0.
    for i in range(14):
        ET.SubElement(
            world,
            "geom",
            name=f"stripe_{i}",
            type="box",
            pos=vec([-L / 2 + (i + 0.5) * L / 14, 0, -0.0008]),
            size=vec([L / 28, W / 2, 0.001]),
            rgba=(".10 .35 .17 1" if i % 2 else ".085 .31 .15 1"),
            contype="0",
            conaffinity="0",
        )
    verts, faces = [], []

    def quad(points):
        n = len(verts)
        verts.extend([[x, y, 0.001] for x, y in points])
        faces.extend([[n, n + 1, n + 2], [n, n + 2, n + 3]])

    def rect(x0, x1, y0, y1):
        quad([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])

    def arc(cx, cy, outer, start=0.0, end=2 * math.pi, segments=160):
        inner = max(0, outer - w)
        for i in range(segments):
            a, b = (
                start + (end - start) * i / segments,
                start + (end - start) * (i + 1) / segments,
            )
            quad(
                [
                    (cx + inner * math.cos(a), cy + inner * math.sin(a)),
                    (cx + outer * math.cos(a), cy + outer * math.sin(a)),
                    (cx + outer * math.cos(b), cy + outer * math.sin(b)),
                    (cx + inner * math.cos(b), cy + inner * math.sin(b)),
                ]
            )

    rect(-L / 2, L / 2, -W / 2, -W / 2 + w)
    rect(-L / 2, L / 2, W / 2 - w, W / 2)
    rect(-L / 2, -L / 2 + w, -W / 2, W / 2)
    rect(L / 2 - w, L / 2, -W / 2, W / 2)
    rect(-w / 2, w / 2, -W / 2, W / 2)
    arc(0, 0, f["center_circle_radius"])
    goal_geoms = []
    for sign in (-1, 1):
        for depth, width in [
            (f["goal_area_depth"], f["goal_area_width"]),
            (f["penalty_area_depth"], f["penalty_area_width"]),
        ]:
            xmin, xmax = sorted([sign * L / 2, sign * (L / 2 - depth)])
            xfront = sign * (L / 2 - depth + w / 2)
            rect(xfront - w / 2, xfront + w / 2, -width / 2, width / 2)
            rect(xmin, xmax, -width / 2, -width / 2 + w)
            rect(xmin, xmax, width / 2 - w, width / 2)
        for name, x, diameter in [
            (
                "penalty",
                sign * (L / 2 - f["penalty_mark_distance"]),
                f["penalty_mark_diameter"],
            )
        ]:
            ET.SubElement(
                world,
                "geom",
                name=f"{name}_{sign}",
                type="cylinder",
                pos=vec([x, 0, 0.001]),
                size=vec([diameter / 2, 0.0005]),
                rgba="1 1 1 1",
                contype="0",
                conaffinity="0",
            )
        for sy in (-1, 1):
            # Corner quadrant points inward from the boundary-line intersection.
            start = {
                (1, 1): math.pi,
                (1, -1): math.pi / 2,
                (-1, 1): 3 * math.pi / 2,
                (-1, -1): 0,
            }[(sign, sy)]
            arc(
                sign * (L / 2 - w / 2),
                sy * (W / 2 - w / 2),
                f["corner_arc_radius"],
                start,
                start + math.pi / 2,
                40,
            )
        x = sign * (L / 2 - w / 2)
        r = g["post_diameter"] / 2
        for sy in (-1, 1):
            name = f"goal_{sign}_post_{sy}"
            goal_geoms.append(name)
            ET.SubElement(
                world,
                "geom",
                name=name,
                type="cylinder",
                size=str(r),
                fromto=vec(
                    [
                        x,
                        sy * (g["width"] / 2 + r),
                        0,
                        x,
                        sy * (g["width"] / 2 + r),
                        g["height"] + r,
                    ]
                ),
                rgba="1 1 1 1",
            )
        name = f"goal_{sign}_bar"
        goal_geoms.append(name)
        ET.SubElement(
            world,
            "geom",
            name=name,
            type="cylinder",
            size=str(r),
            rgba="1 1 1 1",
            fromto=vec(
                [
                    x,
                    -g["width"] / 2 - r,
                    g["height"] + r,
                    x,
                    g["width"] / 2 + r,
                    g["height"] + r,
                ]
            ),
        )
        net_t = g["net_thickness"] / 2
        for part, pos, size in [
            (
                "back",
                [x + sign * g["depth"], 0, g["height"] / 2],
                [net_t, g["width"] / 2, g["height"] / 2],
            ),
            (
                "left",
                [x + sign * g["depth"] / 2, -g["width"] / 2, g["height"] / 2],
                [g["depth"] / 2, net_t, g["height"] / 2],
            ),
            (
                "right",
                [x + sign * g["depth"] / 2, g["width"] / 2, g["height"] / 2],
                [g["depth"] / 2, net_t, g["height"] / 2],
            ),
        ]:
            name = f"goal_{sign}_net_{part}"
            if g["net_collision"]:
                goal_geoms.append(name)
            ET.SubElement(
                world,
                "geom",
                name=name,
                type="box",
                pos=vec(pos),
                size=vec(size),
                rgba=".8 .85 .9 .16",
                contype="1" if g["net_collision"] else "0",
                conaffinity="65535" if g["net_collision"] else "0",
            )
    ET.SubElement(
        world,
        "geom",
        name="centre_mark",
        type="cylinder",
        pos="0 0 .001",
        size=vec([f["center_mark_diameter"] / 2, 0.0005]),
        rgba="1 1 1 1",
        contype="0",
        conaffinity="0",
    )
    # A very shallow closed mesh keeps field lines visible from grazing angles.
    # mesh must enclose volume; place a duplicate underside a millimetre below.
    n = len(verts)
    verts.extend([[x, y, z - 0.001] for x, y, z in verts.copy()])
    original = faces.copy()
    faces.extend([[a + n, cc + n, b + n] for a, b, cc in original])
    for i in range(0, n, 4):
        for j in range(4):
            a, b = i + j, i + (j + 1) % 4
            faces.extend([[a, b, b + n], [a, b + n, a + n]])
    ET.SubElement(
        asset,
        "mesh",
        name="field_lines_mesh",
        vertex=vec([v for xyz in verts for v in xyz]),
        face=" ".join(str(i) for face in faces for i in face),
    )
    ET.SubElement(
        world,
        "geom",
        name="field_lines",
        type="mesh",
        mesh="field_lines_mesh",
        rgba="1 1 1 1",
        contype="0",
        conaffinity="0",
    )
    if f["safety_wall"]:
        for name, pos, size in [
            (
                "north",
                [0, W / 2 + f["runoff"], f["wall_height"] / 2],
                [L / 2 + f["runoff"], f["wall_thickness"] / 2, f["wall_height"] / 2],
            ),
            (
                "south",
                [0, -W / 2 - f["runoff"], f["wall_height"] / 2],
                [L / 2 + f["runoff"], f["wall_thickness"] / 2, f["wall_height"] / 2],
            ),
            (
                "east",
                [L / 2 + f["runoff"], 0, f["wall_height"] / 2],
                [f["wall_thickness"] / 2, W / 2 + f["runoff"], f["wall_height"] / 2],
            ),
            (
                "west",
                [-L / 2 - f["runoff"], 0, f["wall_height"] / 2],
                [f["wall_thickness"] / 2, W / 2 + f["runoff"], f["wall_height"] / 2],
            ),
        ]:
            ET.SubElement(
                world,
                "geom",
                name=f"wall_{name}",
                type="box",
                pos=vec(pos),
                size=vec(size),
                rgba=".04 .04 .04 1",
            )
    ball = ET.SubElement(
        world, "body", name="ball", pos=vec(c["ball"]["initial_position"])
    )
    ET.SubElement(ball, "freejoint", name="ball_free")
    b = c["ball"]
    inertia = b["inertia_factor"] * b["mass"] * b["radius"] ** 2
    ET.SubElement(
        ball,
        "inertial",
        pos="0 0 0",
        mass=str(b["mass"]),
        diaginertia=vec([inertia] * 3),
    )
    ET.SubElement(
        ball,
        "geom",
        name="ball_geom",
        type="sphere",
        size=str(b["radius"]),
        rgba="1 .65 .12 1",
        contype="2",
        conaffinity="65535",
        condim="6",
    )
    contact = ET.SubElement(root, "contact")

    def pair(name, g1, g2, kind):
        d = c["contacts"][kind]
        ET.SubElement(
            contact,
            "pair",
            name=name,
            geom1=g1,
            geom2=g2,
            condim=str(d["condim"]),
            friction=vec(d["friction"]),
            solref=vec(d["solref"]),
            solimp=vec(d["solimp"]),
        )

    pair("ball_turf", "ball_geom", "turf", "ball_turf")
    for name in goal_geoms:
        pair("ball_" + name, "ball_geom", name, "ball_goal")
    if robots:
        robot_dir = Path(c["robot"]["asset_dir"])
        if not robot_dir.is_absolute():
            robot_dir = ROOT / robot_dir
        xml = robot_dir / c["robot"]["xml"]
        if not xml.exists():
            raise FileNotFoundError(f"{xml}: run python tools/fetch_assets.py first")
        src = ET.parse(xml).getroot()
        meshdir = src.find("compiler").get("meshdir", "")
        for child in src.find("asset"):
            if child.tag not in ("mesh", "material"):
                continue
            child = deepcopy(child)
            if child.tag == "mesh":
                child.set(
                    "file", str((robot_dir / meshdir / child.get("file")).resolve())
                )
            # K1 body only uses rgba; unused stock material references its floor texture.
            if child.tag == "material" and child.get("texture"):
                continue
            asset.append(child)
        actuators = ET.SubElement(root, "actuator")
        specs = c["robot"]["instances"]
        all_mask = 3 + sum(1 << (i + 2) for i in range(len(specs)))
        for i, spec in enumerate(specs):
            prefix = spec["name"] + "/"
            body = deepcopy(src.find("worldbody/body"))
            names = {
                e.get("name"): prefix + e.get("name")
                for e in body.iter()
                if e.get("name")
            }
            for e in body.iter():
                if e.get("name"):
                    e.set("name", names[e.get("name")])
            known_links = {b.get("name")[len(prefix) :] for b in body.iter("body")}
            unknown = set(c["robot"]["link_overrides"]) - known_links
            if unknown:
                raise KeyError(f"Unknown link overrides: {unknown}")
            for link in body.iter("body"):
                inertial = link.find("inertial")
                if inertial is None:
                    continue
                rc = c["robot"]
                key = link.get("name")[len(prefix) :]
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
                for attr, value in rc["link_overrides"].get(key, {}).items():
                    inertial.set(
                        attr, vec(value) if isinstance(value, list) else str(value)
                    )
            body.set("pos", vec([*spec["position"], c["robot"]["base_height"]]))
            body.set(
                "quat",
                vec([math.cos(spec["yaw"] / 2), 0, 0, math.sin(spec["yaw"] / 2)]),
            )
            free = body.find("joint[@type='free']")
            if c["robot"]["base_mode"] == "fixed":
                body.remove(free)
            for j in body.iter("joint"):
                if j.get("type") != "free":
                    j.set("damping", str(c["robot"]["joint_damping"]))
            for j, geom in enumerate(body.iter("geom")):
                geom.set("name", prefix + f"geom_{j}")
                if geom.get("contype") == "0":
                    continue
                geom.set("contype", str(1 << (i + 2)))
                geom.set("conaffinity", str(all_mask ^ (1 << (i + 2))))
                geom.set("group", "3")
            for side in ("left", "right"):
                foot = body.find(f".//body[@name='{prefix}{side}_ankle_roll_link']")
                geom = next(e for e in foot.findall("geom") if e.get("type") == "box")
                name = prefix + side + "_foot_collision"
                geom.set("name", name)
                pair(prefix + side + "_ball", "ball_geom", name, "ball_foot")
                pair(prefix + side + "_turf", name, "turf", "foot_turf")
                ET.SubElement(
                    foot,
                    "site",
                    name=prefix + side + "_foot_tip",
                    pos=".116 0 -.002",
                    size=".008",
                    rgba="0 1 0 0",
                )
            color = ".12 .48 .95 1" if spec["team"] == "blue" else ".94 .19 .17 1"
            ET.SubElement(
                body,
                "geom",
                name=prefix + "jersey",
                type="box",
                pos="-.002 0 .10",
                size=".083 .09 .085",
                rgba=color,
                contype="0",
                conaffinity="0",
                mass="0",
            )
            world.append(body)
            for motor in src.find("actuator"):
                motor = deepcopy(motor)
                motor.set("name", prefix + motor.get("name"))
                motor.set("joint", prefix + motor.get("joint"))
                motor.set(
                    "forcerange",
                    vec(
                        [
                            float(v) * c["robot"]["torque_scale"]
                            for v in motor.get("forcerange").split()
                        ]
                    ),
                )
                actuators.append(motor)
    ET.indent(root)
    return ET.tostring(root, encoding="unicode")
