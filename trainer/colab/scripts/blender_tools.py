"""Shared Blender building blocks for Pokerdice (runs INSIDE Blender: `blender -b -P script.py -- ...`).

Units: metres. Axes of the die (local):
  +Z = top, -Z = bottom, -Y = front, +Y = back, -X = left, +X = right
The die net from the trainer (step 4) is mapped via the UV coordinates:
for each face "up in the image" (U) and "right in the image" (R) are fixed as directions on the die
(the way the unfolded net is seen from outside); the rotation from the net turns U/R clockwise.
"""
import json
import math
import os
import sys

import bpy
import bmesh
from mathutils import Vector, Matrix
from bpy_extras.object_utils import world_to_camera_view

FIELDS = ["top", "bottom", "front", "back", "left", "right"]
NORMAL = {"top": (0, 0, 1), "bottom": (0, 0, -1), "front": (0, -1, 0), "back": (0, 1, 0),
          "left": (-1, 0, 0), "right": (1, 0, 0)}
# (U = image up, R = image right) at rotation 0, seen from outside - derived geometrically from the net
BASIS = {"top": ((0, 1, 0), (1, 0, 0)), "front": ((0, 0, 1), (1, 0, 0)), "bottom": ((0, -1, 0), (1, 0, 0)),
         "back": ((0, 0, -1), (1, 0, 0)), "right": ((0, 1, 0), (0, 0, -1)), "left": ((0, 1, 0), (0, 0, 1))}


# ------------------------------------------------------------------ call / arguments
def script_args():
    """Arguments after "--" (Blender) or all of them (as Python module bpy)."""
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]


def load_scene(folder):
    with open(os.path.join(folder, "scene.json"), encoding="utf-8") as f:
        return json.load(f)


def empty_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.unit_settings.system = "METRIC"
    return sc


# ------------------------------------------------------------------ render settings
def setup_render(sc, width=640, height=480, samples=32, gpu=True):
    sc.render.engine = "CYCLES"
    sc.render.resolution_x, sc.render.resolution_y = width, height
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "JPEG"
    sc.render.image_settings.quality = 92
    sc.view_settings.view_transform = "Standard"   # colours close to the textures (no Filmic/AgX)
    sc.view_settings.look = "None"
    cy = sc.cycles
    cy.samples = samples
    cy.use_adaptive_sampling = True
    cy.use_denoising = True
    cy.max_bounces = 4
    cy.diffuse_bounces = 2
    cy.glossy_bounces = 2
    cy.transmission_bounces = 0
    cy.caustics_reflective = False
    cy.caustics_refractive = False
    device = "CPU"
    if gpu:
        device = enable_gpu()
    cy.device = "GPU" if device != "CPU" else "CPU"
    # always OpenImageDenoise: the OptiX denoiser cannot be created with the Colab driver
    # ("Failed to create OptiX denoiser"). Rendering still happens on the GPU.
    cy.denoiser = "OPENIMAGEDENOISE"
    if hasattr(cy, "denoising_use_gpu"):
        cy.denoising_use_gpu = device != "CPU"
    return device


_DENOISE_LEVEL = [0]   # 0 = OIDN (GPU if possible), 1 = OIDN on the CPU, 2 = without denoiser


def render(sc=None):
    """bpy.ops.render.render(write_still=True) with a fallback if the denoiser cannot be created.
    The chosen fallback level stays for the following images."""
    sc = sc or bpy.context.scene
    while True:
        try:
            bpy.ops.render.render(write_still=True)
            return
        except RuntimeError as e:
            if "denois" not in str(e).lower() or _DENOISE_LEVEL[0] >= 2:
                raise
            _DENOISE_LEVEL[0] += 1
            cy = sc.cycles
            if _DENOISE_LEVEL[0] == 1:
                cy.denoiser = "OPENIMAGEDENOISE"
                if hasattr(cy, "denoising_use_gpu"):
                    cy.denoising_use_gpu = False
                print("⚠ Denoiser not possible on the GPU - denoising on the CPU (OpenImageDenoise).", flush=True)
            else:
                cy.use_denoising = False
                print("⚠ Denoiser not possible - rendering WITHOUT denoiser (images a little grainier).", flush=True)


def enable_gpu():
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
    except KeyError:
        return "CPU"
    for kind in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = kind
            prefs.get_devices()
            gpus = [d for d in prefs.devices if d.type == kind]
            if gpus:
                for d in prefs.devices:
                    d.use = d.type == kind
                return kind
        except TypeError:
            continue
    return "CPU"


# ------------------------------------------------------------------ materials
def load_image(path):
    return bpy.data.images.load(path, check_existing=True)


def material_texture(name, image_path, roughness=0.5, tile=None, colorspace="sRGB", emission=None):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = load_image(image_path)
    tex.image.colorspace_settings.name = colorspace
    if tile:
        coord = nt.nodes.new("ShaderNodeTexCoord")
        mapping = nt.nodes.new("ShaderNodeMapping")
        mapping.inputs["Scale"].default_value = (tile, tile, 1)
        nt.links.new(coord.outputs["UV"], mapping.inputs["Vector"])
        nt.links.new(mapping.outputs["Vector"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = roughness
    if emission is not None:
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Emission Color"])
        bsdf.inputs["Emission Strength"].default_value = emission
    return m, tex


def material_color(name, rgb_lin, roughness=0.6):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb_lin, 1.0)
    b.inputs["Roughness"].default_value = roughness
    return m


# ------------------------------------------------------------------ die
def _rotate(u, r, deg):
    for _ in range((deg // 90) % 4):
        u, r = r, tuple(-x for x in u)   # clockwise: up -> right, right -> down
    return u, r


def _set_uv(me, sc_data, k):
    """New UVs per face: every face (also the strips of the rounded edges) gets its OWN face image.
    The face is assigned to the die face its normal points to most, and projected onto that face.
    Without this Blender interpolates the edge strips across the atlas -> foreign symbols/colours on the edges."""
    # directly via the mesh data (not bmesh): this keeps the hardened normals of the rounding
    if not me.uv_layers:
        me.uv_layers.new(name="UVMap")
    uvd = me.uv_layers.active.data
    for poly in me.polygons:
        n = poly.normal
        field = max(FIELDS, key=lambda x: n.dot(Vector(NORMAL[x])))
        info = sc_data["faces"][field]
        U, R = _rotate(BASIS[field][0], BASIS[field][1], info["rotation"])
        U, R = Vector(U), Vector(R)
        c, r = info["cell"]
        for li in poly.loop_indices:
            co = me.vertices[me.loops[li].vertex_index].co
            u = min(max(co.dot(R) / k + 0.5, 0.004), 0.996)   # not beyond the cell edge (neighbouring image)
            v = min(max(co.dot(U) / k + 0.5, 0.004), 0.996)
            uvd[li].uv = ((c + u) / 3.0, 1.0 - (r + 1) / 2.0 + v / 2.0)


def create_die(sc_data, folder, name="Die"):
    """Die with rounded edges, texture atlas according to the die net. Returns the object (origin = centre)."""
    k = sc_data["die"]["edge_mm"] / 1000.0
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=k)
    bm.loops.layers.uv.new("UVMap")
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    rounding = sc_data["die"]["corner_radius_mm"] / 1000.0
    if rounding > 0:
        mod = ob.modifiers.new("Rounding", "BEVEL")
        mod.width = min(rounding, k * 0.3)
        mod.segments = 4
        mod.limit_method = "NONE"
        mod.harden_normals = True
        for p in me.polygons:
            p.use_smooth = True
        # apply the rounding (incl. normals), so that the edge strips can get their own UVs
        dg = bpy.context.evaluated_depsgraph_get()
        fixed = bpy.data.meshes.new_from_object(ob.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
        ob.modifiers.remove(mod)
        ob.data = fixed
        bpy.data.meshes.remove(me)
        me = fixed
        me.name = name
    _set_uv(me, sc_data, k)
    mat, _ = material_texture("Die", os.path.join(folder, sc_data["atlas"]), roughness=0.35)
    mat.node_tree.nodes["Principled BSDF"].inputs["Specular IOR Level"].default_value = 0.4
    me.materials.append(mat)
    return ob


def copy_die(template, name):
    ob = template.copy()
    ob.data = template.data   # same mesh/material
    ob.name = name
    bpy.context.scene.collection.objects.link(ob)
    return ob


def top_face(ob):
    """(field, tilt_deg): which face points up, and how tilted the die lies."""
    rot = ob.matrix_world.to_3x3().normalized()
    best, bz = None, -2.0
    for field in FIELDS:
        z = (rot @ Vector(NORMAL[field])).z
        if z > bz:
            best, bz = field, z
    return best, math.degrees(math.acos(max(-1.0, min(1.0, bz))))


# ------------------------------------------------------------------ tray, table, background
def ring_mesh(name, r_inner, r_outer, z0, z1, segments=96):
    bm = bmesh.new()

    def circle(r, z):
        return [bm.verts.new((r * math.cos(2 * math.pi * i / segments), r * math.sin(2 * math.pi * i / segments), z))
                for i in range(segments)]
    ib, it, ob_, ot = circle(r_inner, z0), circle(r_inner, z1), circle(r_outer, z0), circle(r_outer, z1)
    for i in range(segments):
        j = (i + 1) % segments
        bm.faces.new((ib[i], ib[j], it[j], it[i]))      # inner wall
        bm.faces.new((ot[i], ot[j], ob_[j], ob_[i]))    # outer wall
        bm.faces.new((it[i], it[j], ot[j], ot[i]))      # top edge
        bm.faces.new((ob_[i], ob_[j], ib[j], ib[i]))    # underside
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def create_tray(sc_data, folder):
    t = sc_data["tray"]
    R = t["diameter_mm"] / 2000.0
    h = t["wall_height_mm"] / 1000.0
    rb = t.get("wall_width_mm", 12.0) / 1000.0
    thickness = 0.008
    # bottom (felt) - disc, top side at z = 0
    bpy.ops.mesh.primitive_cylinder_add(vertices=96, radius=R + 0.0005, depth=thickness, location=(0, 0, -thickness / 2))
    bottom = bpy.context.active_object
    bottom.name = "TrayBottom"
    tile = (2 * R * 1000.0) / t.get("felt_tile_mm", 120.0)
    felt, _ = material_texture("Felt", os.path.join(folder, t["felt"]), roughness=0.95, tile=tile)
    # UV of the top side: planar from above (so the tiling in mm is right)
    me = bottom.data
    uvl = me.uv_layers.active or me.uv_layers.new()
    for p in me.polygons:
        for li in p.loop_indices:
            co = me.vertices[me.loops[li].vertex_index].co
            uvl.data[li].uv = (co.x / (2 * R) + 0.5, co.y / (2 * R) + 0.5)
    me.materials.append(felt)
    wall = ring_mesh("TrayWall", R, R + rb, -thickness, h)
    wall.data.materials.append(material_color("Wall", t["wall_color_lin"], roughness=0.55))
    for p in wall.data.polygons:
        p.use_smooth = True
    return bottom, wall, R


def table_and_background(sc_data, folder):
    bpy.ops.mesh.primitive_plane_add(size=4.0, location=(0, 0, -0.0081))
    table = bpy.context.active_object
    table.name = "Table"
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 1.0, 0.5))
    back = bpy.context.active_object
    back.name = "Background"
    back.scale = (2.4, 1.6, 1.0)
    back.rotation_euler = (math.radians(90), 0, 0)
    mats = []
    for i, rel in enumerate(sc_data.get("surroundings") or []):
        m, tex = material_texture(f"Surroundings{i}", os.path.join(folder, rel), roughness=0.8, emission=0.0)
        mats.append(m)
    if not mats:
        mats = [material_color("Surroundings", (0.25, 0.22, 0.2), 0.8)]
    return table, back, mats


def set_material(ob, mat):
    if ob.data.materials:
        ob.data.materials[0] = mat
    else:
        ob.data.materials.append(mat)


# ------------------------------------------------------------------ camera and light
def create_camera(sc_data):
    cam = bpy.data.cameras.new("Camera")
    ob = bpy.data.objects.new("Camera", cam)
    bpy.context.scene.collection.objects.link(ob)
    bpy.context.scene.camera = ob
    w, h = sc_data["camera"]["resolution"]
    diag = math.radians(sc_data["camera"]["fov_diag_deg"])
    hfov = 2 * math.atan(math.tan(diag / 2) * w / math.hypot(w, h))
    cam.sensor_fit = "HORIZONTAL"
    cam.angle = hfov
    cam.clip_start = 0.01
    cam.clip_end = 20
    return ob


def aim_camera(ob, target, elevation_deg, azimuth_deg, distance_m, roll_deg):
    e, a = math.radians(elevation_deg), math.radians(azimuth_deg)
    direction = Vector((math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)))
    ob.location = Vector(target) + direction * distance_m
    view = (Vector(target) - ob.location).normalized()
    q = view.to_track_quat("-Z", "Y")
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = q @ Matrix.Rotation(math.radians(roll_deg), 4, "Z").to_quaternion()


def color_temperature_rgb(kelvin):
    """Rough conversion colour temperature -> RGB (0-1)."""
    t = kelvin / 100.0
    r = 1.0 if t <= 66 else min(1, max(0, 1.292936 * ((t - 60) ** -0.1332047592)))
    g = min(1, max(0, 0.39008157 * math.log(t) - 0.63184144)) if t <= 66 else min(1, max(0, 1.1298909 * ((t - 60) ** -0.0755148492)))
    b = 1.0 if t >= 66 else (0.0 if t <= 19 else min(1, max(0, 0.5432068 * math.log(t - 10) - 1.19625408)))
    return (r, g, b)


def create_light():
    ld = bpy.data.lights.new("Light", "AREA")
    ob = bpy.data.objects.new("Light", ld)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def setup_world(sc):
    w = bpy.data.worlds.new("World")
    sc.world = w
    w.use_nodes = True
    return w.node_tree.nodes["Background"]


def tray_collision(R, rb, h, parts=48):
    """Invisible collision bodies for the tray: a box as bottom + a ring of wall pieces.
    (Collision with the exact mesh shape is unreliable in Blender's background mode - dice fell through.)"""
    objects = []
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, -0.01))
    bottom = bpy.context.active_object
    bottom.name = "Collision_Bottom"
    bottom.scale = (2 * (R + rb) + 0.02, 2 * (R + rb) + 0.02, 0.02)
    objects.append(bottom)
    width = 2 * math.pi * (R + rb / 2) / parts * 1.15
    height = h + 0.03   # a little higher than the real wall, so that no die flies out
    for i in range(parts):
        a = 2 * math.pi * i / parts
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=((R + rb / 2) * math.cos(a), (R + rb / 2) * math.sin(a), height / 2))
        ob = bpy.context.active_object
        ob.name = f"Collision_Wall{i:02d}"
        ob.scale = (rb, width, height)
        ob.rotation_euler = (0, 0, a)
        objects.append(ob)
    for ob in objects:
        ob.hide_render = True
        ob.visible_camera = False
        ob.visible_shadow = False
        ob.visible_diffuse = ob.visible_glossy = ob.visible_transmission = ob.visible_volume_scatter = False
    bpy.context.view_layer.update()
    return objects


# ------------------------------------------------------------------ physics
def setup_physics(sc, fps=30, seconds=3.0):
    if sc.rigidbody_world is None:
        with bpy.context.temp_override(scene=sc):
            bpy.ops.rigidbody.world_add()
    rw = sc.rigidbody_world
    rw.substeps_per_frame = 10
    rw.solver_iterations = 20
    sc.render.fps = fps
    sc.frame_start = 1
    sc.frame_end = int(fps * seconds)
    rw.point_cache.frame_start = 1
    rw.point_cache.frame_end = sc.frame_end
    return rw


def rigid_body(ob, kind="ACTIVE", shape="BOX", mass=0.005, friction=0.5, bounce=0.3):
    with bpy.context.temp_override(object=ob, active_object=ob, selected_objects=[ob], selected_editable_objects=[ob]):
        bpy.ops.rigidbody.object_add(type=kind)
    rb = ob.rigid_body
    rb.collision_shape = shape
    rb.friction = friction
    rb.restitution = bounce
    rb.collision_margin = 0.0004
    if kind == "ACTIVE":
        rb.mass = mass
        rb.linear_damping = 0.15
        rb.angular_damping = 0.2
        rb.use_deactivation = True
    return rb


def simulate(sc):
    """Run the simulation from the start and stay at the last frame."""
    sc.rigidbody_world.point_cache.frame_end = sc.frame_end
    sc.frame_set(sc.frame_start)
    for f in range(sc.frame_start + 1, sc.frame_end + 1):
        sc.frame_set(f)


# ------------------------------------------------------------------ labels
def corners_world(ob, k):
    h = k / 2.0
    return [ob.matrix_world @ Vector((x, y, z)) for x in (-h, h) for y in (-h, h) for z in (-h, h)]


def box_in_image(sc, cam, ob, k):
    """(x0,y0,x1,y1) in pixels (clipped to the image), share visible in the image; None if behind the camera."""
    w, h = sc.render.resolution_x, sc.render.resolution_y
    pts = [world_to_camera_view(sc, cam, p) for p in corners_world(ob, k)]
    if any(p.z <= 0 for p in pts):
        return None, 0.0
    xs = [p.x * w for p in pts]
    ys = [(1 - p.y) * h for p in pts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    area = max(1e-6, (x1 - x0) * (y1 - y0))
    cx0, cx1, cy0, cy1 = max(0, x0), min(w, x1), max(0, y0), min(h, y1)
    visible = max(0, cx1 - cx0) * max(0, cy1 - cy0) / area
    return (cx0, cy0, cx1, cy1), visible


def invisible(obj):
    """Not to be seen in the image (e.g. collision bodies of the physics) -> must not hide anything in the visibility test."""
    o = getattr(obj, "original", obj)
    return o.hide_render or not o.visible_camera or o.name.startswith("Collision")


def top_visible(sc, cam, ob, k, field, depsgraph):
    """Share (0-1) of 9 points on the top face that the camera sees directly (not hidden)."""
    rot = ob.matrix_world.to_3x3().normalized()
    n = rot @ Vector(NORMAL[field])
    a = rot @ Vector(BASIS[field][0])
    bb = rot @ Vector(BASIS[field][1])
    centre = ob.matrix_world.translation + n * (k / 2.0)
    hits = 0
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            p = centre + a * (i * 0.33 * k) + bb * (j * 0.33 * k)
            start = cam.matrix_world.translation
            direction = (p - start)
            dist = direction.length
            r = direction.normalized()
            ok, loc, _, _, obj, _ = sc.ray_cast(depsgraph, start, r, distance=dist + 0.001)
            # invisible bodies (the collision wall of the physics is 3 cm higher than the real wall) hide nothing -
            # otherwise the ray stops there and upright dice at the wall would wrongly get "?"
            for _ in range(8):
                if not (ok and obj is not None and invisible(obj)):
                    break
                further = loc + r * 1e-5
                ok, loc, _, _, obj, _ = sc.ray_cast(depsgraph, further, r, distance=max(0.0, (p - further).length + 0.001))
            if ok and obj is not None and obj.original == ob and (loc - p).length < 0.25 * k:
                hits += 1
    return hits / 9.0
