"""Build the pinned RMUC field without the central fixtures above the assembly area."""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

# CAD meshes 008–025 contain the rotors, support posts, hangars and tech cores.
# The original shells have no face IDs. On this revision these consecutive
# shells match those meshes; the repair patches have explicit face provenance.
CENTRAL_VISUALS = {f'visual_{i:03}' for i in range(8, 26)}
CENTRAL_DOCKS = {'rmuc2027_dock_positive_xy', 'rmuc2027_dock_negative_xy'}
SURFACES_SHA = 'e249beb7b228857c0f7d896767879f4a9a3033657e064764dce1612e15b704d1'


def filter_xml(source, target, tag, remove_names):
    tree = ET.parse(source)
    removed = set()
    used_meshes = set()
    for parent in tree.iter():
        children = list(parent)
        keep = []
        for node in children:
            if node.tag == tag and node.get('name') in remove_names:
                removed.add(node.get('name'))
            else:
                keep.append(node)
                if node.tag == 'geom' and node.get('mesh'):
                    used_meshes.add(node.get('mesh'))
        parent[:] = keep
    if removed != remove_names:
        raise ValueError(f'Missing pinned {tag} names in {source}')
    tree.write(target, encoding='UTF-8', xml_declaration=True)
    return used_meshes


def close_mounting_holes(path):
    """Mirror the original six-sided cutout in the platform's diagonal frame."""
    tree = ET.parse(path)
    world = tree.find('worldbody')
    assets = ET.SubElement(tree.getroot(), 'asset')
    # CAD 孔沿坐标 (u, v)，u 沿场地 45° 对角线；正负两孔绕场心对称。
    outline = [(1.030, -.115), (1.176213, -.115), (1.238, -.053213),
               (1.238, .053213), (1.173150, .118064), (1.030, .115)]
    # 孔沿略微搭接（约 0.1 毫米），避免单精度场地网格在共边处漏光或漏检接触。
    vertices = ' '.join(f'{(u-1.14)*1.001:.7f} {v*1.001:.7f} {z:g}'
                        for z in (-.005, 0.) for u,v in outline)
    mesh = 'rmuc2027_mount_cover'
    ET.SubElement(assets, 'mesh', name=mesh, vertex=vertices)
    for side, sign in [('positive', 1), ('negative', -1)]:
        xy = sign*1.14/math.sqrt(2.)
        yaw = math.radians(45 if sign == 1 else 225)
        quat = f'{math.cos(yaw/2)} 0 0 {math.sin(yaw/2)}'
        for role, top in [('visual', .599), ('collision', .6015)]:
            params = ({'rgba':'0.74902245 0.74902245 0.74902245 1'} if role == 'visual'
                      else {'priority':'1', 'solref':'.004 1'})
            ET.SubElement(world, 'geom', name=f'{mesh}_{side}_{role}', type='mesh', mesh=mesh,
                          attrib={'class':role, **params}, pos=f'{xy} {xy} {top}', quat=quat, mass='0')
    tree.write(path, encoding='UTF-8', xml_declaration=True)


def build_field(source, target):
    assets = target / 'assets'
    assets.mkdir()
    filter_xml(source/'model.xml', target/'model.xml', 'body', CENTRAL_DOCKS)
    close_mounting_holes(target/'model.xml')
    filter_xml(source/'visual_geoms.xml', target/'visual_geoms.xml', 'geom', CENTRAL_VISUALS)
    filter_xml(source/'visual_assets.xml', target/'visual_assets.xml', 'mesh',
               CENTRAL_VISUALS | {'rmuc2027_dock_mesh'})
    for entry in (source/'assets').iterdir():
        if entry.stem not in CENTRAL_VISUALS and entry.name != 'rm27_dock_visual.obj':
            (assets/entry.name).symlink_to(os.path.relpath(entry, assets))

    shells = {f'rmuc2027_surface_{i:05}' for i in range(9021, 19952)}
    removed_patches = set()
    sources = json.loads((source/'surface_repair_sources.json').read_text())['patch_source_faces']
    for i, faces in enumerate(sources):
        parts = {name for name, _ in faces}
        if parts & CENTRAL_VISUALS:
            if not parts <= CENTRAL_VISUALS:
                raise ValueError(f'Repair patch {i} crosses two assemblies')
            removed_patches.add(f'rmuc2027_repaired_surface_{i:05}')
    for layer, names in [('collision', shells), ('surface_repair', removed_patches)]:
        geoms, meshes = f'{layer}_geoms.xml', f'{layer}_assets.xml'
        used = filter_xml(source/geoms, target/geoms, 'geom', names)
        all_meshes = {el.get('name') for el in ET.parse(source/meshes).iter('mesh')}
        filter_xml(source/meshes, target/meshes, 'mesh', all_meshes - used)
    for name in ('closure_geoms.xml', 'closure_assets.xml'):
        shutil.copyfile(source/name, target/name)
    return dict(visual_geoms_removed=sorted(CENTRAL_VISUALS), dock_bodies_removed=sorted(CENTRAL_DOCKS),
                original_shells_removed=len(shells), repair_patches_removed=len(removed_patches))


def default_field(original):
    if not original.is_file():
        raise FileNotFoundError('请先运行 scripts/setup_rmuc2027.sh')
    source = original.parent
    if hashlib.sha256((source/'collision_geoms.xml').read_bytes()).hexdigest() != SURFACES_SHA:
        raise ValueError('Power Rune removal requires the pinned RMUC2027 surface mesh revision')
    cache = original.parents[3]/'fields'
    cache.mkdir(exist_ok=True)
    version = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    dest = cache/f'clear-centre-{version}'
    if not (dest/'model.xml').is_file():
        with TemporaryDirectory(dir=cache) as tmp:
            # Publish the complete directory in one rename. Concurrent starts may reuse the winner.
            target = Path(tmp)/'field'
            target.mkdir()
            summary = build_field(source, target)
            (target/'removed.json').write_text(json.dumps(summary, indent=2)+'\n')
            # Relative asset links must refer to the destination depth after publication.
            for link in (target/'assets').iterdir():
                if link.is_symlink():
                    link.unlink()
                    link.symlink_to(os.path.relpath(source/'assets'/link.name, dest/'assets'))
            try:
                target.rename(dest)
            except OSError:
                if not (dest/'model.xml').is_file():
                    raise
    return dest/'model.xml'
