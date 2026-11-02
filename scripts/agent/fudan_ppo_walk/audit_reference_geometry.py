"""Parse fixed URDF/STL/ONNX metadata and analytic link transforms; never run a simulator/model."""
import argparse,ast,hashlib,json,struct
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET


def blob_sha(path):
    h=hashlib.sha1();h.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def stl_vertices(path):
    with path.open('rb') as f:f.seek(80);n=struct.unpack('<I',f.read(4))[0]
    assert path.stat().st_size==84+50*n
    dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attribute','<u2')])
    records=np.memmap(path,dtype=dtype,mode='r',offset=84,shape=(n,))
    return records['vertices'].reshape(-1,3),n


def rotation(axis,angle):
    axis=np.asarray(axis,dtype=float);axis/=np.linalg.norm(axis)
    x,y,z=axis;K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3)+np.sin(angle)*K+(1-np.cos(angle))*(K@K)


def origin(element):
    T=np.eye(4)
    if element is None:return T
    xyz=np.fromstring(element.get('xyz','0 0 0'),sep=' ');r,p,y=np.fromstring(element.get('rpy','0 0 0'),sep=' ')
    T[:3,:3]=rotation([0,0,1],y)@rotation([0,1,0],p)@rotation([1,0,0],r);T[:3,3]=xyz
    return T


def bounds(vertices,T,scale=(1,1,1)):
    low=np.full(3,np.inf);high=-low
    for start in range(0,len(vertices),200000):
        point=(vertices[start:start+200000]*scale)@T[:3,:3].T+T[:3,3]
        low=np.minimum(low,point.min(0));high=np.maximum(high,point.max(0))
    return [low.tolist(),high.tolist()]


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    tree=json.loads((a.reference/'github-tree.json').read_text());assert tree['truncated'] is False
    lookup={r['path']:r for r in tree['tree']}
    urdf_rel='plane/resources/robots/infantry_V4/urdf/infantry_V4_increase.urdf';urdf=a.reference/urdf_rel;robot=ET.parse(urdf).getroot()
    assert blob_sha(urdf)==lookup[urdf_rel]['sha']
    config=ast.parse((a.reference/'plane/wheel_legged_gym/envs/wheel_legged/wheel_legged_config.py').read_text())
    outer=next(n for n in config.body if isinstance(n,ast.ClassDef) and n.name=='WheelLeggedCfg')
    init=next(n for n in outer.body if isinstance(n,ast.ClassDef) and n.name=='init_state')
    attrs={n.targets[0].id:ast.literal_eval(n.value) for n in init.body if isinstance(n,ast.Assign)}
    defaults=attrs['default_joint_angles'];root_T=np.eye(4);root_T[:3,3]=attrs['pos']
    transforms={'base_link':root_T};joints=list(robot.findall('joint'))
    while joints:
        progress=False
        for joint in list(joints):
            parent=joint.find('parent').get('link');child=joint.find('child').get('link')
            if parent not in transforms:continue
            T=origin(joint.find('origin'));angle=defaults.get(joint.get('name'),0.)
            if joint.get('type')!='fixed':
                Q=np.eye(4);Q[:3,:3]=rotation(np.fromstring(joint.find('axis').get('xyz'),sep=' '),angle);T=T@Q
            transforms[child]=transforms[parent]@T;joints.remove(joint);progress=True
        assert progress
    meshes=[];base_vertices=None
    for link in robot.findall('link'):
        for collision in link.findall('collision'):
            mesh=collision.find('geometry/mesh');assert mesh is not None
            path=(urdf.parent/mesh.get('filename')).resolve();relative=str(path.relative_to(a.reference.resolve()))
            actual=blob_sha(path);assert actual==lookup[relative]['sha']
            vertices,n=stl_vertices(path);scale=np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
            local=origin(collision.find('origin'))
            record=dict(link=link.get('name'),file=relative,git_blob_sha1=actual,triangles=n,
                local_bounds_m=bounds(vertices,local,scale),configured_reset_world_bounds_m=bounds(vertices,transforms[link.get('name')]@local,scale))
            meshes.append(record)
            if link.get('name')=='base_link':base_vertices=vertices
    tilt=[]
    for pitch_deg in (0.,5.,10.,20.,35.):
        T=np.eye(4);T[:3,:3]=rotation([0,1,0],np.deg2rad(pitch_deg))
        b=bounds(base_vertices,T);tilt.append(dict(pitch_deg=pitch_deg,minimum_local_z_m=b[0][2],root_height_for_first_base_plane_contact_m=-b[0][2]))
    limits=[dict(name=j.get('name'),type=j.get('type'),origin=j.find('origin').attrib,
        default_angle=defaults.get(j.get('name')),limit=None if j.find('limit') is None else j.find('limit').attrib) for j in robot.findall('joint')]
    import onnx
    models=[]
    for key,item in lookup.items():
        if not key.endswith('.onnx'):continue
        path=a.reference/key;assert blob_sha(path)==item['sha'];model=onnx.load(str(path))
        shape=lambda value:[d.dim_value if d.HasField('dim_value') else d.dim_param for d in value.type.tensor_type.shape.dim]
        models.append(dict(path=key,git_blob_sha1=item['sha'],producer_name=model.producer_name,producer_version=model.producer_version,
            metadata={p.key:p.value for p in model.metadata_props},model_doc_string=model.doc_string,graph_doc_string=model.graph.doc_string,
            inputs={v.name:shape(v) for v in model.graph.input},outputs={v.name:shape(v) for v in model.graph.output}))
    report=dict(reference_commit='8204e853dfd2ed06d85a322e1a998c3d20a3be2c',tree_entries=len(lookup),tree_truncated=False,
        published_training_checkpoints=[key for key in lookup if key.endswith(('.pt','.pth'))],urdf_git_blob_sha1=blob_sha(urdf),
        configured_root_position_m=attrs['pos'],configured_default_angles=defaults,
        joints=limits,collision_meshes=meshes,base_pitch_contact_geometry=tilt,published_onnx=models,
        physics_steps=0,model_forward_calls=0,scope='Static mesh vertices and analytic URDF transforms only. Not PhysX contact/initial-state simulation or behavior evidence.')
    (a.output/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(report,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
