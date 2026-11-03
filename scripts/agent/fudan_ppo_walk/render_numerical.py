"""Show each numerically terminated trajectory at 1x, followed by an explicit missing-data card."""
import argparse,json
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image,ImageDraw
from ct_mjlab.fudan_ppo_walk.protocol import samples
from ct_mjlab.rsl_rl.runners.fudan_on_policy_runner import sha


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    report=json.loads((a.evaluation/'score.json').read_text());lookup={s['id']:s for s in samples()}
    model=mujoco.MjModel.from_binary_path(str(a.model));data=mujoco.MjData(model)
    from mjlab.viewer.model_sync import disable_model_sameframe_shortcuts
    disable_model_sameframe_shortcuts(model);model.vis.global_.offwidth=960;model.vis.global_.offheight=640
    option=mujoco.MjvOption();option.geomgroup[:]=[1,1,1,0,0,0]
    camera=mujoco.MjvCamera();camera.distance=1.8;camera.elevation=-18.;camera.azimuth=110.
    output=[]
    with mujoco.Renderer(model,height=640,width=960) as renderer:
        for row in report['samples']:
            if not row.get('numerical_failure'):continue
            path=a.evaluation/'traces'/(row['sample_id']+'.npz');sample=lookup[row['sample_id']]
            with np.load(path) as saved:trace={k:saved[k] for k in saved.files}
            observed=float(trace['time'][-1]);planned=sample['case']['duration'];video=a.output/(row['sample_id']+'.mp4')
            with imageio.get_writer(video,fps=50,codec='libx264',quality=7) as writer:
                # The final partial PD interval is reported numerically; no invented continuation.
                for t in np.arange(0.,observed,.02):
                    i=min(int(round(t/.01)),len(trace['qpos'])-2);tick=min(int(round(t/.002)),len(trace['time'])-1)
                    data.qpos[:]=trace['qpos'][i];mujoco.mj_forward(model,data);camera.lookat[:]=data.qpos[:3]
                    renderer.update_scene(data,camera=camera,scene_option=option)
                    canvas=Image.fromarray(renderer.render().copy());draw=ImageDraw.Draw(canvas)
                    draw.rectangle((0,0,960,70),fill='#542020')
                    draw.text((10,6),f'CPU NUMERICAL FAILURE | {row["sample_id"]} | t={t:.2f}s | 1x speed',fill='white')
                    draw.text((10,28),f'command={sample["case"]["segments"][0]["command"]} actual vx={trace["linear_body"][tick,0]:.3f} h={trace["position"][tick,2]:.3f}',fill='white')
                    writer.append_data(np.asarray(canvas))
                canvas=Image.new('RGB',(960,640),'#241414');draw=ImageDraw.Draw(canvas)
                for i,line in enumerate(['NUMERICAL TERMINATION - NO FURTHER DATA',row['sample_id'],
                    f'Observed {observed:.3f}s / planned {planned:.3f}s; missing {planned-observed:.3f}s.',
                    'MuJoCo reported unstable acceleration. This is a failed sample, not a completed rollout.',
                    'This 2-second end card is an explanation, not simulated time.']):draw.text((30,100+35*i),line,fill='white')
                for _ in range(100):writer.append_data(np.asarray(canvas))
            output.append(dict(sample_id=row['sample_id'],observed_s=observed,planned_s=planned,complete=False,
                video_sha256=sha(video),trajectory_sha256=sha(path),video=str(video)))
    (a.output/'index.json').write_text(json.dumps(output,indent=2)+'\n')

if __name__=='__main__':main()
