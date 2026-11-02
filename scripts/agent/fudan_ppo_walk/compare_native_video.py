"""Compare existing encoded GPU/CPU frames only: no inference or mesh rendering."""
import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw,ImageFont


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    root=args.experiment;args.output.mkdir(parents=True,exist_ok=False)
    budget=json.loads((root/'budget.json').read_text());source=Path(budget['source_recording'])
    paths={'gpu':source.parent/(source.name+'-video')/'index.json','cpu2ms':root/'cpu2ms-video/index.json'}
    indexes={k:json.loads(p.read_text()) for k,p in paths.items()}
    videos={k:index['videos'][0] for k,index in indexes.items()}
    assert all(index['checkpoint_sha256']==budget['checkpoint_sha256'] for index in indexes.values())
    assert all(sha(Path(v['path']))==v['sha256'] for v in videos.values())
    readers={k:imageio.get_reader(v['path']) for k,v in videos.items()}
    chapters={k:{c['sample_id']:c for c in v['chapters']} for k,v in videos.items()}
    metrics=json.loads((root/'metrics.json').read_text());scores={r['sample_id']:r for r in metrics['samples']}
    output=args.output/'gpu-vs-native2ms.mp4';font=ImageFont.truetype('DejaVuSans.ttf',17)
    total=0;selected=[];review=[]
    with imageio.get_writer(output,fps=50,codec='libx264',quality=7) as writer:
        for sid in ['diagnostic_0','diagnostic_2','diagnostic_9','diagnostic_14']:
            trace={}
            for name,directory in [('gpu',source),('cpu2ms',root/'cpu2ms')]:
                path=directory/'traces'/(sid+'.npz')
                assert sha(path)==chapters[name][sid]['trajectory_sha256'] and chapters[name][sid]['complete']
                with np.load(path,allow_pickle=False) as data:trace[name]={k:data[k] for k in ('base_contact','roll_pitch_deg')}
            duration=chapters['gpu'][sid]['duration_s'];count=round(duration*50)
            assert duration==chapters['cpu2ms'][sid]['duration_s']
            start=total
            for i in range(count):
                canvas=Image.new('RGB',(1920,688),'#17212b')
                for name,x,label in [('gpu',0,'GPU mjlab'),('cpu2ms',960,'Native CPU | integration2ms')]:
                    frame=readers[name].get_data(round(chapters[name][sid]['start_s']*50)+i)
                    panel=Image.fromarray(frame);draw=ImageDraw.Draw(panel)
                    draw.rectangle((0,0,960,23),fill='#17212b')
                    met=scores[sid]['engines'][name]['technical_passed']
                    draw.text((10,2),f'{label} | iter7000 | {sid} | v2 thresholds {"MET" if met else "NOT MET"}',font=font,fill='white')
                    canvas.paste(panel,(x,0));draw=ImageDraw.Draw(canvas)
                    lo=i*10;hi=min(lo+10,len(trace[name]['base_contact']))
                    peak=float(trace[name]['base_contact'][lo:hi].max())
                    pitch=float(np.abs(trace[name]['roll_pitch_deg'][lo:hi,1]).max())
                    draw.text((x+10,649),f'Raw500Hz in this20ms: body peak={peak:.1f}N | pitch peak={pitch:.1f}deg',font=font,fill='white')
                pixels=np.asarray(canvas);writer.append_data(pixels);total+=1
                if (sid=='diagnostic_2' and i in (15,16)) or (sid=='diagnostic_14' and i==600):
                    path=args.output/(sid+f'-frame{i}.jpg');imageio.imwrite(path,pixels)
                    review.append(dict(sample_id=sid,local_time_s=i/50,path=str(path)))
            selected.append(dict(sample_id=sid,start_s=start/50,duration_s=count/50,complete_chapter=True))
    for reader in readers.values():reader.close()
    reader=imageio.get_reader(output);count=reader.count_frames();fps=reader.get_meta_data()['fps'];reader.get_data(count-1);reader.close()
    assert count==total==1800 and fps==50.
    result=dict(path=str(output),sha256=sha(output),frames=count,fps=fps,duration_s=count/fps,
        layout=['left existing GPU','right Native CPU2ms'],chapters=selected,review_frames=review,
        source_video_sha256={k:v['sha256'] for k,v in videos.items()},checkpoint_sha256=budget['checkpoint_sha256'],
        score_protocol='fudan-walk-stage1-v2',old_encoded_v1_header_relabelled_from_verified_v2_metrics=True,
        scope='Existing encoded frames only; full selected chapters, normal1x speed',
        physics_steps=0,model_forwards=0,mesh_rendering_steps=0,optimizer_updates=0)
    (args.output/'index.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('path','frames','fps','duration_s','physics_steps','model_forwards','mesh_rendering_steps')},indent=2))


if __name__=='__main__':main()
