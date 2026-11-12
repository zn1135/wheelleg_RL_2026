"""Track visible hip/wheel centers in the supplied clip; image-plane proxies only."""
import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw
from scipy.signal import correlate,find_peaks


def match(gray,template,previous,radius=16):
    h=template.shape[0]//2;x,y=np.rint(previous).astype(int)
    left=x-h-radius;top=y-h-radius
    area=gray[top:y+h+radius+1,left:x+h+radius+1]
    zero=template-template.mean();ones=np.ones_like(template)
    sums=correlate(area,ones,mode='valid',method='fft')
    squares=correlate(area*area,ones,mode='valid',method='fft')
    denominator=np.sqrt(np.maximum(squares-sums*sums/template.size,1e-6)*(zero*zero).sum())
    score=correlate(area,zero,mode='valid',method='fft')/denominator
    yy,xx=np.unravel_index(np.argmax(score),score.shape)
    return np.array([left+xx+h,top+yy+h],float),float(score[yy,xx])


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--video',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--geometry-json',type=Path,help='Manual window_s, reference_s, points and crop for another camera view')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    reader=imageio.get_reader(args.video);meta=reader.get_meta_data();fps=meta['fps']
    geometry=json.loads(args.geometry_json.read_text()) if args.geometry_json else dict(
        window_s=[14.8,17.3],reference_s=15.5,points={'hip':[441,462,12],'wheel':[518,578,22]},crop=[320,320,720,660])
    start,end=[round(t*fps) for t in geometry['window_s']];reference=round(geometry['reference_s']*fps)
    frames={i:reader.get_data(i) for i in range(start,end+1)}
    gray={i:np.asarray(Image.fromarray(f).convert('L'),dtype=np.float32) for i,f in frames.items()}
    # 每段先人工选可见轴心/轮心；仅跟踪画面，不能当编码器角度。
    points=geometry['points']
    tracks={};scores={}
    for name,(x,y,h) in points.items():
        template=gray[reference][y-h:y+h+1,x-h:x+h+1]
        positions={reference:np.array([x,y],float)};quality={reference:1.}
        for sequence in (range(reference+1,end+1),range(reference-1,start-1,-1)):
            previous=np.array([x,y],float)
            for index in sequence:
                previous,score=match(gray[index],template,previous);positions[index]=previous.copy();quality[index]=score
        tracks[name]=np.asarray([positions[i] for i in range(start,end+1)])
        scores[name]=np.asarray([quality[i] for i in range(start,end+1)])
    time=np.arange(start,end+1)/fps;vector=tracks['wheel']-tracks['hip']
    angle=np.rad2deg(np.arctan2(vector[:,0],vector[:,1]));distance=np.linalg.norm(vector,axis=1)
    peaks,_=find_peaks(angle,prominence=2.,distance=round(.3*fps))
    result=dict(video=str(args.video),video_sha256=hashlib.sha256(args.video.read_bytes()).hexdigest(),fps=fps,window_s=geometry['window_s'],
        reference_frame_s=geometry['reference_s'],reference_points=points,apparent_hip_wheel_angle_range_deg=[float(angle.min()),float(angle.max())],
        apparent_angle_peak_to_peak_deg=float(np.ptp(angle)),hip_wheel_distance_px_range=[float(distance.min()),float(distance.max())],
        minimum_template_score={k:float(v.min()) for k,v in scores.items()},peak_times_s=time[peaks].tolist(),
        note='Image-plane hip-to-wheel proxy, not motor encoder or calibrated3D angle. Camera/body pitch and yaw can affect it; short window cannot prove steady frequency. Manual marker QA required.')
    extra={}
    if 'body_rear' in tracks and 'body_front' in tracks:
        body=tracks['body_front']-tracks['body_rear']
        body_angle=np.rad2deg(np.unwrap(np.arctan2(body[:,1],body[:,0])))
        # 用刚性机身两点另看姿态变化，避免把髋轮连线摆幅当成机身摆幅。
        extra=dict(body_image_angle_deg=body_angle,leg_relative_body_proxy_deg=angle+body_angle)
        result['body_image_angle_pp_deg']=float(np.ptp(body_angle))
        result['leg_relative_body_proxy_pp_deg']=float(np.ptp(angle+body_angle))
    (args.output/'measurement.json').write_text(json.dumps(result,indent=2)+'\n')
    values=np.column_stack((time,tracks['hip'],tracks['wheel'],angle,distance,scores['hip'],scores['wheel']))
    np.savetxt(args.output/'tracking.csv',values,delimiter=',',header='time,hip_x,hip_y,wheel_x,wheel_y,apparent_angle_deg,distance_px,hip_score,wheel_score',comments='')
    np.savez_compressed(args.output/'all-tracks.npz',time=time,**tracks,**extra)
    sheet=Image.new('RGB',(1200,600),'white');draw=ImageDraw.Draw(sheet)
    for j,index in enumerate(np.linspace(start,end,8).astype(int)):
        frame=Image.fromarray(frames[index]);d=ImageDraw.Draw(frame);k=index-start
        for name,color in [('hip','red'),('wheel','lime'),('body_rear','cyan'),('body_front','orange')]:
            if name not in tracks:continue
            x,y=tracks[name][k];d.ellipse((x-7,y-7,x+7,y+7),outline=color,width=3)
        frame=frame.crop(geometry['crop']);frame.thumbnail((300,270));x=j%4*300;y=j//4*300;sheet.paste(frame,(x,y));draw.text((x+5,y+275),f'{index/fps:.2f}s',fill='black')
    sheet.save(args.output/'tracking-qa.jpg')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
