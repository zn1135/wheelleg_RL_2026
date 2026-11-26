"""按用户确认的试次对应关系，将实拍动作事件与日志腿长分段对齐。"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .analyze_real022 import read_csv, matrix
from .probe_takeoff_work import write_json

FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
# 视频逐帧人工判读。区间保留30fps的事件边界不确定性。
EVENTS = [
    dict(trial=1, extension_frame=49, takeoff_bracket_frames=[48, 49],
         touchdown_bracket_frames=[61, 62], apex_bracket_frames=[53, 56],
         landing_compression_frames=[64, 66]),
    dict(trial=2, extension_frame=168, takeoff_bracket_frames=[167, 169],
         touchdown_bracket_frames=[179, 180], apex_bracket_frames=[173, 176],
         landing_compression_frames=[182, 185]),
]


def longest_interval(time, mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return max((float(time[b-1]-time[a]) for a, b in zip(starts, ends)), default=0.)


def plot_background(time, lengths, row):
    fig, ax = plt.subplots(figsize=(12.8, 3.04), dpi=100)
    fig.subplots_adjust(left=.07, right=.985, bottom=.24, top=.88)
    colors = ['#2388d1', '#e78a16']
    for side, name in enumerate(('左腿', '右腿')):
        ax.plot(time, lengths[:, side]*100, label=name+'实测解算腿长', color=colors[side], lw=2)
        metric = row['legs'][side]
        ax.scatter([metric['extension_peak_log_s'], metric['air_min_log_s']],
                   [metric['extension_peak_cm'], metric['air_min_cm']], color=colors[side], s=28, zorder=4)
    begin, end = row['clear_air_video_s']
    ax.axvspan(begin-row['video_offset_s'], end-row['video_offset_s'], color='#76b58b', alpha=.13,
               label='视频明确腾空窗口')
    ax.set_xlim(-.1, .8);ax.set_ylim(10, 39)
    ax.set_xlabel('日志时间，相对首次Jump推理 / s', fontsize=12)
    ax.set_ylabel('腿长 / cm', fontsize=12)
    ax.grid(alpha=.2);ax.legend(loc='upper right', ncol=3, fontsize=11)
    ax.set_title('实测关节解算腿长；动作事件对齐，非硬件同步', fontsize=13)
    fig.canvas.draw()
    bitmap = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    bounds = ax.get_window_extent()
    geometry = dict(left=bounds.x0, right=bounds.x1, top=304-bounds.y1, bottom=304-bounds.y0)
    plt.close(fig)
    return bitmap, geometry


def run(video, archive_path, estimate_path, output):
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams['font.family'] = ['Noto Sans CJK SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    reader = imageio.get_reader(video)
    meta = reader.get_meta_data();fps=meta['fps'];assert abs(fps-30)<1e-6
    source_frames = [frame for frame in reader]
    reader.close()
    estimate = np.load(estimate_path)
    rows, traces = [], []
    with zipfile.ZipFile(archive_path) as archive:
        for event in EVENTS:
            number=event['trial'];data=read_csv(archive,f'试次{number}-H7反馈.csv')
            time=data['rel_s'];lengths=matrix(data,['left_leg_0','right_leg_0'])
            early=(time>=.08)&(time<=.23)
            early_ids=np.flatnonzero(early)
            peak_index=int(early_ids[np.argmax(lengths[early].mean(1))])
            offset=event['extension_frame']/fps-time[peak_index]
            clear_air=[event['takeoff_bracket_frames'][1]/fps,event['touchdown_bracket_frames'][0]/fps]
            airborne=(time>=clear_air[0]-offset)&(time<=clear_air[1]-offset)
            # 首次收腿谷值限定在已明确腾空窗口内，远离落地压缩。
            retract=airborne&(time>=.21)&(time<=.42)
            ids=np.flatnonzero(retract)
            legs=[]
            for side,name in enumerate(('left','right')):
                high=int(early_ids[np.argmax(lengths[early,side])])
                low=int(ids[np.argmin(lengths[retract,side])])
                high_cm,low_cm=lengths[[high,low],side]*100
                min_video=offset+time[low]
                clear_length=np.interp(clear_air[0]-offset,time,lengths[:,side])*100
                legs.append(dict(side=name,extension_peak_cm=float(high_cm),air_min_cm=float(low_cm),
                    shortening_cm=float(high_cm-low_cm),shortening_percent=float(100*(high_cm-low_cm)/high_cm),
                    extension_peak_log_s=float(time[high]),air_min_log_s=float(time[low]),
                    air_min_video_s=float(min_video),contraction_time_ms=float((time[low]-time[high])*1000),
                    length_at_first_clear_air_frame_cm=float(clear_length),
                    shortening_after_first_clear_air_frame_cm=float(clear_length-low_cm),
                    longest_below20cm_in_air_ms=1000*longest_interval(time,airborne&(lengths[:,side]<.2)),
                    min_stays_in_clear_air_with_two_frame_offset_error=bool(
                        clear_air[0]<min_video-2/fps and min_video+2/fps<clear_air[1])))
            est_time=estimate[f'trial{number}_relative_s'];height=estimate[f'trial{number}_height_m']
            mask=(est_time>=0)&(est_time<=1);apex_ids=np.flatnonzero(mask)
            apex_time=float(est_time[apex_ids[np.argmax(height[mask])]])
            landing=(time>=.62)&(time<=.8);landing_ids=np.flatnonzero(landing)
            compressed=int(landing_ids[np.argmin(lengths[landing].mean(1))])
            row=dict(**event,video_offset_s=float(offset),video_time_formula='video_s = log_relative_s + offset',
                clear_air_video_s=clear_air,legs=legs,
                predicted_imu_apex_video_s=offset+apex_time,
                predicted_landing_compression_video_s=float(offset+time[compressed]),
                first_jump_board_us=float(np.median(data['board_us']-time*1e6)),
                measured_fault_values=np.unique(data['fault'][airborne]).astype(int).tolist())
            row['apex_within_visual_interval']=bool(event['apex_bracket_frames'][0]/fps <= row['predicted_imu_apex_video_s'] <= event['apex_bracket_frames'][1]/fps)
            row['landing_compression_within_visual_interval']=bool(event['landing_compression_frames'][0]/fps <= row['predicted_landing_compression_video_s'] <= event['landing_compression_frames'][1]/fps)
            assert row['apex_within_visual_interval'] and row['landing_compression_within_visual_interval']
            assert all(x['min_stays_in_clear_air_with_two_frame_offset_error'] for x in legs)
            rows.append(row);traces.append((time,lengths))
    result=dict(source_video=str(video.resolve()),video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
        archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),fps=fps,
        pairing='User confirmed trial1/2 match the two jumps and the waiting interval was cut from video.',
        alignment='Each jump independently anchored at maximal extension near liftoff; apex and landing compression checked separately.',
        uncertainty='Action-event alignment, no shared hardware clock. Minima checked against offsets ±2 video frames.',
        measurement='Logged hip-to-wheel distance computed from measured joints. Peak-to-air-min reduction starts near liftoff, not an exactly timed contact event.',
        trials=rows,physics_steps=0,training_updates=0)
    write_json(output/'result.json',result)
    font=ImageFont.truetype(FONT,25);small=ImageFont.truetype(FONT,20)
    normal_count=slow_count=0
    with imageio.get_writer(output/'aligned-normal.mp4',fps=30,codec='libx264',quality=8) as normal, \
         imageio.get_writer(output/'aligned-slow4x.mp4',fps=30,codec='libx264',quality=8) as slow:
        for row,(time,lengths) in zip(rows,traces):
            chart,box=plot_background(time,lengths,row)
            chart.save(output/f'trial{row["trial"]}-curve.png')
            first=int(np.ceil((row['video_offset_s']-.1)*fps));last=int(np.floor((row['video_offset_s']+.8)*fps))
            min_frame=round(np.mean([x['air_min_video_s'] for x in row['legs']])*fps)
            for frame_id in range(first,last+1):
                video_s=frame_id/fps;log_s=video_s-row['video_offset_s']
                current=np.array([np.interp(log_s,time,lengths[:,j])*100 for j in range(2)])
                # 去除原画面底部空地，保留机器人和接触地面；原像素比例保持。
                frame=Image.fromarray(source_frames[frame_id]).crop((0,0,1280,480))
                canvas=Image.new('RGB',(1280,864),'white');canvas.paste(frame,(0,80));canvas.paste(chart,(0,560))
                draw=ImageDraw.Draw(canvas);draw.rectangle((0,0,1280,80),fill='#17212c')
                draw.text((14,6),f'第{row["trial"]}跳 | 视频 {video_s:.3f}s | 日志 {log_s:+.3f}s | 左腿 {current[0]:.1f}cm  右腿 {current[1]:.1f}cm',font=font,fill='white')
                draw.text((14,45),'按蹬伸动作分段对齐；绿色窗口为视频明确腾空段；腿长按日志左右定义',font=small,fill='#c4dce8')
                x=box['left']+(log_s+.1)/.9*(box['right']-box['left'])
                draw.line((x,560+box['top'],x,560+box['bottom']),fill='#c32939',width=2)
                for value,color in zip(current,('#2388d1','#e78a16')):
                    y=560+box['bottom']-(value-10)/29*(box['bottom']-box['top'])
                    draw.ellipse((x-5,y-5,x+5,y+5),fill=color)
                normal.append_data(np.asarray(canvas));normal_count+=1
                draw.rectangle((1010,82,1279,114),fill='#17212c')
                draw.text((1020,84),'0.25×慢放',font=small,fill='white')
                for _ in range(4):slow.append_data(np.asarray(canvas));slow_count+=1
                if frame_id==min_frame:canvas.save(output/f'trial{row["trial"]}-minimum.png')
    validation=[]
    for name,count in [('aligned-normal.mp4',normal_count),('aligned-slow4x.mp4',slow_count)]:
        movie=imageio.get_reader(output/name);metadata=movie.get_meta_data();decoded=sum(1 for _ in movie);movie.close()
        assert decoded==count and metadata['fps']==30
        validation.append(dict(file=name,frames=count,duration_s=count/30,full_decode_passed=True))
    result['videos']=validation
    write_json(output/'result.json',result)
    print(json.dumps(rows,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',type=Path,required=True)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--height-estimate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.video,args.archive,args.height_estimate,args.output)
