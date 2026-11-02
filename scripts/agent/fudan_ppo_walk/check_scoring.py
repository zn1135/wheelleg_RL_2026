"""Synthetic acceptance checks: failures/missing windows retain the fixed denominator."""
import copy
import numpy as np
from ct_mjlab.fudan_ppo_walk.protocol import samples
from .score import score,DT


def fixture(sample):
    n=round(sample['case']['duration']/DT)+1
    trace=dict(time=np.arange(n)*DT,position=np.zeros((n,3)),linear_body=np.zeros((n,3)),
        linear_world=np.zeros((n,3)),angular_body=np.zeros((n,3)),roll_pitch_deg=np.zeros((n,2)),
        wheel_support=np.ones((n,2))*50.,base_contact=np.zeros(n),weight=np.ones(n)*100.,
        terminated=np.zeros(n,dtype=bool),numerical_failure=np.zeros(n,dtype=bool))
    for segment in sample['case']['segments']:
        mask=trace['time']>=segment['start'];v,y,h=segment['command']
        trace['linear_body'][mask,0]=v;trace['linear_world'][mask,0]=v
        trace['angular_body'][mask,2]=y;trace['position'][mask,2]=h
    trace['position'][:,0]=np.cumsum(trace['linear_world'][:,0])*DT
    return trace


def main():
    sample=next(s for s in samples() if s['case']['id']=='forward_2_0_0.26')
    trace=fixture(sample);assert score(sample,trace)['passed']
    for field in ('terminated','numerical_failure'):
        bad=copy.deepcopy(trace);bad[field][100]=True;assert not score(sample,bad)['passed']
    bad=copy.deepcopy(trace);bad['roll_pitch_deg'][1500,1]=10.01;assert not score(sample,bad)['passed']
    bad={k:v[:-1] for k,v in trace.items()};assert not score(sample,bad)['complete']
    bad=copy.deepcopy(trace);bad['base_contact'][50]=10.01;assert not score(sample,bad)['passed']
    bad=copy.deepcopy(trace);bad['wheel_support'][-20,0]=9.99;assert not score(sample,bad)['passed']
    bad=copy.deepcopy(trace);bad['linear_body'][100,0]=np.nan
    assert score(sample,bad)['numerical_failure'] and not score(sample,bad)['passed']
    # 转向角速度不应进入横滚/俯仰失衡判断。
    spin=next(s for s in samples() if s['case']['id']=='spin_0_2_0.26')
    assert score(spin,fixture(spin))['passed']
    stop=next(s for s in samples() if s['case']['id']=='stop_1_0.26')
    stopped=fixture(stop);assert score(stop,stopped)['passed']
    stopped['linear_world'][5000,0]=.101;assert not score(stop,stopped)['passed']
    print('Scoring checks passed: missing/early/fall/pitch/contact/stop/yaw; no simulation or optimizer updates.')

if __name__=='__main__':main()
