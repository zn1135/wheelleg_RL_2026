"""两种稳站诊断共用的接触统计；不改变各自采样和判定流程。"""
import hashlib
import json
from pathlib import Path
import numpy as np
from ct_mjlab.envs.chuanliantui_stand_v2.diagnostics import BASE_CONTACT_THRESHOLD_N

FIELDS = ("x", "y", "height", "roll", "pitch", "planar_speed", "max_leg_speed",
          "left_wheel_support", "right_wheel_support", "base_force", "roll_pitch_speed",
          "isaac_has_stood", "recover_mode", "velocity_error")

def contact_summary(values, active, late, dt):
    force = values[:, :, FIELDS.index("base_force")]
    normal = values[:, :, FIELDS.index("recover_mode")] < .5
    contact = force > BASE_CONTACT_THRESHOLD_N
    mismatch = contact & normal & active
    counts = active.sum(0).clip(min=1)
    late_counts = late.sum(0).clip(min=1)
    late_fraction = (mismatch & late).sum(0) / late_counts
    rows = [dict(env=i, mean_base_contact_force_N=float(np.where(active[:, i], force[:, i], 0.).sum()/counts[i]),
        base_contact_fraction=float((contact[:, i] & active[:, i]).sum()/counts[i]),
        normal_base_contact_fraction=float(mismatch[:, i].sum()/counts[i]),
        normal_base_contact_s=float(mismatch[:, i].sum()*dt),
        last5_normal_base_contact_fraction=float(late_fraction[i])) for i in range(values.shape[1])]
    return dict(base_contact_threshold_N=BASE_CONTACT_THRESHOLD_N,
        definition="base force vector norm > threshold AND current mode NORMAL at the same policy step",
        normal_base_contact_fraction=float(mismatch.sum()/max(1, active.sum())),
        normal_base_contact_envs=int(mismatch.any(0).sum()),
        last5_normal_base_contact_envs=int((mismatch & late).any(0).sum()),
        last5_persistent_normal_base_contact_envs=int((late_fraction >= .5).sum()),
        persistent_definition="at least 50% of observed policy samples in the planned final 5 seconds",
        episodes=rows)


def reprocess_contacts(args, source_path):
    source = args.from_report.resolve()
    original = json.loads(source.read_text())
    measurements = source.parent/"measurements.npz"
    with np.load(measurements) as data:
        assert tuple(data["fields"]) == FIELDS
        values, active = data["values"], data["active"]
    dt = original["policy_dt_s"]
    late = active.copy()
    late[:max(0, round((original["duration_s"]-5.)/dt))] = False
    report = contact_summary(values, active, late, dt)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report.update(scope="Contact diagnostics recomputed from recorded policy samples; no simulation or learning",
        source_report=str(source), source_report_sha256=sha(source),
        measurements_sha256=sha(measurements), checkpoint_sha256=original["checkpoint_sha256"],
        parameter_updates=0, source_sha256=sha(source_path))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"contact-diagnostics.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k != "episodes"}, indent=2))

