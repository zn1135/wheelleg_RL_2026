"""Produce a concise delivery index without turning technical metrics into user acceptance."""
import argparse,json,hashlib
from pathlib import Path
from ct_mjlab.fudan_ppo_walk.protocol import samples


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--experiment',type=Path,required=True)
    a=p.parse_args();root=Path.cwd();run=a.run.resolve();exp=a.experiment.resolve()
    training=json.loads((run/'training_report.json').read_text());budget=json.loads((exp/'budget.json').read_text())
    lines=['# 复旦PPO首轮结果','', '**最终状态：待用户视频验收；自动数值检查不代表最终通过。**','',
        '**本轮入口存在实现遗漏：未随机错开首轮回合进度，启动当前帧/历史噪声也曾分开采样。已修复并通过真实step/reset控制流的零物理检查，但本轮模型未重训，不能称为完整参考配方复现。**','',
        f'从零训练；完成{training["iterations"]}轮采样，{training["transitions"]}条转移；PPO与估速器分别更新{training["primary_updates"]}/{training["encoder_updates"]}次。',
        f'训练循环墙钟{training["training_wall_s"]:.2f}秒；512环境×48步=24576条/轮，4个minibatch，每个6144条，5个epoch。',
        f'固定总窗口：{budget["start"]} 至 {budget["deadline"]}；详见[预算审计](budget.json)。','',
        f'模型：[model_final.pt]({run}/model_final.pt)，SHA256 `{training["checkpoint_sha256"]}`。',
        f'导出：[ONNX]({exp}/export/policy.onnx)；[接口元数据]({exp}/export/interface.json)。','',
        '## 两引擎技术检查','', '| 引擎 | 类别 | 名义通过/总数 | 随机通过/总数 | 随机完整记录 |','|---|---|---:|---:|---:|']
    overall={}
    for engine in ('gpu','cpu'):
        score_path=exp/f'final-{engine}'/'score.json'
        if not score_path.exists():overall[engine]={'missing':True};continue
        score=json.loads(score_path.read_text());overall[engine]=dict(technical_thresholds_met=score['technical_thresholds_met'],
            complete=sum(r['complete'] for r in score['samples']),passed=sum(r['passed'] for r in score['samples']),
            numerical_failures=sum(r.get('numerical_failure',False) for r in score['samples']),total=score['total'])
        categories=sorted({key.rsplit(':',1)[0] for key in score['groups']})
        for category in categories:
            n=score['groups'][category+':nominal'];r=score['groups'][category+':random']
            lines.append(f'| {engine} | {category} | {n["passed"]}/{n["total"]} | {r["passed"]}/{r["total"]} | {r["complete"]}/{r["total"]} |')
        detailed=['# 名义工况逐段成绩','', '| 样本 | 段 | vx MAE m/s | yaw MAE rad/s | 高度 MAE m | max俯仰 ° | 技术通过 |', '|---|---:|---:|---:|---:|---:|---|']
        for row in score['samples']:
            if row['randomized']:continue
            for i,seg in enumerate(row.get('segments',[])):
                detailed.append(f'| {row["sample_id"]} | {i} | {seg["velocity"]["mae"]:.4f} | {seg["yaw"]["mae"]:.4f} | {seg["height"]["mae"]:.4f} | {seg["max_abs_pitch_deg"]:.2f} | {row["passed"]} |')
        (exp/f'final-{engine}'/'nominal-results.md').write_text('\n'.join(detailed)+'\n')
    lines.extend(['','每个引擎固定987样本；失败、提前结束和缺测保留在原分母。全部逐样本指标/窗口比例/接触条件及停车漂移见下列JSON。','',
        '- [mjlab逐样本数值](final-gpu/score.json) · [名义工况表](final-gpu/nominal-results.md) · [原始轨迹](final-gpu/traces/)',
        '- [原生CPU逐样本数值](final-cpu/score.json) · [名义工况表](final-cpu/nominal-results.md) · [原始轨迹](final-cpu/traces/)','',
        '## 视频','', '正常速度、完整起步/刹停，黄标间距1m；每段标注模型哈希、样本/种子、指令、实际vx/yaw/高度和时间。需由用户观看后明确确认。'])
    for engine in ('gpu','cpu'):
        index=exp/f'video-{engine}'/'index.json'
        if not index.exists():lines.append(f'- {engine}视频：缺测/未完成。');continue
        data=json.loads(index.read_text())
        for video in data['videos']:
            path=Path(video['path']).resolve();lines.append(f'- [{engine} {path.stem}]({path})（{video["frames"]/50:.2f}秒）')
        lines.append(f'- [{engine}章节/视频与轨迹哈希]({index})；缺失视频样本{len(data["missing"])}个。')
    lines.extend(['','## 复现与边界','',
        '- [实际命令](COMMANDS.md)、[来源与修复前后哈希](provenance.json)、[原训练源码](training-source.tar.gz)、[入口修复正式检查](initialization-fix-final-check.json)。',
        '- CPU原有3个数值终止样本保持失败；[0.5ms积分对照](native-integration-diagnostic-4substeps/report.json)仅这3例完整/无数值异常，不替代正式987例或代表行为通过。',
        '- [3个CPU数值终止原视频索引](video-cpu-numerical/index.json)：保留原始短轨迹，末尾明确标注缺测，不伪造7秒后续。',
        f'- [完整规格与差异]({root}/docs/mjlab-fudan-ppo-walk.md)，[接口]({root}/docs/interfaces/20261101-fudan-ppo-walk.md)。',
        '- 参考固定提交8204e853，算法四模块仅包内导入变化；网络、16项系数与PPO超参数审计通过。机身/电机/闭链及接触模型适配不属于逐位复现。',
        '- 初始没有合格源模型，不伪造保留基线；第523/1084轮独立诊断，两端均0/20联合技术通过、无数值异常。',
        '- 原维护总检查在既有SuperMimic源码哈希映射处失败，本次未改该源码；隔离检查确认3396个物理/部署文件与HEAD一致、55个CPU保护文件一致、五个旧task仍拒绝。',
        '- H7未改、未适配/负责人待确认；无硬件试验或真机通过。延迟后续必须实机分段测量；本轮零额外延迟不代表真机零延迟。','',
        'TensorBoard（本轮未自动启动）：','',f'```bash\nmjlab_training/.venv/bin/python -m tensorboard.main --logdir {run} --port 6006\n```'])
    (exp/'README.md').write_text('\n'.join(lines)+'\n')
    (exp/'summary.json').write_text(json.dumps(dict(training=training,engines=overall,final_acceptance='pending_user_video_review'),indent=2)+'\n')
    print(json.dumps(overall,indent=2))

if __name__=='__main__':main()
