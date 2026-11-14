"""Create the requested Chinese Word handoff with bundled python-docx; no simulation."""
import json
from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'docs/word/20261109-双腿负载拟合流程与训练仿真接入.docx'
RUN=ROOT/'artifacts/length-response-fit/20261109-formal-integration-01'


def main():
    d=Document();s=d.sections[0]
    s.page_width=Inches(8.5);s.page_height=Inches(11)
    s.top_margin=s.bottom_margin=Inches(.7);s.left_margin=s.right_margin=Inches(.75)
    for name in ('Normal','Title','Subtitle','Heading 1','Heading 2','Caption'):
        st=d.styles[name];st.font.name='Noto Sans CJK SC'
        st._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Noto Sans CJK SC')
        st.font.color.rgb=RGBColor.from_string('000000')
    n=d.styles['Normal'];n.font.size=Pt(10.5);n.paragraph_format.line_spacing=1.12;n.paragraph_format.space_after=Pt(5)
    d.styles['Title'].font.size=Pt(23)
    d.styles['Subtitle'].font.size=Pt(11);d.styles['Subtitle'].font.italic=False
    d.styles['Subtitle'].font.color.rgb=RGBColor.from_string('555555')
    d.styles['Heading 1'].font.size=Pt(16)
    d.styles['Heading 2'].font.size=Pt(12)
    for name in ('Heading 1','Heading 2'):
        d.styles[name].paragraph_format.space_before=Pt(10)
        d.styles[name].paragraph_format.space_after=Pt(6)
    header=s.header.paragraphs[0];header.text='轮腿机器人　负载模型接入记录';header.style=d.styles['Caption']
    header.runs[0].font.size=Pt(9)
    footer=s.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    footer.add_run('2026年11月9日　 |　 ')
    field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field)
    def p(text,style=None):return d.add_paragraph(text,style)
    pending=[False]
    def h(text):
        para=d.add_heading(text,level=1)
        if pending[0]:para.paragraph_format.page_break_before=True;pending[0]=False
    def table(headers,rows,widths=None):
        t=d.add_table(rows=1,cols=len(headers));t.autofit=False
        if widths:
            for col,width in zip(t.columns,widths):col.width=Inches(width)
        for i,text in enumerate(headers):t.rows[0].cells[i].text=text
        for row in rows:
            cells=t.add_row().cells
            for i,text in enumerate(row):cells[i].text=str(text)
        for ri,row in enumerate(t.rows):
            for ci,cell in enumerate(row.cells):
                if widths:cell.width=Inches(widths[ci])
                tcPr=cell._tc.get_or_add_tcPr();borders=OxmlElement('w:tcBorders')
                for side in ('top','left','bottom','right'):
                    b=OxmlElement('w:'+side);b.set(qn('w:val'),'single');b.set(qn('w:sz'),'4');b.set(qn('w:color'),'D9D9D9');borders.append(b)
                tcPr.append(borders)
                sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'E8EEF3' if ri==0 else ('F7F9FB' if ri%2==0 else 'FFFFFF'));tcPr.append(sh)
                for para in cell.paragraphs:
                    para.paragraph_format.space_after=Pt(3);para.paragraph_format.space_before=Pt(3)
                    for run in para.runs:run.font.size=Pt(9);run.bold=ri==0
        prop=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(prop)
        p('')
        return t
    def page():pending[0]=True

    d.add_paragraph('双腿负载拟合与训练仿真接入记录','Title')
    p('采用版本与可复查流程　2026年11月9日','Subtitle')
    p('本记录供机器人训练与部署维护使用。用户观察到双腿新模型的表现接近真机，决定正式采用当前双腿镜像负载。本文保留采用依据、拟合步骤、参数来源、接入方式和未完成事项，防止把主观相似或软件检查通过写成真机标定完成。')
    h('当前决定')
    table(['项目','当前状态'],[
        ('正式默认物理','双腿启用 mirrored-local-load-dahl-20261109-v1'),
        ('左腿来源','右腿参数按电机轴方向镜像，尚未独立辨识'),
        ('共用实现','训练与GPU仿真使用同一Torch实现；原生MuJoCo使用同一公式的NumPy实现'),
        ('网络与控制','8000-v2权重、观测接口、PD60/2、策略100Hz保持'),
        ('本次未执行','策略训练、固件烧录、带动力真机试验'),
        ('采用理由的性质','用户观察反馈与后续训练选择，不是独立真机误差验收通过')],[1.25,5.7])
    h('已知结果与判断边界')
    p('原8000模型在同初态20例诊断中，旧物理技术指标通过19例；双腿拟合物理通过0例，主要表现为上下振动和轮载中断。正式默认接入后的GPU复跑也未通过该组指标。旧策略不适应新物理，不等于可以唯一判断负载参数错误。')
    p('当前模型仍有明显残差，尚未消除sim2real差异。这里的“正式采用”表示默认代码和参数已替换，不表示全行程、接地运动、左腿或真机动态已经标定完成。')

    page();h('实录数据与拟合步骤')
    table(['数据项目','实录或处理方式'],[
        ('动作','右腿约251 mm缓慢收至190 mm，目标角锁定入场约19.675°'),
        ('控制与支撑','机身固定、双轮离地；5秒接管加20秒目标过渡，随后长保持'),
        ('时间与样本','189.602秒，94802个2 ms完整配对点'),
        ('输入完整性','189603个1 kHz分发连续；2 ms更新之间保持上一请求，未发现漏掉中间指令'),
        ('力矩的含义','请求力矩与驱动器估计力矩分别保存；后者不是轴端扭矩传感器'),
        ('配置修正','旧录制标签误写主H7；实际烧录H7_ros。两份映射相同，原始记录未篡改')],[1.25,5.7])
    for title,body in [
        ('一 核对输入','冻结原始数据、固件归档、配置与文件校验值；检查序号、数据年龄、故障、饱和和遥控状态。新录制入口明确选择H7或H7_ros。'),
        ('二 重建真实控制','用归档C代码复算目标、前馈、PID与限幅。核对MIT编码顺序和反馈时延；不通过修改零点、极性或力矩增益来压低误差。'),
        ('三 建立自由回放','仅首帧设置实测姿态，后续由MuJoCo Warp GPU积分。并行比较同实录请求力矩及同目标同控制器两分支，真实位置只作评分。'),
        ('四 分离数值问题','先保持已标定限位与闭链连接。等虚功静载检查发现原生滑杆摩擦表示会爬动；替代表示改善合成静载，但不能据此宣布真机模型对齐。'),
        ('五 搜索局部负载','先拟合保守负载和阻尼，再比较静动摩擦与预滑移记忆。有限搜索与坐标细调选出当前候选，保留失败和长保持退化结果。'),
        ('六 复核并接入','跑完整长保持，复算所有物理步的状态更新、施力、力界和耗散；双腿镜像试跑8000策略后，按用户决定接为默认模型。')]:
        para=p('');para.add_run(title+'　').bold=True;para.add_run(body)

    page();h('当前模型组成与参数')
    p('当前是组合等效模型：原名义气弹簧弹性及50 N·s/m阻尼继续存在，另叠加保守关节负载、转动摩擦、转动阻尼、滑杆阻尼和有记忆的预滑移阻力。它不是一条已经实测确认的厂家气弹簧推力曲线。')
    table(['参数','采用值','单位'],[
        ('名义气弹簧基础项','零位压缩10 mm，参考行程90 mm，弹性上限150','mm 与 N'),
        ('原弹簧速度阻尼','50','N·s/m'),
        ('前后电机摩擦','0.5 / 0.5','N·m'),
        ('前后电机粘性阻尼','0.143606 / 0.972227','N·m·s/rad'),
        ('滑杆附加阻尼','2000','N·s/m'),
        ('预滑移静止与滑动容量','110 / 55.5','N'),
        ('摩擦过渡速度','0.0005','m/s'),
        ('预滑移刚度','300000','N/m'),
        ('新回合摩擦初始力','0','N'),
        ('右腿参考关节角','0.676062 / −0.189548','rad'),
        ('基础保守负载偏置','−4.761855 / 5.119923','N·m'),
        ('偏置修正','−0.256768 / −0.574830','N·m'),
        ('长度方向刚度修正','1.471268','N·m/rad'),
        ('交叉刚度修正','−1.130576','N·m/rad'),
        ('长度方向二次项','12.863273','N·m/rad²')],[2.25,3.7,1.0])
    p('基础刚度矩阵右腿2×2部分为 [0.713149, 6.011501; 6.011501, −0.141552] N·m/rad。数值完整精度及来源以 fitted_leg_load.json 为准。包内恢复时间等未激活字段保留溯源，不当作已独立辨识的物理量。')
    p('左侧电机轴与右侧相反，因此同时镜像关节角与关节力矩，前后电机不交换；滑杆仍以伸长为正。每个环境、每条腿独立维护摩擦记忆，局部复位只清理指定环境，并在下一物理拍锁存新姿态，避免旧状态产生假冲击。')

    page();h('拟合效果与迁移结果')
    table(['对照方式','全段腿长RMSE','全段腿角RMSE','移动段腿长RMSE'],[
        ('同实录请求力矩','2.555 mm','0.662°','5.365 mm'),
        ('同目标同控制器','1.052 mm','0.058°','2.082 mm')],[2.1,1.6,1.6,1.65])
    p('完整段189.602秒，移动段为5至25秒。同请求最大腿长误差21.726 mm，同目标最大7.725 mm。长静置拉低全段平均，不能用末端到位代替动态对齐；这些数据都参与过开发或回归，不是新的独立盲测。')
    d.add_picture(str(ROOT/'artifacts/length-response-fit/20261109-bristle-final-full-01/comparison.png'),width=Inches(6.85))
    p('图一　前35秒实录与自由仿真　黑线为实录　蓝线为同请求力矩　橙线为同目标控制','Caption')
    p('8000模型接入双腿负载后：20例原模型技术19/20，新模型0/20。原地0.22 m工况，垂直速度RMS约0.00093变为0.20696 m/s，出现零轮载；0.26 m前进0.25 m/s工况，实际前向均速0.226变为0.090 m/s，俯仰峰约0.033变为8.798°。')
    p('程序复算、镜像和施力一致说明接入实现有依据，但不能替代行为验收。原拟合为0.2 ms物理步长，当前行走为2 ms；初始记忆力、姿态变化和接地条件也不同，尚未完成因果拆分。')

    page();h('正式接入位置与回退方式')
    table(['维护位置','职责'],[
        ('resources/.../mjlab/fitted_leg_load.json','唯一参数包，保存来源与双腿镜像选择'),
        ('chuanliantui/fitted_leg_load.py','共用负载与记忆公式，Torch和原生MuJoCo状态管理'),
        ('pd_action.py 与 huananhu_action.py','各物理拍施加负载，处理所有或局部环境复位'),
        ('fudan_ppo_walk_config.py','sim.fitted_leg_load 默认True'),
        ('evaluate.py','默认新模型，legacy选项回放旧模型；显式旧诊断包先关闭默认施力'),
        ('mj_sim2sim_huananhu_multimode.py','原生回放默认接入；stand_v2的每个积分子步更新记忆'),
        ('新checkpoint元数据','记录物理profile与参数SHA，旧权重和日志不覆盖')],[3.0,3.95])
    p('共享XML继续负责机构、限位、闭链和原气弹簧执行器；新等效负载在运行时施加。仅复制XML不包含全部物理模型，交付时必须同时保留参数JSON和共用代码。H7未修改，本次没有烧录。')
    d.add_heading('如何复现与回退',level=2)
    p('默认GPU评估保持原evaluate命令，不再需要额外挂载诊断参数。使用新输出目录和本次截止时间，不能覆盖历史记录。')
    p('旧物理回放：evaluate增加 --leg-load-profile legacy。复旦训练配置回退：FudanWalkCfg.sim.fitted_leg_load=False。其他共用任务可将动作配置 fitted_leg_load=False；原生 Playback 构造参数同名设False。')
    p('本次只接入模型，没有启动新训练。训练侧既有质量、接触和电机随机化仍会围绕新名义参数或叠加既有扰动；具体分布应在下一次训练前核对，不能把随机化样本全部称为精确复现该候选。')
    d.add_heading('复核入口',level=2)
    p('归档报告目录：artifacts/length-response-fit/20261109-formal-integration-01/。check_formal_leg_load 检查归档施力与跨后端公式；gpu-lifecycle.json检查20环境局部复位；default-acceptance保存正式默认路径的20例复跑。')
    p('模型回放页面：http://127.0.0.1:8781/。原始记录、拟合输入、所有候选和失败结论保留在 artifacts/length-response-fit 与 pace-joint-capture 中，不能用本文件替代原始证据。')

    page();h('未完成事项与优化顺序')
    table(['优先项','下一项具体工作与有效判据'],[
        ('一 步长敏感性','保持500 Hz PD和100 Hz策略，对照0.2 ms与2 ms物理积分；看垂直速度、零轮载时长和同力矩误差是否显著随步长改变。'),
        ('二 左腿独立验证','固定相同准备姿态，采左腿小幅收伸及保持；冻结当前参数做同输入回放，不通过改左右符号或增益消除误差。'),
        ('三 腿角与负载分离','固定腿长，缓慢变角往返；加入逐姿态重力分析，判断保守负载与伸缩阻力能否共同解释不同角度。'),
        ('四 初态与重复性','同准备流程重复同动作，检查启动时刻和停走差；比较合理的初始摩擦记忆，不能任意给每条轨迹独立初力来作弊。'),
        ('五 独立动态验证','用未参与调参的新轨迹，同时核对同请求矩和同目标分支；不再使用用户明确排除的大幅中断数据。'),
        ('六 新物理下策略适配','先完成物理与数值核对，再按明确预算训练。保留旧8000作为对照，完整运动矩阵与正常速度视频须重新验收。'),
        ('七 隐状态与历史模型','摩擦记忆未加入Actor观测；SuperMimic世界模型输入也未显式包含该状态。核对可辨识性，并重新验证旧世界模型、BC示范和SMP先验的适用性。'),
        ('八 实机与全行程','目前没有左腿独立标定、接地sim2real量化或完整987通过。已标定关节限位保留；正常跨界返回补测与跨姿态模型验证仍待完成。')],[1.45,5.5])
    p('推荐先做第一项，再做第二和第三项。若缩小物理步长显著减振，应先修正数值接入；若差异仍在，继续用新姿态数据约束负载和记忆模型，再判断是否需要策略重训。')
    p('来源索引：20261109-right-length-fullhold-01 为实录复核；20261109-inputs-02 为冻结输入；20261109-bristle-final-full-01 为当前候选；20261109-fudan8000-right-load-01 为原物理与双腿对照。新参数包SHA256为 e6ef82fa4380fa01341fa2166be56282c5290c82fec2b2a04e5b3107ce49218f。')
    d.core_properties.title='双腿负载拟合与训练仿真接入记录'
    d.core_properties.subject='拟合流程 参数来源 默认接入 验证边界 未完成事项'
    d.core_properties.author='Codex'
    for part in (d._element,d.styles.element):
        for border in list(part.iter(qn('w:pBdr'))):border.getparent().remove(border)
    OUT.parent.mkdir(parents=True,exist_ok=True);d.save(OUT);print(OUT)


if __name__=='__main__':main()
