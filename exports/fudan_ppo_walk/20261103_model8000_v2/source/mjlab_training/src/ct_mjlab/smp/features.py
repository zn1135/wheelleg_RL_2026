"""One shared, yaw/translation-invariant feature definition for data and rewards."""
import torch

FEATURE_SCHEMA = 'chuanliantui_smp_motion30_v1'
FEATURE_DIM = 30
SAMPLE_DT = .01
WINDOW = 10
FEATURE_GROUPS = {'height': [0,1], 'gravity_b': [1,4], 'linear_velocity_b': [4,7],
                  'angular_velocity_b': [7,10], 'leg_sin': [10,14], 'leg_cos': [14,18],
                  'motor_velocity': [18,24], 'wheel_position_b': [24,30]}


def inverse_rotate(quaternion, vector):
    # MuJoCo四元数顺序wxyz；所有方向量统一转到机体系。
    xyz = -quaternion[..., 1:]
    twice_cross = 2 * torch.linalg.cross(xyz, vector)
    return vector + quaternion[..., :1] * twice_cross + torch.linalg.cross(xyz, twice_cross)


def motion_features(root, motor_q, motor_dq, wheel_position_w, ground_height):
    """root=(...,13) pos/quat/world linear+angular velocity; motors in XML order."""
    quat = root[..., 3:7]
    gravity = torch.zeros_like(root[..., :3])
    gravity[..., 2] = -1
    wheels = inverse_rotate(quat[..., None, :].expand(*wheel_position_w.shape[:-1], 4),
                            wheel_position_w-root[..., None, :3])
    leg = motor_q[..., [0,1,3,4]]
    return torch.cat(((root[..., 2]-ground_height)[..., None], inverse_rotate(quat, gravity),
                      inverse_rotate(quat, root[..., 7:10]), inverse_rotate(quat, root[..., 10:13]),
                      leg.sin(), leg.cos(), motor_dq, wheels.flatten(-2)), -1)


class MotionStateReader:
    def __init__(self, env):
        from ..envs.chuanliantui.model import ACTUATED_JOINTS
        self.env = env
        robot = env.scene['robot']
        joints = robot.find_joints(ACTUATED_JOINTS, preserve_order=True)[0]
        self.qadr, self.vadr = robot.indexing.joint_q_adr[joints], robot.indexing.joint_v_adr[joints]
        self.wheels = robot.find_bodies(('lfwheel','rfwheel'), preserve_order=True)[0]

    def read(self):
        data = self.env.scene['robot'].data
        state = dict(root=torch.cat((data.root_link_pos_w, data.root_link_quat_w,
                                     data.root_link_lin_vel_w, data.root_link_ang_vel_w), -1),
                     motor_q=self.env.sim.data.qpos[:, self.qadr],
                     motor_dq=self.env.sim.data.qvel[:, self.vadr],
                     wheel_position_w=data.body_link_pos_w[:, self.wheels],
                     ground_height=self.env.scene.env_origins[:, 2])
        return state

    def features(self):
        return motion_features(**self.read())


class MotionHistory:
    def __init__(self, num_envs, device):
        self.frames = torch.zeros(num_envs, WINDOW, FEATURE_DIM, device=device)
        self.count = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self, ids):
        self.frames[ids] = 0
        self.count[ids] = 0

    def append(self, frame):
        finite = torch.isfinite(frame).all(-1)
        self.frames.copy_(torch.roll(self.frames, -1, dims=1))
        self.frames[:, -1] = torch.where(finite[:, None], frame, 0.)
        self.count.add_(1).clamp_(max=WINDOW)
        self.reset(~finite)
        return self.frames, self.count == WINDOW
