"""Native MuJoCo evaluation of the wheel response stored in the shared XML.

The recorded/PD input remains nominal Nm. The user gain converts that input
to equivalent simulated drive; this is not an H7 torque calibration.
"""
import math
import ctypes
import hashlib
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import mujoco

_gain_library = None


def wheel_gain(model, data, actuator):
    gain, knee, exponent = model.actuator_gainprm[actuator, :3]
    command = float(data.ctrl[actuator])
    along = (math.copysign(1., command)*float(data.actuator_velocity[actuator])
             if command != 0. else 0.)
    return float(gain)*math.pow(max(0., 1.-max(0., along)/float(knee)), float(exponent))


def prepare_wheel_response(model):
    """Register the native callback only for models that actually request it."""
    ids = [i for i in range(model.nu)
           if model.actuator_gaintype[i] == mujoco.mjtGain.mjGAIN_USER]
    if not ids:
        return
    for i in ids:
        name = model.actuator(i).name.rsplit('/', 1)[-1]
        if name not in ('lfwheel_motor', 'rfwheel_motor'):
            raise ValueError(f'Unsupported user gain actuator: {name}')
        if not (model.actuator_gainprm[i, :3] > 0).all():
            raise ValueError(f'Invalid wheel gain parameters: {name}')
    # An experiment may already have installed the identical native C gain.
    if mujoco.get_mjcb_act_gain() is None:
        global _gain_library
        root = Path(__file__).resolve().parents[1]
        source = Path(__file__).with_name('h7_drive_response.c')
        key = hashlib.sha256(source.read_bytes()).hexdigest()[:16]+'-'+mujoco.__version__
        cache = root/'artifacts/mujoco-wheel-response'/key
        library = cache/'drive_gain.so'
        if not library.is_file():
            compiler = os.environ.get('CC') or shutil.which('cc') or shutil.which('gcc')
            if not compiler:
                raise RuntimeError('Wheel response needs a host C compiler: set CC to an existing gcc path')
            cache.mkdir(parents=True, exist_ok=True)
            temporary = cache/f'drive_gain-{os.getpid()}.so'
            subprocess.run(shlex.split(compiler)+[
                '-shared', '-fPIC', '-O2', '-I'+str(Path(mujoco.__file__).parent/'include'),
                str(source), '-lm', '-o', str(temporary)], check=True)
            temporary.replace(library)
        _gain_library = ctypes.CDLL(str(library))
        function = _gain_library.h7_drive_gain
        function.restype = ctypes.c_double
        function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        # A Python callback cannot wrap MuJoCo's temporary model while another
        # XML compiles. The identical native callback also works during compile.
        mujoco.set_mjcb_act_gain(function)
