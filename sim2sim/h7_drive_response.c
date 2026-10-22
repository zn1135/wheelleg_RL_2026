/* Offline MuJoCo candidate gain: nominal command to equivalent response only.
 * No policy, feedback injection, board I/O, or changes to driver calibration.
 * Gain falls with simulated speed in the commanded direction. Opposing motion
 * uses the low-speed gain; that quadrant is an explicit unverified extension.
 */
#include <math.h>
#include <mujoco/mujoco.h>

mjtNum h7_drive_gain(const mjModel* m, const mjData* d, int actuator) {
  const mjtNum* p = m->actuator_gainprm + actuator * mjNGAIN;
  mjtNum u = d->ctrl[actuator];
  mjtNum along = u == 0 ? 0 : copysign(1.0, u) * d->actuator_velocity[actuator];
  mjtNum fraction = fmax(0.0, 1.0 - fmax(0.0, along) / p[1]);
  return p[0] * pow(fraction, p[2]);
}
