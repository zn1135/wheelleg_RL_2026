/* Host-only adapters around the unchanged H7 implementation. */
#include "leg_solver.h"
#include "lqr_balance.h"
#include "leg_balance.h"
#include "standup.h"
#include "leg_compensation.h"
#include "slip.h"
#include "air_detection.h"
#include <string.h>

static standup_ctx_t preparation;
static leg_balance_t balance;
static slip_state_t velocity;
static air_detection_t air;

static leg_state_t solve(const float *q) {
    leg_state_t leg = {0};
    leg.config.lu = machine->leg_lu;
    leg.config.lg = machine->leg_lg;
    leg.input = (leg_input_t){q[0], q[1], q[2], q[3]};
    Leg_Solve(&leg);
    return leg;
}

void ref_geometry(const float *q, float force, float tp, float *out) {
    leg_state_t leg = solve(q);
    out[0] = leg.output.virtual_leg_length;
    out[1] = leg.output.virtual_leg_angle;
    out[2] = leg.output.d_virtual_leg_length;
    out[3] = leg.output.d_virtual_leg_angle;
    memcpy(out + 4, leg.output.leg_jac, 4 * sizeof(float));
    Leg_Force_Map_Forward(&leg, force, tp, out + 8);
    out[10] = leg.output.valid && leg.output.force_valid;
}

void ref_models(float length, float angle, float *out) {
    gravity_leg_result_t gravity;
    Gravity_Comp_Compute(machine->gravity, length, angle, &gravity);
    out[0] = Gas_Spring_Force(length);
    out[1] = gravity.torque;
}

void ref_control(const float *length, const float *x, const float *target, int air, float *out) {
    lqr_state_t st;
    LQR_Init(&st);
    memcpy(st.x, x, sizeof(st.x));
    memcpy(st.target, target, sizeof(st.target));
    memcpy(st.len, length, sizeof(st.len));
    st.valid = 1;
    st.air_control = air;
    st.air_hip_max = 10.f;
    LQR_Control_Update(&st);
    memcpy(out, st.u, sizeof(st.u));
    memcpy(out + 4, st.K, sizeof(st.K));
}

void ref_reset(void) {
    Standup_Init(&preparation);
    preparation.recovery_enabled = 0;
    Leg_Balance_Init(&balance);
    Slip_Reset(&velocity);
    Air_Detection_Reset(&air);
}

void ref_velocity(float wheel, float acceleration, int contact, float *out) {
    Slip_Update_Contact(&velocity, wheel, acceleration, .001f, contact);
    out[0] = velocity.velocity;
    out[1] = velocity.acceleration;
    memcpy(out + 2, velocity.covariance, 4 * sizeof(float));
}

void ref_balance(const float *q, const float *u, float roll, float length, int flight, float *out) {
    leg_state_t left = solve(q), right = solve(q + 4);
    imu_state_t imu = {0};
    imu.online = imu.pitch_world_valid = 1;
    imu.quat[0] = 1.f;
    leg_compensation_values_t values;
    Leg_Compensation_Prepare(&values, &left, &right, &imu, 7);
    lqr_state_t st;
    LQR_Init(&st);
    st.roll = roll;
    st.leg_len_tgt[0] = st.leg_len_tgt[1] = length;
    st.air_release = flight;
    memcpy(st.u, u, sizeof(st.u));
    torque_output_t torque = {0};
    Leg_Balance_Compute(&balance, &st, &left, &right, &values, &lqr_debug, .001f, &torque);
    memcpy(out, balance.F, 2 * sizeof(float));
    memcpy(out + 2, torque.dm, 4 * sizeof(float));
    memcpy(out + 6, torque.dji, 2 * sizeof(float));
}

void ref_air(const float *q, const float *motor, float pitch, float pitch_rate, float az, float *out) {
    leg_state_t left = solve(q), right = solve(q + 4);
    imu_state_t imu = {0};
    imu.online = imu.pitch_world_valid = 1;
    imu.quat[0] = 1.f;
    imu.euler_rad[ATTITUDE_ROLL] = imu.pitch_world = pitch;
    imu.gyro_rad_s[1] = pitch_rate;
    imu.accel_vertical = az;
    leg_compensation_values_t values;
    Leg_Compensation_Prepare(&values, &left, &right, &imu, 7);
    air_input_t input = {.imu=&imu, .leg={&left,&right}, .compensation=&values,
                         .joint_ready=1, .permit=1};
    memcpy(input.torque, motor, sizeof(input.torque));
    Air_Detection_Update(&air, &input);
    out[0] = air.leg[0].force;
    out[1] = air.leg[1].force;
    out[2] = air.armed;
    out[3] = air.flight;
}

void ref_standup(const float *q, float pitch, float *out) {
    leg_state_t left = solve(q), right = solve(q + 4);
    imu_state_t imu = {0};
    imu.online = imu.pitch_world_valid = 1;
    imu.quat[0] = 1.f;
    imu.euler_rad[ATTITUDE_ROLL] = imu.pitch_world = pitch;
    leg_compensation_values_t values;
    Leg_Compensation_Prepare(&values, &left, &right, &imu, 7);
    torque_output_t torque = {.valid = 1};
    int route = Standup_Update(&preparation, &imu, &left, &right, &values, 1, 1, .001f, &torque);
    out[0] = preparation.phase;
    out[1] = route;
    memcpy(out + 2, preparation.force, 2 * sizeof(float));
    memcpy(out + 4, preparation.tp, 2 * sizeof(float));
    memcpy(out + 6, preparation.length_cmd, 2 * sizeof(float));
    memcpy(out + 8, preparation.angle_cmd, 2 * sizeof(float));
    memcpy(out + 10, torque.dm, 4 * sizeof(float));
}
