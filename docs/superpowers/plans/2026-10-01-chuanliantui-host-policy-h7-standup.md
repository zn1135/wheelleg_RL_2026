# Chuanliantui Host Policy + H7 Standup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the complete standup checkpoint on the computer and execute its actions through a guarded 500 Hz H7 control loop, then verify a ground rear-swing standup on the robot.

**Architecture:** Keep the established 2701 disabled diagnostic intact. Add a separate, mutually exclusive execution session: H7 streams 25/125 observations at 100 Hz, the host returns sequence-bound actions, and H7 computes virtual PD and closed-chain motor torque every 2 ms. A one-shot operator contact event switches from zero policy action to actor action on the following policy sample.

**Tech Stack:** C11/STM32H723/FreeRTOS/USB CDC/CAN, Python 3.8/PyTorch/Isaac Gym, native C tests plus H7 ARM GCC build, MuJoCo comparison, staged physical bench evidence.

**Spec:** `docs/superpowers/specs/2026-10-01-chuanliantui-host-policy-h7-standup-design.md`

## Global Constraints

- Train workspace: `<训练仓根目录>`, branch `26_wheelleg`; H7 workspace: `$H7_REPO_PATH` without Git metadata. Preserve all existing files and captures. Do not create branches or worktrees.
- Checkpoint: `logs/chuanliantui_standup/Sep22_16-26-11_standup_no_legangle_resume/model_6000.pt`, SHA-256 `8070977e19114cb9bbf358a090bd78cdbd1be568a76e02d417c369e828bc8f36`. Root `logs/` remains read-only.
- Model contract: 25 obs, 125 history, 6 actions; 100 Hz policy, 500 Hz PD; 0.5 leg position scale, 10.0 wheel velocity scale; gains 10/1 leg and 0/0.1 wheel; virtual torque limits `[40,40,3.9,40,40,3.9]` N·m. Body +x forward; no imcawl yaw-heading outer loop.
- Initial standup: rear-swing near 0.15 m, `vx=0`, `yaw_rate=0`, height command 0.20 m. Actor takes over on the sample after operator-confirmed wheel contact. No contact-force or body-height sensor exists on this H7.
- User-selected B mapping, pending disarmed physical verification: DM0/DM1 are physical left → model `rf`, DM2/DM3 physical right → model `lf`; DM1/DM3 drive the CAD front long arms (~210 mm), DM0/DM2 the rear short arms (~113 mm). Virtual `lf1/rf1` are not physical DM identities. Old observation, zero and numerical evidence uses the opposite front/rear slot order and cannot unlock torque.
- User confirmed the original per-slot `dm_sign` and `dm_zero` values must remain unchanged. The original CAD-angle constants swapped into B predicted the right wheel at 279.85 mm behind/95.24 mm below the hip. The user supplied the original SolidWorks screenshot, confirmed it depicts the same motor-raw-zero pose as the 125-frame disabled capture and that the dimension starts at the hip motor axis: 319.93 mm behind/107.49 mm below. A single-pose closed-chain fit produced provisional odd/front CAD zero 0.8138714045 rad and even/rear CAD zero 2.3227884081 rad, preserving the four per-slot motor signs/zeros. Right-side endpoint agreement is numerical CAD evidence only; left zero, physical signs and Jacobian remain gated.
- Preserve DR16 fresh down→up/right-down gate, USB/CAN/motor/IMU checks, zero/disable on fault, explicit STOP, and distinct disabled 2701 and joint-bench modes. Keep wheel reduction 15.5 and `LOWER_OBS_MAPPING_VERIFIED=0` until separately verified.
- Use configured Isaac Gym Python 3.8 and import `isaacgym` before `torch`. Do not start training or TensorBoard. No `git commit`, amend, push, or PR without showing exact staged scope and a Chinese commit message and receiving explicit confirmation. H7 has no Git, so save source/ELF snapshots and hashes per experiment.

## Review Focus

1. Front/rear DM input axes swapped while side pairing still looks correct: disabled single-slot feedback and asymmetric Jacobian tests must detect it; Task 0 tests it.
2. Host response at exactly 10 ms or after uint32 millisecond wrap: H7 must reject stale action and stop; Task 1 tests it.
3. Duplicate or reordered contact confirmation: H7 must never activate the actor twice or skip the zero-action contact sample; Task 1 tests it.
4. Remote disconnect or USB loss while a valid action is queued: H7 must zero and disable before output; Task 3 integration tests it.
5. Jacobian invalid, motor-mapped torque above physical limit, or host interruption: H7 must stop without nonfinite/unbounded output or relying on host cleanup; Tasks 2 and 4 test these cases.

---

### Task 0: Verify B and rebuild the observation source map

**Files:**
- Modify: `$H7_REPO_PATH/App/lower_observation.c`, `$H7_REPO_PATH/imcalib/user-lib/machine_config.c` (correct slot comments; change numeric offsets only if measurements require it)
- Modify: `<训练仓根目录>/scripts/agent/check_h7_observation.py`
- Modify: `$H7_REPO_PATH/docs/joint-response-validation.md`
- Create: a uniquely named evidence directory under `$H7_REPO_PATH/captures/`

**Interfaces:**
- Produces a measured `physical_left/right × CAD_front/rear → DM0..3` table, raw disabled feedback at documented known poses, angular direction and zero-offset record, and a pass/fail B gate. Store measured poses in `captures/20261001-b-mapping-disabled-01/physical-poses.json` with each DM's driver `pos_rad` and zero-adjusted `pos_zero_rad`, identified CAD front/rear angles, measurement tolerance, and photo/trace references. If B fails, revise the spec and this plan before actuator-code implementation.
- Produces `Lower_Observation_Build` with physical side and front/rear source indices derived from that table; `check_h7_observation.py` remains an independent training/CAD reference and records that numerical agreement does not prove physical zero or direction.

- [ ] **Step 1: Preserve the baseline.** Record H7 source/ELF/board identity and capture hashes, train-worktree status and checkpoint hash; keep pre-existing captures immutable.
- [ ] **Step 2: Resolve the physical identity gate with actuators disabled.** Trace the four DM cables to the CAD front long/rear short input shafts; record raw and zero-adjusted slot feedback at safely supported known poses and, where mechanically safe, after passive motion. Record both slots even if closed-chain coupling moves both. Use shaft marks/photos to identify sign and offset. If pairing, sign, or pose remains uncertain, block the mapping changes and actuator output while continuing only the independent protocol/host work in Tasks 1 and 4.
- [ ] **Step 3: Write failing source-order checks.** In `check_h7_observation.py`, use asymmetric front/rear perturbations to assert DM1 and DM3 change the corresponding model hip and knee via `J_front`, while DM0 and DM2 change the knee via `J_rear`; assert opposite-side observation entries stay fixed. Add `--physical-poses` to read the measured JSON and test signs/zeros within its documented measurement tolerance. Keep the existing nonfinite/invalid-geometry checks.
- [ ] **Step 4: Run the independent observation checker and expect the old-order assertions to fail.** From the train root after loading `.env.local`, run `"$WHEELLEGGED_PYTHON" scripts/agent/check_h7_observation.py --h7-root "$H7_REPO_PATH" --out "$H7_REPO_PATH/captures/20261001-b-mapping-disabled-01/before.json"` with the configured environment library path; save its exit code and output.
- [ ] **Step 5: Implement the measured mapping.** Change `lower_observation.c` source indices, position and velocity signs and offsets together; correct `machine_config.c` slot comments and change its numeric offsets only if the disabled pose measurements require it. Do not set `LOWER_OBS_MAPPING_VERIFIED=1` on numerical evidence alone.
- [ ] **Step 6: Re-run native observation checks using the recorded raw disabled feedback.** Run `"$WHEELLEGGED_PYTHON" scripts/agent/check_h7_observation.py --h7-root "$H7_REPO_PATH" --physical-poses "$H7_REPO_PATH/captures/20261001-b-mapping-disabled-01/physical-poses.json" --out "$H7_REPO_PATH/captures/20261001-b-mapping-disabled-01/after.json"`; compare 25 obs, virtual positions/velocities and Jacobians against the independent reference and physical poses. Record tolerances, hashes and the physical gate. The current board still emits old-order observations until Task 5 flashes the updated firmware; its old 25 obs must not be counted as a new-map pass. If physical zero remains uncertain, block Task 5's torque gate.

### Task 1: Isolated execution session and wire contract

**Files:**
- Create: `$H7_REPO_PATH/App/lower_policy_run.h`, `$H7_REPO_PATH/App/lower_policy_run.c`
- Create: `$H7_REPO_PATH/tests/test_policy_run.c`
- Modify: `$H7_REPO_PATH/App/lower_protocol.h`, `$H7_REPO_PATH/tools/test_host.py`, `$H7_REPO_PATH/docs/protocol.md`

**Interfaces:**
- Produces `lower_policy_run_t` with `session`, `sample_seq`, `sample_ms`, `obs[25]`, `history[125]`, `action[6]`, `phase` (`OFF`, `PRECONTACT`, `CONTACT_EDGE`, `ACTIVE`, `FAULT`), fault and counters.
- Produces `Lower_Policy_Run_Init`, `Start(state, session, ready)`, `Sample(state, obs[25], now, source_valid)`, `Action(state, session, source_seq, action[6], now)`, `Confirm_Contact(state, session, source_seq)`, `Fault(state, reason)`, `Stop(state, session)` and `Abort(state)`. All functions return `bool` except init/fault/abort.
- Use layout `2901`; command IDs `0x30 START`, `0x31 ACTION`, `0x32 CONTACT_CONFIRM`, `0x33 STOP`, `0x34 QUERY`; response IDs `0xB0 SAMPLE`, `0xB1 STATUS`. START payload is `u32 layout, u32 session, 32 raw bytes checkpoint SHA-256`; ACTION is `u32 session, u32 source_seq, float[6]`; CONTACT_CONFIRM and STOP are `u32 session, u32 source_seq` and `u32 session` respectively. Use the existing 768-byte payload maximum and CRC. The status/sample serialization is Task 3's job.

- [ ] **Step 1: Write failing C tests.** Cover first-frame fivefold history, exact previous raw action in obs[19:25], zero action before contact, a contact-edge zero sample followed by ACTIVE sample, duplicate/reordered contact rejection, session mismatch, nonfinite action, action deadline at 9/10 ms including uint32 wrap, and abort clearing all outputs.
- [ ] **Step 2: Run the native test driver.** `cd "$H7_REPO_PATH" && "$WHEELLEGGED_PYTHON" tools/test_host.py`; expect the new test to fail before implementation.
- [ ] **Step 3: Implement the pure C state machine and protocol identifiers.** Reuse the 2701 FIFO semantics but maintain a separate state, session and fault namespace; `Confirm_Contact` must bind to the latest published sample and take effect at the next sample boundary.
- [ ] **Step 4: Re-run native tests.** Expect all C/Python host checks to pass; record exact command, exit code and source hashes in a new H7 capture directory.

### Task 2: Pure 500 Hz PD and closed-chain mapping

**Files:**
- Create: `$H7_REPO_PATH/App/lower_policy_control.h`, `$H7_REPO_PATH/App/lower_policy_control.c`
- Create: `$H7_REPO_PATH/tests/test_policy_control.c`
- Modify: `$H7_REPO_PATH/tools/test_host.py`
- Create: `<训练仓根目录>/scripts/agent/check_h7_policy_control.py`

**Interfaces:**
- Produces `bool Lower_Policy_Control_Compute(const float action[6], const float joint_pos[4], const float joint_vel[6], const float jac[2][2], const float motor_limit[6], float virtual_tau[6], float motor_tau[6], uint8_t saturated[6])`.
- `joint_pos` is `[lf0,lf1,rf0,rf1]`; `joint_vel` and `action` use `[lf0,lf1,lfwheel,rf0,rf1,rfwheel]`; `jac[0/1]` is lf/rf with `[front,rear]` derivatives. `motor_tau` is `[DM0,DM1,DM2,DM3,rf/physical-left wheel,lf/physical-right wheel]` before the existing driver-sign boundary. B gives `DM0=J_rf_rear*tau_rf1`, `DM1=tau_rf0+J_rf_front*tau_rf1`, `DM2=-J_lf_rear*tau_lf1`, `DM3=-(tau_lf0+J_lf_front*tau_lf1)`; wheel slots remain `tau_rfwheel,-tau_lfwheel`.
- Reject nonfinite input, invalid Jacobian, or a nonpositive/above-machine motor limit. Clamp virtual tau to the spec limits and mapped motor tau to the supplied physical limits; set each `saturated[i]` to 1 exactly when the corresponding mapped motor request was clipped.

- [ ] **Step 1: Write failing C tests.** Assert zero-action PD produces the expected default-pose leg torque, wheel damping sign, B's lf/rf→DM order and sign with asymmetric `J_front != J_rear`, finite clipping, and rejection of NaN/Inf or invalid limits; a front/rear swap must fail.
- [ ] **Step 2: Run native test driver; expect the new tests to fail.** Use the Task 1 command and save the failing output in a new capture directory.
- [ ] **Step 3: Implement the pure C computation.** Match `sim2sim/mj_sim2sim_ct.py:compute_torques` and the spec's B virtual-work mapping after Task 0's physical identity gate; do not duplicate polarity already applied by `dm.c`/`dji.c`.
- [ ] **Step 4: Implement an independent Python differential check.** `check_h7_policy_control.py` compiles the C module for host use and compares boundary/asymmetric cases against the training-side reference with explicit float32 tolerance; it writes a JSON report and does not open hardware.
- [ ] **Step 5: Run both checks.** Native tests and `"$WHEELLEGGED_PYTHON" scripts/agent/check_h7_policy_control.py --out "$H7_REPO_PATH/captures/20261001-policy-control-check-01/comparison.json"` must exit 0, using a freshly created output directory; retain source hashes and tolerance results.

### Task 3: Integrate guarded execution into H7 application

**Files:**
- Modify: `$H7_REPO_PATH/App/lower_app.c`, `$H7_REPO_PATH/CMakeLists.txt`, `$H7_REPO_PATH/App/lower_safety.c`, `$H7_REPO_PATH/App/lower_safety.h`
- Create: `$H7_REPO_PATH/tests/test_policy_run_integration.c` (HAL/CAN stubs)
- Modify: `$H7_REPO_PATH/tools/test_host.py`, `$H7_REPO_PATH/docs/protocol.md`

**Interfaces:**
- Consume Task 1 session API and Task 2 `Lower_Policy_Control_Compute`.
- New USB execution sample carries a 64-byte metadata header plus 150 float32 values, within the 768-byte limit. Its sixteen `u32` fields are layout, session, sample_seq, sample_ms, tx_ms, applied_action_source_seq, applied_action_rx_ms, phase, fault, online_mask, dm_enabled_mask, period_ms, contact_source_seq, tick_seq, guard_flags, saturation_count. STATUS uses the same 64-byte header without floats.
- The 2 ms tick owns the output and an independent stale-feedback/late-tick/tilt/limit fault path; the 10 ms observation tick owns sample/history publication. Neither path may let `SET_TORQUE`, 2701 or joint bench share the output mode.

- [ ] **Step 1: Write failing integration tests.** Use stubbed motor send functions to assert no nonzero output in OFF/FAULT; queued action loses to remote drop/USB disconnect; missing ACTION faults on next sample; 2 ms tick overrun or stale motor/IMU data faults; STOP zeros and disables; modes are mutually exclusive.
- [ ] **Step 2: Run native tests and expect failure.** Use `tools/test_host.py` and retain output.
- [ ] **Step 3: Integrate session/PD into `lower_app.c`.** Build fresh observation during execution even while armed using Task 0's verified source map; compute Jacobian from the same physical DM snapshot; keep every limit and fault inside the board output path. Keep current diagnostic and bench behavior unchanged.
- [ ] **Step 4: Run native tests and ARM build.** `cmake --preset Debug && cmake --build --preset Debug -j4`; inspect the map for DTCM/RAM_D1 headroom and measure worst observed loop time via board logs later. Save ELF/bin/source hashes. Do not flash yet.

### Task 4: Computer policy runner and offline protocol verification

**Files:**
- Create: `<训练仓根目录>/sim2sim/host_policy_run.py`
- Create: `<训练仓根目录>/scripts/agent/check_policy_run.py`
- Modify: `$H7_REPO_PATH/docs/protocol.md`, `<训练仓根目录>/docs/deployment-contract.md`

**Interfaces:**
- Consume Task 3 execution sample/status layout; use the existing `host_policy_diag.py` model-loading and USB framing patterns without changing the disabled diagnostic.
- CLI requires explicit `--run`, `--port` by stable `/dev/serial/by-id/` name, checkpoint and a new `--out` directory; default invocation only prints preflight. A local terminal `c` sends one `CONTACT_CONFIRM`, `q` sends STOP. The runner refuses non-TTY interactive execution and wrong checkpoint hash/interface.
- Save raw frame bytes, decoded sample/action/event JSONL, action and board timing, firmware/source/checkpoint/config hashes and terminal stop confirmation. Buffer active 100 Hz records outside filesystem writes as in the diagnostic host.

- [ ] **Step 1: Write failing offline checks.** Synthetic frames exercise CRC fragmentation, layout mismatch, model/hash mismatch, precontact zero actions, one contact event, correct first ACTIVE action, missing/late sample, host interruption after action, and STOP/terminal disarm confirmation.
- [ ] **Step 2: Run the check under the configured Python 3.8 environment; expect failure.** `"$WHEELLEGGED_PYTHON" scripts/agent/check_policy_run.py`.
- [ ] **Step 3: Implement runner and verifier.** Use the full checkpoint with `isaacgym` imported before `torch`, CPU single-thread inference, session-bound seq, nonblocking terminal controls and bounded waits; never silently fall back to old HPI1 or 2701 protocol.
- [ ] **Step 4: Run offline checks and a 2701 regression.** The new checker and `"$WHEELLEGGED_PYTHON" sim2sim/host_policy_diag.py --verify "$H7_REPO_PATH/captures/20261001-policy-runtime-normal-02/records.jsonl"` must pass without opening hardware; record exact outputs and SHA-256.

### Task 5: Board bring-up and physical mapping gates

**Files:**
- Modify: `$H7_REPO_PATH/docs/joint-response-validation.md`, `$H7_REPO_PATH/docs/tests.md`
- Create: `<训练仓根目录>/docs/model-deliveries/20261001-chuanliantui-h7-policy-execution.md` and unique new directories under `$H7_REPO_PATH/captures/`

**Interfaces:**
- Consume Tasks 1–4 firmware/host artifacts and the existing joint-response and remote test tools.
- Produce a versioned experimental profile with four DM identity/direction, physical zero, virtual-knee/Jacobian consistency, actual motor limit and board timing evidence. Task 0's disabled B gate is necessary but insufficient: controlled nonzero single-slot direction and magnitude must also pass here. Do not infer unknown values from names, older interrupted trials or the `LOWER_OBS_MAPPING_VERIFIED` flag.

- [ ] **Step 1: Preserve a source and firmware baseline.** Record H7 source hashes, old ELF/bin, board readback, training diff and checkpoint hash in a new capture directory. Check remote double-down, six online, four DM disabled, six targets zero before any flash.
- [ ] **Step 2: Flash the new Debug ELF and verify byte-for-byte readback.** Use the documented 100 kHz pyOCD transport settings. With the DM disabled, collect a fresh observation capture and compare the board's new 25 obs, virtual positions/velocities and Jacobians to Task 0's reference; then run zero-output protocol and sample/history timing tests before permitting torque.
- [ ] **Step 3: Resolve the four-DM bench evidence.** Each nonzero test has an independently configured bound, one motor target, local operator observation, board log and STOP/disarm confirmation. Record failing runs; never auto-repeat or enlarge torque. Verify front/rear virtual-work sign with measured motion and physical zero before unlocking full policy output.
- [ ] **Step 4: Verify execution protection.** Intentionally drop one ACTION and change remote/USB state in zero-output mode; prove fault, zero targets and DM disabled. Measure board 2 ms loop and 10 ms sample timing from board records.
- [ ] **Step 5: Record a gate decision.** Only if all physical mapping, limits and timing pass, mark the experimental profile ready for the first complete-action trial. Otherwise retain the stop reason and remaining evidence gap; do not label standup complete.

### Task 6: Ground standup and result adjudication

**Files:**
- Create: `$H7_REPO_PATH/docs/policy-standup-validation.md`, a new model-delivery record under `<训练仓根目录>/docs/model-deliveries/`, and a new immutable capture directory per trial

**Interfaces:**
- Consume the Task 5 experimental profile, Task 4 host runner, Task 3 firmware and matching full checkpoint.
- Produce a result with exact physical start pose, operator contact event, body-height/attitude observation, action/torque/timing/fault trace and end-state disarm proof.

- [ ] **Step 1: Re-run the formal MuJoCo standup comparison** with `"$WHEELLEGGED_PYTHON" sim2sim/mj_sim2sim_ct.py --standup --render --checkpoint logs/chuanliantui_standup/Sep22_16-26-11_standup_no_legangle_resume/model_6000.pt --cmd_vx 0 --cmd_height 0.20 --friction 0.75` in the configured environment. Record behavior, not merely `--selfcheck`.
- [ ] **Step 2: Run one supported zero-action ground contact rehearsal.** Verify operator `c` maps to the contact edge, with nonzero actions blocked for this rehearsal; STOP and inspect board disarm.
- [ ] **Step 3: Run one short, supervised standup trial** only after Task 5 gate is complete, with fall protection, remote operator and independent body-height observation. The first contact sample remains zero-action; capture 100/500 Hz records and the actual mechanical outcome.
- [ ] **Step 4: Analyze and document.** To pass, show at least 0.18 m height for at least 0.5 s, upright/grounded physical observation, no protection fault, correct timing and verified final disarm. Otherwise document the failure and next physical cause; no automatic rerun or walking command.

## Execution handoff

Plan and spec require user review before implementation. The project rules override the generic skill's worktree and frequent-commit suggestions: operate directly in current workspaces and request explicit confirmation before each Git commit. H7 has no Git and relies on versioned snapshots. Hardware steps require contemporaneous operator observations; no read-only file can substitute for them.
