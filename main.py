import os, sys, math, time, signal, atexit, threading, queue
import numpy as np
import numpy.linalg as la
import pandas as pd
import torch
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────
#  EMG related imports
# ──────────────────────────────────────────────────────────────
from Sensors.EMGSensor import DelsysEMG
from SignalProcessing.Filtering import rt_filtering, rt_net_muscle_activation_lowpass
from Calibration.ApplyCalibration import ApplyCalibration as AC
from Methods.IntegratoinBasedMethods import Method4
import PredictionModels.LSTM as LSTM

# ──────────────────────────────────────────────────────────────
#  MOTOR related imports
# ──────────────────────────────────────────────────────────────
from Motors.DynamixelHardwareInterface import Motors


# ═════════════════════════════════════════════════════════════=
#  Global parameters
# ═════════════════════════════════════════════════════════════=

# ── EMG parameters ─────────────────────────────────────────────────
FS           = 2000
EMG_DT       = 1.0 / FS
USER_NAME    = 'User' # TODO: Change to user name
LSTM_PATH    = "Path to model" # TODO: Change to actual path

EMG_B        = 4.0
EMG_K        = np.pi * 1.4

plot_dq = []

# ── Joint range ──────────────────────────────────────────────────
THETA_MIN       = np.deg2rad(0)
THETA_MAX       = np.deg2rad(140)
THETA_RANGE     = THETA_MAX - THETA_MIN

# ── Controller parameters ────────────────────────────────────────────────
plot_q   = []
plot_tau = []
SAMPLE_RATE  = 200
DT           = 1.0 / SAMPLE_RATE
TORQUE_MAX   = 10.6
TORQUE_MIN   = -TORQUE_MAX

# ── Filter parameters ──────────────────────────────────────────────────
VEL_FILTER_ALPHA_CTRL = 0.5
VEL_FILTER_ALPHA_ACC  = 0.92
ACC_FILTER_ALPHA      = 0.70
TAU_FILTER_ALPHA      = 0.01
DDTHETA_SMOOTH_N      = 5
N_LAG                 = 3

# ── Motor parameters ──────────────────────────────────────────────────
MOTOR_PORT       = 'COM4'
MOTOR_BAUD       = 4_500_000
TORQUE_DIRECTION = 1

# ── Calibration parameters ──────────────────────────────────────────────────
SINE_CENTER_DEG  = 0.0
RAW_MIN          = -15427
RAW_MAX          = -2922
ANGLE_RANGE_DEG  = 140.0
RAW_RANGE        = RAW_MAX - RAW_MIN
VEL_UNIT_RAD_S   = 0.229 * 2.0 * math.pi / 60.0

# ── Experiment parameters ──────────────────────────────────────────────────
NUM_TRIALS       = 3
TRIAL_DURATION_S = 30

# ── Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Global stop event ──────────────────────────────────────────────
stop_event = threading.Event()


# ==============================================================
#  PID controller
# ==============================================================

class PositionTorquePID:
    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        torque_min: float,
        torque_max: float,
        integral_min: float = -2.0,
        integral_max: float = 2.0,
        deadband_rad: float = np.deg2rad(0.5),
        derivative_filter_alpha: float = 0.8,
    ):
        """
        PID controller for position control using torque output.

        Inputs:
            q_d: desired position in radians
            q: actual position in radians
            dq: actual velocity in rad/s
            dt: timestep in seconds

        Output:
            torque command in Nm
        """

        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.torque_min = torque_min
        self.torque_max = torque_max

        self.integral_min = integral_min
        self.integral_max = integral_max

        self.deadband_rad = deadband_rad
        self.derivative_filter_alpha = derivative_filter_alpha

        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_derivative = 0.0
        self.first_update = True

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_derivative = 0.0
        self.first_update = True

    def update(self, q_d: float, q: float, dq: float, dt: float) -> float:
        if dt <= 0.0:
            return 0.0

        # Clamp desired position to safe joint range
        q_d = float(np.clip(q_d, THETA_MIN, THETA_MAX))
        q = float(q)

        error = q_d - q

        # Small deadband to avoid buzzing around target
        if abs(error) < self.deadband_rad:
            error = 0.0

        # Proportional torque
        p_term = self.kp * error

        # Integral torque with anti-windup
        self.integral += error * dt
        self.integral = float(np.clip(
            self.integral,
            self.integral_min,
            self.integral_max
        ))
        i_term = self.ki * self.integral

        # Derivative term
        #
        # For position control, using -dq is often better than differentiating
        # noisy encoder position error.
        #
        # If q_d changes quickly, this ignores desired velocity. That is okay
        # for a simple first version.
        derivative = -dq

        self.filtered_derivative = (
            self.derivative_filter_alpha * self.filtered_derivative
            + (1.0 - self.derivative_filter_alpha) * derivative
        )

        d_term = self.kd * self.filtered_derivative

        torque = p_term + i_term + d_term
        torque = float(np.clip(torque, self.torque_min, self.torque_max))

        self.prev_error = error

        return torque

# ═════════════════════════════════════════════════════════════=
#  EMG thread (producer, 2000 Hz)
# ═════════════════════════════════════════════════════════════=
def emg_thread_fn(qd_queue: queue.Queue):
    """
    Collect EMG → filter → activation → Method 4 → LSTM → predicted_angle
    Write results to qd_queue; controller reads the latest value.
    """
    filter_bicep  = rt_filtering(FS, 450, 20, 2)
    filter_tricep = rt_filtering(FS, 450, 20, 2)
    net_a_lowpass = rt_net_muscle_activation_lowpass(FS, lp_cutoff=2, order=2)

    interpreter = AC(theta_min=THETA_MIN, theta_max=THETA_MAX, user_name=USER_NAME, BicepEMG=True, TricepEMG=True)

    Bicep_RMS_queue  = queue.Queue(maxsize=50)
    Tricep_RMS_queue = queue.Queue(maxsize=50)

    model = LSTM.LSTMModel(input_size=1, hidden_size=64, output_size=1, num_layers=1, batch_first=True).to(device)
    model.load_state_dict(torch.load(LSTM_PATH, map_location=device))
    model.eval()

    window         = deque(maxlen=100)
    optimized_angle = float(np.deg2rad(SINE_CENTER_DEG))
    sample_counter  = 0

    emg = DelsysEMG(channel_range=(0, 1))
    emg.start()
    print("[EMG] Thread started, beginning acquisition...")
    pt_analysis = []

    while not stop_event.is_set():
        pt_start = time.time()
        reading    = emg.read()
        sample_counter += 1

        filtered_bicep  = filter_bicep.bandpass(reading[0])
        filtered_tricep = filter_tricep.bandpass(reading[1])

        if Bicep_RMS_queue.full():
            Bicep_RMS_queue.get_nowait()
        Bicep_RMS_queue.put_nowait(filtered_bicep)
        if Tricep_RMS_queue.full():
            Tricep_RMS_queue.get_nowait()
        Tricep_RMS_queue.put_nowait(filtered_tricep)

        Bicep_RMS  = np.sqrt(np.mean(np.array(list(Bicep_RMS_queue.queue))**2))
        Tricep_RMS = np.sqrt(np.mean(np.array(list(Tricep_RMS_queue.queue))**2))

        filtered_bicep_rms  = float(filter_bicep.lowpass(np.atleast_1d(Bicep_RMS))[0])
        filtered_tricep_rms = float(filter_tricep.lowpass(np.atleast_1d(Tricep_RMS))[0])

        activation = interpreter.compute_activation([filtered_bicep_rms, filtered_tricep_rms])
        net_a = activation[0] - activation[1]

        filtered_net_a = float(net_a_lowpass.lowpass(np.atleast_1d(net_a))[0])

        optimized_angle = Method4(
            np.pi*0.9, filtered_net_a, EMG_DT,
            optimized_angle, THETA_MIN, THETA_MAX
        )

        window.append(optimized_angle)
        if len(window) < window.maxlen:
            continue

        if len(window) == window.maxlen and sample_counter % 10 == 0:
            with torch.inference_mode():
                input_tensor = torch.as_tensor(window, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
                lstm_output     = model(input_tensor)
                predicted_angle = float(lstm_output.detach().cpu().item())

            try:
                qd_queue.put_nowait(predicted_angle)
            except queue.Full:
                qd_queue.get_nowait()
                qd_queue.put_nowait(predicted_angle)

        pt_end = time.time()
        pt_duration = pt_end - pt_start
        pt_analysis.append(pt_duration)

    emg.stop()
    Bicep_RMS_queue.queue.clear()
    Tricep_RMS_queue.queue.clear()
    print(f"EMG average processing time: {np.mean(pt_analysis)*1000:.4f} ms per sample")
    print("[EMG] Thread stopped.")


# ═════════════════════════════════════════════════════════════=
#  Motor interface
# ═════════════════════════════════════════════════════════════=
class Motor:
    def __init__(self):
        _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        if _SCRIPT_DIR not in sys.path:
            sys.path.insert(0, _SCRIPT_DIR)

        if not getattr(Motors, '_patched', False):
            _orig = Motors.__init__
            def _safe(s, *a, **kw):
                s.num_motors = 0; s.motor_ids = []
                _orig(s, *a, **kw)
            Motors.__init__ = _safe
            Motors._patched = True

        print(f"[Motor] Connecting: {MOTOR_PORT} @ {MOTOR_BAUD}...")
        self._m   = Motors(port=MOTOR_PORT, baudrate=MOTOR_BAUD)
        if not self._m.motor_ids:
            raise RuntimeError("No motors found!")
        self._mid = self._m.motor_ids[0]
        print(f"[Motor] ✅ ID={self._mid} connected")

        self._set_current_mode()

        raw_now        = self._raw_signed()
        self._raw_zero = raw_now - int(SINE_CENTER_DEG / ANGLE_RANGE_DEG * RAW_RANGE)
        deg_now        = self._raw_to_deg(raw_now)
        print(f"[Motor] Zero offset calibration: current={deg_now:.1f}°  (centered={deg_now - SINE_CENTER_DEG:.1f}°)")

        atexit.register(self.stop)

    def _raw_signed(self):
        v = int(self._m.get_position(motor_id=self._mid))
        return v - 4294967296 if v > 2147483647 else v

    def _raw_to_deg(self, raw_signed):
        return float((raw_signed - self._raw_zero) / RAW_RANGE * ANGLE_RANGE_DEG)

    def _deg_to_raw(self, deg):
        return int(deg / ANGLE_RANGE_DEG * RAW_RANGE + self._raw_zero)

    def _set_current_mode(self):
        try:
            from dynamixel_sdk import PacketHandler, PortHandler
            ph = PortHandler(MOTOR_PORT)
            pk = PacketHandler(2.0)
            if ph.openPort() and ph.setBaudRate(MOTOR_BAUD):
                pk.write1ByteTxRx(ph, self._mid, 64, 0)
                time.sleep(0.1)
                pk.write1ByteTxRx(ph, self._mid, 11, 0)
                time.sleep(0.1)
                pk.write1ByteTxRx(ph, self._mid, 64, 1)
                ph.closePort()
                print("[Motor] ✅ Current control mode")
        except Exception as e:
            print(f"[Motor] [WARN] Mode set failed: {e}")

    def read(self):
        for attempt in range(5):
            try:
                raw     = self._raw_signed()
                vel_raw = float(self._m.get_velocity(motor_id=self._mid))
                if vel_raw > 1023: vel_raw -= 2048
                deg_abs = self._raw_to_deg(raw)
                theta   = math.radians(deg_abs)
                dtheta  = vel_raw * VEL_UNIT_RAD_S
                return theta, dtheta
            except Exception as e:
                if attempt == 4:
                    raise RuntimeError(f"Failed to read 5 times in a row: {e}")
                time.sleep(0.1 * (attempt + 1))

    def send(self, tau: float):
        tau = float(np.clip(tau, TORQUE_MIN, TORQUE_MAX))
        try:
            deg_abs = self._raw_to_deg(self._raw_signed())
        except Exception:
            deg_abs = SINE_CENTER_DEG
        # Soft limit: zero torque when exceeding joint range
        if deg_abs <= 0.5 and tau * TORQUE_DIRECTION < 0:
            tau = 0.0
        elif deg_abs >= ANGLE_RANGE_DEG - 0.5 and tau * TORQUE_DIRECTION > 0:
            tau = 0.0
        try:
            self._m.sendMotorCommand(
                self._mid, self._m.torq2curcom(tau * TORQUE_DIRECTION)
            )
        except Exception as e:
            print(f"[Motor] [WARN] Send failed: {e}")

    def stop(self):
        for _ in range(3):
            try:
                self._m.sendMotorCommand(self._mid, 0)
                time.sleep(0.02)
            except Exception:
                pass

    def home(self, target_deg=SINE_CENTER_DEG, gain=0.5, tol=1.0, max_steps=500):
        """Center to target_deg (absolute angle)"""
        print(f"[Motor] Homing to {target_deg:.1f}°...")
        for _ in range(max_steps):
            if stop_event.is_set():
                break
            try:
                theta, _ = self.read()
                err      = target_deg - math.degrees(theta)
                self.send(float(np.clip(err * gain, -5.0, 5.0)))
                if abs(err) < tol:
                    break
            except Exception:
                pass
            time.sleep(0.05)
        self.stop()
        time.sleep(0.8)
        theta, _ = self.read()
        print(f"[Motor] Homing complete: {math.degrees(theta):.2f}°")

# ═════════════════════════════════════════════════════════════=
#  EMG reference source
# ═════════════════════════════════════════════════════════════=
class EMGReference:
    """
    Read the latest EMG target angle from `qd_queue`.
    Hold last value when no new data; estimate dtheta_d and ddtheta_d via numerical differentiation.
    """
    def __init__(self, qd_queue: queue.Queue):
        self._queue     = qd_queue
        self._theta_d   = float(np.deg2rad(SINE_CENTER_DEG))
        self._dtheta_d  = 0.0
        self._ddtheta_d = 0.0
        self._prev_theta_d  = self._theta_d
        self._prev_dtheta_d = 0.0
        self._vel_alpha = 0.3
        self._acc_alpha = 0.3

    def update(self) -> tuple[float, float, float]:
        latest = None
        while True:
            try:
                latest = self._queue.get_nowait()
            except queue.Empty:
                break

        if latest is not None:
            new_theta_d = float(latest)
            raw_vel = (new_theta_d - self._prev_theta_d) / DT
            self._dtheta_d = (self._vel_alpha * raw_vel
                              + (1 - self._vel_alpha) * self._dtheta_d)
            raw_acc = (self._dtheta_d - self._prev_dtheta_d) / DT
            self._ddtheta_d = (self._acc_alpha * raw_acc
                               + (1 - self._acc_alpha) * self._ddtheta_d)
            self._prev_theta_d  = new_theta_d
            self._prev_dtheta_d = self._dtheta_d
            self._theta_d       = new_theta_d

        return self._theta_d, self._dtheta_d, self._ddtheta_d

    def current_theta_d(self) -> float:
        return self._theta_d

pt = []

# ═════════════════════════════════════════════════════════════=
#  Single trial
# ═════════════════════════════════════════════════════════════=
def run_trial(motor: Motor, qd_queue: queue.Queue, trial_num: int, duration_s: float) -> dict | None:
    motor.home()
    lowpass_dq = rt_net_muscle_activation_lowpass(106, lp_cutoff=2)
    ref = EMGReference(qd_queue)

    errors_deg: list[float] = []          # For stats/printing (degrees)

    # PID
    pid = PositionTorquePID(kp=5.0, ki=0.0, kd=0.01, torque_min=TORQUE_MIN, torque_max=TORQUE_MAX)

    print(f"\n[Trial {trial_num}] ▶ Start, duration={duration_s}s")
    t_start = t_last = time.time()

    while True:
        now     = time.time()
        elapsed = now - t_start
        if elapsed >= duration_s or stop_event.is_set():
            break

        dt_actual = now - t_last
        pt.append(dt_actual)
        if dt_actual < DT:
            time.sleep(DT - dt_actual)
            continue
        t_last = now

        # ── Reference trajectory: EMG input ─────────────────────────────────
        theta_d, _, _ = ref.update()
        theta_d = float(lowpass_dq.lowpass(np.atleast_1d(theta_d)))

        # ── Read motor ──────────────────────────────────────────
        try:
            theta, dtheta = motor.read()
        except RuntimeError as e:
            print(f"[Trial {trial_num}] [WARN] {e}, skipping this step")
            motor.stop()
            time.sleep(0.3)
            continue

        plot_q.append(theta)
        plot_dq.append(theta_d)

        motorcom = pid.update(theta_d, theta, dtheta, DT)

        motor.send(motorcom)

        plot_tau.append(motorcom)

        # ── Logging ──────────────────────────────────────────────
        errors_deg.append(abs(math.degrees(theta_d - theta)))

    print(f"Processing time per step: mean={np.mean(pt)*1000:.2f}ms")
    motor.stop()

    result = dict(track_rmse = float(np.sqrt(np.mean(np.square(errors_deg)))), max_err = float(np.max(errors_deg)))
    print(f"\n[Trial {trial_num}] Results:")
    print(f"    Tracking RMSE = {result['track_rmse']:.3f}°")
    print(f"    Max error     = {result['max_err']:.2f}°")
    return result


# ══════════════════════════════════════════════════════════════
#  Summary print
# ══════════════════════════════════════════════════════════════
def print_summary(results: list):
    print(f"\n{'='*55}")
    print(f"  Summary  —  ada_imp_con + ILC (AAN)")
    print(f"  Num trials: {len(results)}")
    print(f"{'='*55}")
    track_rmse = [r['track_rmse'] for r in results]
    max_err = [r['max_err'] for r in results]
    print(f"  Mean tracking RMSE: {np.mean(track_rmse):.3f}°")
    print(f"  Mean max error:     {np.mean(max_err):.2f}°")
    print(f"{'='*55}\n")


# ══════════════════════════════════════════════════════════════
#  Entry
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    num_trials = int(sys.argv[1]) if len(sys.argv) > 1 else NUM_TRIALS

    print(f"\n{'='*55}")
    print(f"  Num trials: {num_trials}  Duration: {TRIAL_DURATION_S}s each")
    print(f"{'='*55}")

    # Connect motor
    try:
        motor = Motor()
    except Exception as e:
        print(f"\n❌ Motor connection failed: {e}")
        sys.exit(1)
    time.sleep(1.0)

    # Signal processing
    def _on_interrupt(*_):
        print("\n[Interrupt] Performing safe stop...")
        stop_event.set()
        motor.stop()

    signal.signal(signal.SIGINT,  _on_interrupt)
    signal.signal(signal.SIGTERM, _on_interrupt)

    # Shared queue
    qd_queue = queue.Queue(maxsize=2)

    # Start EMG thread
    emg_thread = threading.Thread(target=emg_thread_fn, args=(qd_queue,), name="EMG-Thread", daemon=True)
    emg_thread.start()

    print("[Main] Waiting for EMG initialization...")
    while qd_queue.empty() and not stop_event.is_set():
        time.sleep(0.1)
    print("[Main] EMG ready, starting trials.")

    all_results = []
    for i in range(num_trials):
        if stop_event.is_set():
            break
        print(f"\n  Press Enter to start trial {i+1}/{num_trials} (Ctrl+C to exit)...")
        try:
            input()
        except (EOFError, KeyboardInterrupt):
            break

        res = run_trial(motor = motor, qd_queue = qd_queue, trial_num = i + 1, duration_s = TRIAL_DURATION_S)
        if res:
            all_results.append(res)

        print(f"length of qd {len(plot_dq)}, length of q {len(plot_q)}, "
              f"operational frequency: {len(plot_q)/TRIAL_DURATION_S:.2f} Hz")
        t_qd = np.arange(len(plot_dq)) * DT
        t_q  = np.arange(len(plot_q))  * DT

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(t_qd, plot_dq, label='θ_d (rad)')
        plt.xlabel('Time (s)'); plt.ylabel('Angle (rad)')
        plt.subplot(1, 2, 2)
        plt.plot(t_q, plot_q, label='θ (rad)')
        plt.xlabel('Time (s)'); plt.ylabel('Angle (rad)')
        plt.tight_layout()
        plt.show()

        plot_dq.clear()
        plot_q.clear()
        plot_tau.clear()

    # Stop
    stop_event.set()
    motor.stop()
    emg_thread.join(timeout=3.0)

    if all_results:
        print_summary(all_results)

    print("Program finished, motors safely stopped.")