from Methods.IntegratoinBasedMethods import *
from Methods.pDMPMethods import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import time

mpl.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'text.latex.preamble': r'\usepackage{amsmath}',
    
    'font.size': 10,          # default text size
    'axes.titlesize': 14,     # title
    'axes.labelsize': 12,     # x and y labels
    'xtick.labelsize': 10,    # x tick labels
    'ytick.labelsize': 10,    # y tick labels
    'legend.fontsize': 10,    
    'figure.titlesize': 16
})

THETA_MIN = np.deg2rad(0)
THETA_MAX = np.deg2rad(140)

def compute_jerk_metrics(j):

    abs_j = np.abs(j)

    metrics = {
        "mean": np.mean(abs_j),
        "median": np.median(abs_j),
        "sigma": np.std(abs_j),
        "max": np.max(abs_j),
        "q25": np.percentile(abs_j, 25),
        "q75": np.percentile(abs_j, 75),
    }

    return j, abs_j, metrics

labels = [
        "Method 3",
        "Method 4",
        "Method 5",
        "Method 6",
        "Method 7",
        "Method 8",
        "Method 9",
        "Method 10",
        "Method 11",
        "Method 12"
    ]

Method3Traj = []
Method4Traj = []
Method5Traj = []
Method6Traj = []
Method7Traj = []
Method8Traj = []
Method9Traj = []
Method10Traj = []
Method11Traj = []
Method12Traj = []

FS = 2000 # EMG
if __name__ == "__main__":
    print("Starting EMG optimization test at 2000 Hz...")
    print(f"Theta max: {THETA_MAX}, Theta min: {THETA_MIN}")
    # Generate test muscle activations (EMG signal) using sinewave between -1 and 1
    time_v= np.linspace(0, 20, FS*20)  # Time vector from 0 to 20 seconds
    activation = np.sin(2 * np.pi * 0.15 * time_v)  # Sine wave with frequency of 0.2 Hz

    # Small random noise
    rng = np.random.default_rng(seed=42)
    noise = rng.normal(0, 1, size=time_v.shape)

    # Smooth it with a moving average
    window_size = 100  # increase for smoother wobble
    kernel = np.ones(window_size) / window_size
    smooth_noise = np.convolve(noise, kernel, mode="same")

    # Scale the noise so it only creates a small wobble
    noise_amplitude = 0.06

    activation += noise_amplitude * smooth_noise

    activation = np.clip(activation, -1, 1)

    # calculate the difference in activations
    activation_diff = np.diff(activation, prepend=activation[0]) / (1/FS)

    # Plot activation and activation difference
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time_v, activation, label='Activation')
    plt.xlabel('Time (s)')
    plt.ylabel('Activation')
    plt.title('Muscle Activation (EMG Signal)')
    plt.subplot(2, 1, 2)
    plt.plot(time_v, activation_diff, label='Activation Difference', color='orange')
    plt.xlabel('Time (s)')
    plt.ylabel('Activation Difference')
    plt.title('Difference of Muscle Activation')
    plt.tight_layout()
    plt.show()

    # Create empty lists to store optimized angles for each optimizer
    Method3Angles = []
    Method4Angles = []
    Method5Angles = []
    Method6Angles = []
    Method7Angles = []
    Method8Angles = []
    Method11Angles = []
    Method12Angles = []
    
    # Initialize parameters for the optimizers along with the optimizers themselves
    k = (1.2*np.pi) / 3 # EMG
    t = 1/FS  # Time between updates (seconds)
    q = 0  # Initial angle (degrees)
    Method3Angles.append(q)
    for a in activation:
        Method3Angles.append(Method3(k, a, t, Method3Angles[-1], THETA_MIN, THETA_MAX))

    print(f"maximum angle for optimizer 1: {np.rad2deg(max(Method3Angles)):.2f} degrees, minimum angle for optimizer 1: {np.rad2deg(min(Method3Angles)):.2f} degrees")

    # k= 2 * np.pi # EMG
    k = np.pi * 0.9
    Method4Angles.append(q)
    for a in activation:
        Method4Angles.append(Method4(k, a, t, Method4Angles[-1], THETA_MIN, THETA_MAX))
    print(f"maximum angle for optimizer 2: {np.rad2deg(max(Method4Angles)):.2f} degrees, minimum angle for optimizer 2: {np.rad2deg(min(Method4Angles)):.2f} degrees")
    
    k = (1.6*np.pi) / 4 # EMG
    # k = (1.4*np.pi)/3
    Method5Angles.append(q)
    delta_q_prev = 0
    for a in activation:
        optimized_angle, delta_q_prev = Method5(k, a, t, Method5Angles[-1], delta_q_prev, THETA_MIN, THETA_MAX)
        Method5Angles.append(optimized_angle)
    print(f"maximum angle for optimizer 4: {np.rad2deg(max(Method5Angles)):.2f} degrees, minimum angle for optimizer 4: {np.rad2deg(min(Method5Angles)):.2f} degrees")
    
    k = 0 # EMG
    # k = np.pi / 4
    n = (1.3*np.pi) / 3
    b = 0.01 # 0.001
    Method6Angles.append(q)
    for a in activation:
        q_next, k = Method6(a, k, t, Method6Angles[-1], THETA_MIN, THETA_MAX, np.pi, n, b)
        Method6Angles.append(q_next)
    print(f"maximum angle for optimizer 5: {np.rad2deg(max(Method6Angles)):.2f} degrees, minimum angle for optimizer 5: {np.rad2deg(min(Method6Angles)):.2f} degrees")
    
    v = 0  # Initial velocity
    k = np.pi * 1.6
    b = 4
    Method7Angles.append(q)
    for a in activation:
        q_next, v, acc = Method7(a, v, t, Method7Angles[-1], THETA_MIN, THETA_MAX, np.pi, b, k)
        Method7Angles.append(q_next)
    print(f"maximum angle for optimizer 6: {np.rad2deg(max(Method7Angles)):.2f} degrees, minimum angle for optimizer 6: {np.rad2deg(min(Method7Angles)):.2f} degrees")

    Method8Angles.append(q)
    kn = 4
    kd = 4
    b = 4
    for a, da in zip(activation, activation_diff):
        q_next, v, acc = Method8(a, da, v, kn, kd, b, Method8Angles[-1], THETA_MIN, THETA_MAX, np.pi, t)
        Method8Angles.append(q_next)
    print(f"maximum angle for optimizer 7: {np.rad2deg(max(Method8Angles)):.2f} degrees, minimum angle for optimizer 7: {np.rad2deg(min(Method8Angles)):.2f} degrees")

    #========================================= DMP's ====================================
    # Teach DMP
    dt = 1/FS
    phi = 0
    tau = 0.5
    DMP = Method11(DOF=1, N=25, alpha=8, beta=2, lambd=0.9, dt=dt)
    DMP.set_output_limits(THETA_MIN, THETA_MAX, squash_gain=1.0)
    DMP.set_output_state(np.array([0.0]))
    y_old = 0
    dy_old = 0
    start_time = time.time()
    while time.time() - start_time < 3:  # Teach for 3 seconds
        print(f"elapsed time: {time.time() - start_time:.2f} seconds", end='\r')
        phi += 2*np.pi * dt/tau #16*np.pi * dt/tau
        y = np.array([0])
        dy = (y - y_old) / dt 
        ddy = (dy - dy_old) / dt
        DMP.set_phase(np.array([phi]))
        DMP.set_period(np.array([tau]))
        DMP.learn(y, dy, ddy)
        DMP.integration()

        # old values	
        y_old = y
        dy_old = dy
            
        # store data for plotting
        x, dx, ph, ta = DMP.get_state()

    # Run DMP
    v = np.pi/35 #np.pi/22
    # v = np.pi/2
    for a in activation:
        DMP.set_phase(np.array([phi]))
        DMP.set_period(np.array([tau]))

        U = np.asarray([a*v])  # EMG activation as input
        DMP.update(U)
        DMP.integration()
        x, dx, ph, ta = DMP.get_state()
        Method11Angles.append(x[0])

    print(f"maximum angle for DMP: {np.rad2deg(max(Method11Angles)):.2f} degrees, minimum angle for DMP: {np.rad2deg(min(Method11Angles)):.2f} degrees")

    # Teach Coupled DMP
    DMP = Method12(DOF=1, N=25, alpha=8, beta=2, lambd=0.9, dt=dt)
    DMP.set_output_limits(THETA_MIN, THETA_MAX, squash_gain=1.0)
    DMP.set_output_state(np.array([0.0]))
    # Teach DMP 0 trajectory for 3s
    y_old = 0
    dy_old = 0
    start_time = time.time()
    while time.time() - start_time < 3:  # Teach for 3 seconds
        print(f"elapsed time: {time.time() - start_time:.2f} seconds", end='\r')
        phi += 2*np.pi * dt/tau
        y = np.array([0])
        dy = (y - y_old) / dt 
        ddy = (dy - dy_old) / dt
        DMP.set_phase(np.array([phi]))
        DMP.set_period(np.array([tau]))
        DMP.learn(y, dy, ddy)

        # old values	
        y_old = y
        dy_old = dy
            
        # store data for plotting
        x, dx, ph, ta = DMP.get_state()

    # Run Coupled DMP
    for a in activation:
        DMP.set_phase(np.array([phi]))
        DMP.set_period(np.array([tau]))

        DMP.repeat()

        DMP.integration(np.array([a]))

        x, dx, ph, ta = DMP.get_state()
        Method12Angles.append(x[0])

    print(f"maximum angle for Coupled DMP: {np.rad2deg(max(Method12Angles)):.2f} degrees, minimum angle for Coupled DMP: {np.rad2deg(min(Method12Angles)):.2f} degrees")


    t = 1/FS

    # Remove the initial angle from the optimized angles lists
    Method3Angles.remove(Method3Angles[0])
    Method4Angles.remove(Method4Angles[0])
    Method5Angles.remove(Method5Angles[0])
    Method6Angles.remove(Method6Angles[0])
    Method7Angles.remove(Method7Angles[0])
    Method8Angles.remove(Method8Angles[0])

    Method3Traj.extend(Method3Angles)
    Method4Traj.extend(Method4Angles)
    Method5Traj.extend(Method5Angles)
    Method6Traj.extend(Method6Angles)
    Method7Traj.extend(Method7Angles)
    Method8Traj.extend(Method8Angles)
    Method11Traj.extend(Method11Angles)
    Method12Traj.extend(Method12Angles)
    

    # Calculate the velocity, acceleration and jerk for each optimizer
    velocities_1 = np.gradient(Method3Angles, t)
    accelerations_1 = np.gradient(velocities_1, t)
    jerks_1 = np.gradient(accelerations_1, t)

    velocities_2 = np.gradient(Method4Angles, t)
    accelerations_2 = np.gradient(velocities_2, t)
    jerks_2 = np.gradient(accelerations_2, t)

    velocities_4 = np.gradient(Method5Angles, t)
    accelerations_4 = np.gradient(velocities_4, t)
    jerks_4 = np.gradient(accelerations_4, t)

    velocities_5 = np.gradient(Method6Angles, t)
    accelerations_5 = np.gradient(velocities_5, t)
    jerks_5 = np.gradient(accelerations_5, t)

    velocities_6 = np.gradient(Method7Angles, t)
    accelerations_6 = np.gradient(velocities_6, t)
    jerks_6 = np.gradient(accelerations_6, t)

    velocities_7 = np.gradient(Method8Angles, t)
    accelerations_7 = np.gradient(velocities_7, t)
    jerks_7 = np.gradient(accelerations_7, t)

    DMP_velocities = np.gradient(Method11Angles, t)
    DMP_accelerations = np.gradient(DMP_velocities, t)
    DMP_jerks = np.gradient(DMP_accelerations, t)

    DMP_Coupled_velocities = np.gradient(Method12Angles, t)
    DMP_Coupled_accelerations = np.gradient(DMP_Coupled_velocities, t)
    DMP_Coupled_jerks = np.gradient(DMP_Coupled_accelerations, t)

    # Plot each optimized angle in different graphs comparing them to the input signal and with the position, velocity, acceleration and jerk.
    plt.figure(figsize=(12, 10))
    plt.title("Optimizer 1: EMG")
    plt.subplot(5, 1, 1)
    plt.plot(time_v, activation, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.xlim(time_v[0], time_v[-1])

    plt.subplot(5, 1, 2)
    plt.plot(time_v, Method3Angles, label="Optimized Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Optimized Angle (rad)")
    plt.xlim(time_v[0], time_v[-1])

    plt.subplot(5, 1, 3)
    plt.plot(time_v, velocities_1, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")
    plt.xlim(time_v[0], time_v[-1])

    plt.subplot(5, 1, 4)
    plt.plot(time_v, accelerations_1, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (rad/$s^2$)")
    plt.xlim(time_v[0], time_v[-1])

    plt.subplot(5, 1, 5)
    plt.plot(time_v, jerks_1, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (rad/$s^3$)")
    plt.xlim(time_v[0], time_v[-1])
    plt.tight_layout()
    plt.show()

    #-----------------------------------------------------------------

    plt.figure(figsize=(12, 10))
    plt.title("Optimizer 2: EMG")
    plt.subplot(5, 1, 1)
    plt.plot(time_v, activation, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.xlim(time_v[0], time_v[-1])
    
    plt.subplot(5, 1, 2)
    plt.plot(time_v, Method4Angles, label="Optimized Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Optimized Angle (rad)")
    plt.xlim(time_v[0], time_v[-1])
    
    plt.subplot(5, 1, 3)
    plt.plot(time_v, velocities_2, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")
    plt.xlim(time_v[0], time_v[-1])

    plt.subplot(5, 1, 4)
    plt.plot(time_v, accelerations_2, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (rad/$s^2$)")
    plt.xlim(time_v[0], time_v[-1])

    plt.subplot(5, 1, 5)
    plt.plot(time_v, jerks_2, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (rad/$s^3$)")
    plt.xlim(time_v[0], time_v[-1])
    plt.tight_layout()
    plt.show()

    #-----------------------------------------------------------------

    plt.figure(figsize=(12, 10))
    plt.title("Optimizer 4: EMG")
    plt.subplot(5, 1, 1)
    plt.plot(time_v, activation, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.xlim(time_v[0], time_v[-1])
    
    plt.subplot(5, 1, 2)
    plt.plot(time_v, Method5Angles, label="Optimized Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Optimized Angle (rad)")
    plt.xlim(time_v[0], time_v[-1])
    
    plt.subplot(5, 1, 3)
    plt.plot(time_v, velocities_4, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")
    plt.xlim(time_v[0], time_v[-1])

    plt.subplot(5, 1, 4)
    plt.plot(time_v, accelerations_4, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (rad/$s^2$)")
    plt.xlim(time_v[0], time_v[-1])

    plt.subplot(5, 1, 5)
    plt.plot(time_v, jerks_4, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (rad/$s^3$)")
    plt.xlim(time_v[0], time_v[-1])
    plt.tight_layout()
    plt.show()

    #-----------------------------------------------------------------

    plt.figure(figsize=(12, 10))
    plt.title("Optimizer 5: EMG")
    plt.subplot(5, 1, 1)
    plt.plot(time_v, activation, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.xlim(time_v[0], time_v[-1])

    plt.subplot(5, 1, 2)
    plt.plot(time_v, Method6Angles, label="Optimized Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Optimized Angle (rad)")
    plt.xlim(time_v[0], time_v[-1])

    plt.subplot(5, 1, 3)
    plt.plot(time_v, velocities_5, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")
    plt.xlim(time_v[0], time_v[-1])

    plt.subplot(5, 1, 4)
    plt.plot(time_v, accelerations_5, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (rad/$s^2$)")
    plt.xlim(time_v[0], time_v[-1])

    plt.subplot(5, 1, 5)
    plt.plot(time_v, jerks_5, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (rad/$s^3$)")
    plt.xlim(time_v[0], time_v[-1])
    plt.tight_layout()
    plt.show()

    #-----------------------------------------------------------------

    plt.figure(figsize=(12, 10))
    plt.title("Optimizer 6: EMG")
    plt.subplot(5, 1, 1)
    plt.plot(time_v, activation, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 2)
    plt.plot(time_v, Method7Angles, label="Optimized Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Optimized Angle (rad)")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 3)
    plt.plot(time_v, velocities_6, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 4)
    plt.plot(time_v, accelerations_6, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (rad/$s^2$)")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 5)
    plt.plot(time_v, jerks_6, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (rad/$s^3$)")
    plt.xlim(time_v[0], time_v[-1])
    plt.tight_layout()
    plt.show()

    #-----------------------------------------------------------------

    plt.figure(figsize=(12, 10))
    plt.title("Optimizer 7: EMG with PD control and acceleration term")
    plt.subplot(5, 1, 1)
    plt.plot(time_v, activation, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 2)
    plt.plot(time_v, Method8Angles, label="Optimized Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Optimized Angle (rad)")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 3)
    plt.plot(time_v, velocities_7, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 4)
    plt.plot(time_v, accelerations_7, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (rad/$s^2$)")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 5)
    plt.plot(time_v, jerks_7, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (rad/$s^3$)")
    plt.xlim(time_v[0], time_v[-1])
    plt.tight_layout()
    plt.show()

    #-----------------------------------------------------------------

    plt.figure(figsize=(12, 10))
    plt.title("pDMP")
    plt.subplot(5, 1, 1)
    plt.plot(time_v, activation, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 2)
    plt.plot(time_v, Method11Angles, label="DMP Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("DMP Angle (rad)")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 3)
    plt.plot(time_v, DMP_velocities, label="DMP Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("DMP Velocity (rad/s)")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 4)
    plt.plot(time_v, DMP_accelerations, label="DMP Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("DMP Acceleration (rad/$s^2$)")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 5)
    plt.plot(time_v, DMP_jerks, label="DMP Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("DMP Jerk (rad/$s^3$)")
    plt.xlim(time_v[0], time_v[-1])
    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------------------

    plt.figure(figsize=(12, 10))
    plt.title("pDMP Coupled")
    plt.subplot(5, 1, 1)
    plt.plot(time_v, activation, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 2)
    plt.plot(time_v, Method12Angles, label="DMP Coupled Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("DMP Coupled Angle (rad)")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 3)
    plt.plot(time_v, DMP_Coupled_velocities, label="DMP Coupled Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("DMP Coupled Velocity (rad/s)")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 4)
    plt.plot(time_v, DMP_Coupled_accelerations, label="DMP Coupled Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("DMP Coupled Acceleration (rad/$s^2$)")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 5)
    plt.plot(time_v, DMP_Coupled_jerks, label="DMP Coupled Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("DMP Coupled Jerk (rad/$s^3$)")
    plt.xlim(time_v[0], time_v[-1])
    plt.tight_layout()
    plt.show()

    # create labels
    labels = [
        "Optimizer 1",
        "Optimizer 2",
        "Optimizer 3",
        "Optimizer 4",
        "Optimizer 5",
        "Optimizer 6",
        "pDMP",
        "pDMP Coupled",
        "pDMP Omega"
    ]
    # Calculate the jerk metrics
    j1, abs_j1, j_metrics1 = compute_jerk_metrics(jerks_1)
    j2, abs_j2, j_metrics2 = compute_jerk_metrics(jerks_2)
    j4, abs_j4, j_metrics4 = compute_jerk_metrics(jerks_4)
    j5, abs_j5, j_metrics5 = compute_jerk_metrics(jerks_5)
    j6, abs_j6, j_metrics6 = compute_jerk_metrics(jerks_6)
    j7, abs_j7, j_metrics7 = compute_jerk_metrics(jerks_7)
    jDMP, abs_jDMP, j_metricsDMP = compute_jerk_metrics(DMP_jerks)
    jDMP_Coupled, abs_jDMP_Coupled, j_metricsDMP_Coupled = compute_jerk_metrics(DMP_Coupled_jerks)

    # create vectors for the metrics
    means = [j_metrics1["mean"], j_metrics2["mean"], j_metrics4["mean"], j_metrics5["mean"], j_metrics6["mean"], j_metrics7["mean"], j_metricsDMP["mean"], j_metricsDMP_Coupled["mean"]]
    medians = [j_metrics1["median"], j_metrics2["median"], j_metrics4["median"], j_metrics5["median"], j_metrics6["median"], j_metrics7["median"], j_metricsDMP["median"], j_metricsDMP_Coupled["median"]]
    sigmas = [j_metrics1["sigma"], j_metrics2["sigma"], j_metrics4["sigma"], j_metrics5["sigma"], j_metrics6["sigma"], j_metrics7["sigma"], j_metricsDMP["sigma"], j_metricsDMP_Coupled["sigma"]]
    maxs = [j_metrics1["max"], j_metrics2["max"], j_metrics4["max"], j_metrics5["max"], j_metrics6["max"], j_metrics7["max"], j_metricsDMP["max"], j_metricsDMP_Coupled["max"]]
    q25s = [j_metrics1["q25"], j_metrics2["q25"], j_metrics4["q25"], j_metrics5["q25"], j_metrics6["q25"], j_metrics7["q25"], j_metricsDMP["q25"], j_metricsDMP_Coupled["q25"]]
    q75s = [j_metrics1["q75"], j_metrics2["q75"], j_metrics4["q75"], j_metrics5["q75"], j_metrics6["q75"], j_metrics7["q75"], j_metricsDMP["q75"], j_metricsDMP_Coupled["q75"]]
    lower_errors = [mean - q25 for mean, q25 in zip(means, q25s)]
    upper_errors = [q75 - mean for mean, q75 in zip(means, q75s)]
    lower_median_errors = [mean - median for mean, median in zip(means, medians)]
    upper_median_errors = [median - mean for mean, median in zip(means, medians)]
    lower_errors = np.maximum(lower_errors, 0)
    upper_errors = np.maximum(upper_errors, 0)
    lower_median_errors = np.maximum(lower_median_errors, 0)
    upper_median_errors = np.maximum(upper_median_errors, 0)

    # Print mean and median jerk for each optimizer
    print("Jerk Metrics for Each Optimizer:")
    for label, mean, median, sigma, max_val in zip(labels, means, medians, sigmas, maxs):
        print(f"{label}: Mean Jerk = {mean:.2e}, Median Jerk = {median:.2e}, Sigma = {sigma:.2e}, Max Jerk = {max_val:.2e}")

    # Create bar plots
    plt.figure(figsize=(7, 4))
    plt.bar(labels, means, yerr=[lower_median_errors, upper_median_errors], color='skyblue')
    plt.scatter(labels, maxs, color='red', label='Max Jerk')
    plt.ylabel('Mean Absolute Jerk (rad/s^3)')
    plt.yscale("symlog", linthresh=0.01)
    plt.ylim(bottom=0)
    plt.xticks(rotation=45)
    plt.xlabel("Optimizer")
    plt.ylabel("Mean Jerk (log scale)")
    # plt.title("Mean Jerk")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 4))
    plt.bar(labels, medians, yerr=[lower_median_errors, upper_median_errors], color='lightgreen')
    plt.scatter(labels, maxs, color='red', label='Max Jerk')
    plt.yscale("symlog", linthresh=0.01)
    plt.ylim(bottom=0)
    plt.xticks(rotation=45)
    plt.xlabel("Optimizer")
    plt.ylabel("Median Jerk (log scale)")
    # plt.title("Median Jerk for Different Optimizers")
    plt.legend()
    plt.tight_layout()
    plt.show()

    abs_jerk_data = [abs_j1, abs_j2, abs_j4, abs_j5, abs_j6, abs_j7, abs_jDMP, abs_jDMP_Coupled]

    #create box plots
    plt.figure(figsize=(7, 4))
    plt.boxplot(abs_jerk_data, labels=labels, showfliers=False)
    # plt.yscale('log')
    plt.yscale("symlog", linthresh=0.01)
    plt.xticks(rotation=45)
    plt.xlabel("Optimizer")
    plt.ylabel("Absolute Jerk (log scale)")
    # plt.title("Distribution of Absolute Jerk for Different Optimizers")
    plt.tight_layout()
    plt.show()

    # create violin plot
    plt.figure(figsize=(7, 4))
    violin = plt.violinplot(
        abs_jerk_data,
        showmeans=True,
        showmedians=True,
        showextrema=True
    )

    plt.yscale("symlog", linthresh=0.01)
    plt.xticks(
        ticks=np.arange(1, len(labels) + 1),
        labels=labels,
        rotation=45
    )
    plt.xlabel("Optimizer")
    plt.ylabel("Absolute Jerk (log scale)")
    # plt.title("Violin Plot of Absolute Jerk for Different Optimizers")
    plt.tight_layout()
    plt.show()

    print(f"best median jerk: {min(medians):.2e}, optimizer: {labels[medians.index(min(medians))]}")





    print("Starting IMU optimization test at 148 Hz...")
    FS = 148 # IMU
    # Generate test muscle activations (EMG signal) using sinewave between -1 and 1
    time_v = np.linspace(0, 20, FS*20)  # Time vector from 0 to 10 seconds
    activation = np.sin(2 * np.pi * 0.15 * time_v)  # Sine wave with frequency of 0.2 Hz

    # Small random noise
    rng = np.random.default_rng(seed=42)
    noise = rng.normal(0, 1, size=time_v.shape)

    # Smooth it with a moving average
    window_size = 30  # increase for smoother wobble
    kernel = np.ones(window_size) / window_size
    smooth_noise = np.convolve(noise, kernel, mode="same")

    # Scale the noise so it only creates a small wobble
    # noise_amplitude = 0.03
    noise_amplitude = 0.06

    activation += noise_amplitude * smooth_noise

    activation = np.clip(activation, -1, 1)

    delay = 0.08  # 80 ms delay (typical electromechanical delay)
    q_true = np.sin(2 * np.pi * 0.15 * (time_v-delay))
    omega = np.gradient(q_true, t)
    imu_q = q_true + 0.01 * np.random.randn(len(q_true))  # noisy angle

    plt.plot(time_v, activation, label="Activation")
    plt.plot(time_v, imu_q, label="True Angle")
    plt.legend()
    plt.show()

    # Create empty lists to store optimized angles for each optimizer
    Method9Angles = []
    Method10Angles = []

    v = 0
    Method9Angles.append(q)
    for a, da, w, imu in zip(activation, activation_diff, omega, imu_q):
        q_next, v, acc = Method9(
            a, da, v, w,
            kn=2, kd=2, kp=2, b=3,
            q=Method9Angles[-1],
            imu_q=imu,
            theta_min=THETA_MIN,
            theta_max=THETA_MAX,
            v_max=np.pi,
            t=t
        )
        Method9Angles.append(q_next)

    Method10Angles.append(q)
    for a, da, w, imu in zip(activation, activation_diff, omega, imu_q):
        q_next, v = Method10(
            a, da, w,
            kn=1.2, kd=1.2,
            imu_q=Method10Angles[-1],
            theta_min=THETA_MIN,
            theta_max=THETA_MAX,
            v_max=np.pi,
            t=t
        )
        Method10Angles.append(q_next)

    # Remove the initial angle from the optimized angles lists
    Method9Angles.remove(Method9Angles[0])
    Method10Angles.remove(Method10Angles[0])

    Method9Traj.extend(Method9Angles)
    Method10Traj.extend(Method10Angles)
    

    # Calculate the velocity, acceleration and jerk for each optimizer
    velocities_8 = np.gradient(Method9Angles, t)
    accelerations_8 = np.gradient(velocities_8, t)
    jerks_8 = np.gradient(accelerations_8, t)

    velocities_9 = np.gradient(Method10Angles, t)
    accelerations_9 = np.gradient(velocities_9, t)
    jerks_9 = np.gradient(accelerations_9, t)

    # Plot each optimized angle in different graphs comparing them to the input signal and with the position, velocity, acceleration and jerk.
    plt.figure(figsize=(12, 10))
    plt.title("Optimizer 1: IMU")
    plt.subplot(5, 1, 1)
    plt.plot(time_v, activation, label="Activation")
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.xlim(time_v[0], time_v[-1])

    plt.figure(figsize=(12, 10))
    plt.title("Optimizer 8: IMU with PD control and acceleration term")
    plt.subplot(5, 1, 1)
    plt.plot(time_v, activation, label="Activation")
    plt.plot(time_v, imu_q, label="IMU Angle", color='green')
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 2)
    plt.plot(time_v, Method9Angles, label="Optimized Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Optimized Angle (rad)")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 3)
    plt.plot(time_v, velocities_8, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 4)
    plt.plot(time_v, accelerations_8, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (rad/$s^2$)")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 5)
    plt.plot(time_v, jerks_8, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (rad/$s^3$)")
    plt.xlim(time_v[0], time_v[-1])
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 10))
    plt.title("Optimizer 9: IMU with PD control and no acceleration term")
    plt.subplot(5, 1, 1)
    plt.plot(time_v, activation, label="Activation")
    # plt.plot(time, omega, label="Angular Velocity", color='orange')
    plt.plot(time_v, imu_q, label="IMU Angle", color='green')
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Activation")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 2)
    plt.plot(time_v, Method10Angles, label="Optimized Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("Optimized Angle (rad)")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 3)
    plt.plot(time_v, velocities_9, label="Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (rad/s)")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 4)
    plt.plot(time_v, accelerations_9, label="Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (rad/$s^2$)")
    plt.xlim(time_v[0], time_v[-1])
    plt.subplot(5, 1, 5)
    plt.plot(time_v, jerks_9, label="Jerk")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (rad/$s^3$)")
    plt.xlim(time_v[0], time_v[-1])
    plt.tight_layout()
    plt.show()