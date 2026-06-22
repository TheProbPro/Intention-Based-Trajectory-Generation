import numpy as np

MARGIN_RATIO = 0.2

def _boundary_scaling(q, activation, theta_min, theta_max, margin_ratio=0.1):
    span = theta_max - theta_min
    margin = margin_ratio * span

    if activation > 0:
        # moving toward theta_max
        dist = theta_max - q
    elif activation < 0:
        # moving toward theta_min
        dist = q - theta_min
    else:
        return 1.0

    if dist <= 0:
        return 0.0
    elif dist >= margin:
        return 1.0
    else:
        x = dist / margin
        # return x * x * (3 - 2 * x)  # cubic smoothstep
        return x**3 * (10 - 15*x + 6*x**2) # quintic smoothstep

# ==============================================================================================================================

def Method3(k, activation, t, q, theta_min, theta_max):
    """
    Generates trajectory based on EMG signal
    Parameters:
    k: Proportional gain
    activation: Muscle activation level [-1,1]
    t: Time between updates [s]
    q: Current angle
    theta_min: Lower ROM limit
    theta_max: Upper ROM limit
    Returns:
    q_next: The next position in the trajectory
    """
    scale = _boundary_scaling(q, activation, theta_min, theta_max, margin_ratio=MARGIN_RATIO)

    delta_q = k * activation * t * scale
    q_next = q + delta_q
    
    return q_next

def Method4(k, activation, t, q, theta_min, theta_max):
    """
    Generates trajectory based on EMG signal
    Parameters:
    k: Proportional gain
    activation: Muscle activation level [-1,1]
    t: Time between updates [s]
    q: Current angle
    theta_min: Lower ROM limit
    theta_max: Upper ROM limit
    Returns:
    q_next: The next position in the trajectory
    """
    w = 0
    if activation > 0:
        w = (theta_max - q) / theta_max
    elif activation < 0:
        w = q / theta_max

    delta_q = k * activation * t * w
    q_next = q + delta_q
    
    return q_next

def Method5(k, activation, t, q, delta_q_prev, theta_min, theta_max, alpha=0.5):
    """
    Generates trajectory based on EMG signal with momentum
    Parameters:
    k: Proportional gain
    activation: Muscle activation level [-1,1]
    t: Time between updates [s]
    q: Current angle
    delta_q_prev: Previous change in angle
    theta_min: Lower ROM limit
    theta_max: Upper ROM limit
    alpha: Momentum factor
    Returns:
    q_next: The next position in the trajectory
    delta_q: The change in angle for this update
    """
    scale = _boundary_scaling(q, activation, theta_min, theta_max, margin_ratio=MARGIN_RATIO)
    
    delta_q_raw = k * activation * t * scale
    delta_q = alpha * delta_q_raw + (1-alpha) * delta_q_prev
    
    q_next = q + delta_q

    return q_next, delta_q

def Method6(activation, velocity, t, q, theta_min, theta_max, v_max, k, b=0.5,):
    """
    Generates trajectory based on EMG signal with velocity control
    Parameters:
    activation: Muscle activation level [-1,1]
    velocity: Current velocity
    t: Time between updates [s]
    q: Current angle
    theta_min: Lower ROM limit
    theta_max: Upper ROM limit
    v_max: Maximum velocity
    k: Proportional gain
    b: Velocity feedback gain
    Returns:
    q_next: The next position in the trajectory
    velocity: The updated velocity
    """
    scale = _boundary_scaling(q, activation, theta_min, theta_max, margin_ratio=MARGIN_RATIO)
    velocity = b * velocity + k * activation * scale
    velocity = np.clip(velocity, -v_max, v_max)
    
    q_next = q + velocity * t

    return q_next, velocity

def Method7(activation, velocity, t, q, theta_min, theta_max, v_max=np.pi, b = 6.0, k = None):
    """
    Generates trajectory based on EMG signal with acceleration control
    Parameters:
    activation: Muscle activation level [-1,1]
    velocity: Current velocity
    t: Time between updates [s]
    q: Current angle
    theta_min: Lower ROM limit
    theta_max: Upper ROM limit
    v_max: Maximum velocity
    b: Damping coefficient
    k: Proportional gain (if None, it will be set to b * pi)
    Returns:
    q_next: The next position in the trajectory
    velocity: The updated velocity
    acc: The updated acceleration
    """
    # Smoothen acceleration
    k = b * np.pi if k is None else k
    scale = _boundary_scaling(q, activation, theta_min, theta_max, margin_ratio=MARGIN_RATIO)
    acc = k * activation * scale - b * velocity

    # Update velocity and position
    velocity += acc * t
    velocity = np.clip(velocity, -v_max, v_max)
    q_next = q + velocity * t * scale

    return q_next, velocity, acc

def Method8(a, d_a, v, kn, kd, b, q, THETA_MIN, THETA_MAX, v_max, t):
   """
   Generates trajectory based on EMG signal with acceleration and velocity control
    Parameters:
    a: Muscle activation level [-1,1]
    d_a: derivative of muscle activation level
    v: Current velocity
    kn: Proportional gain for activation
    kd: Proportional gain for activation derivative
    b: Damping coefficient
    q: Current angle
    THETA_MIN: Lower ROM limit
    THETA_MAX: Upper ROM limit
    v_max: Maximum velocity
    t: Time between updates [s]
    Returns:
    q_next: The next position in the trajectory
    v: The updated velocity
    acc: The updated acceleration
   """
   scale = _boundary_scaling(q, a, THETA_MIN, THETA_MAX, margin_ratio=MARGIN_RATIO)
   
   # Calculate desired acceleration
   acc = (kn * a + kd * d_a - b * v) * scale

   # Update velocity and position
   v += t * acc
   v = np.clip(v, -v_max, v_max)
   q_next = q + t * v * scale

   return q_next, v, acc

def Method9(a, d_a, v, omega, kn, kd, kp, b, q, imu_q, theta_min, theta_max, v_max, t):
    """
    Generates trajectory based on EMG signal with acceleration, velocity, and position control and IMU feedback
    Parameters:
    a: Muscle activation level [-1,1]
    d_a: derivative of muscle activation level
    v: Current velocity
    omega: Desired velocity based on IMU feedback
    kn: Proportional gain for activation
    kd: Proportional gain for activation derivative
    kp: Proportional gain for position error
    b: Damping coefficient
    q: Current angle
    imu_q: Current angle from IMU
    theta_min: Lower ROM limit
    theta_max: Upper ROM limit
    v_max: Maximum velocity
    t: Time between updates [s]
    Returns:
    q_next: The next position in the trajectory
    v: The updated velocity
    acc: The updated acceleration
    """
    scale = _boundary_scaling(q, a, theta_min, theta_max, margin_ratio=MARGIN_RATIO)

    # Calculate desired acceleration
    acc = (kn * a + kd * d_a - b * (v - omega) - kp * (q - imu_q)) * scale

    # Update velocity and position
    v += t * acc
    v = np.clip(v, -v_max, v_max)
    q_next = q + t * v * scale

    return q_next, v, acc

def Method10(a, d_a, omega, kn, kd, imu_q, theta_min, theta_max, v_max, t):
    """
    Generates trajectory based on EMG signal with velocity control and IMU feedback
    Parameters:
    a: Muscle activation level [-1,1]
    d_a: derivative of muscle activation level
    omega: Desired velocity based on IMU feedback
    kn: Proportional gain for activation
    kd: Proportional gain for activation derivative
    imu_q: Current angle from IMU
    theta_min: Lower ROM limit
    theta_max: Upper ROM limit
    v_max: Maximum velocity
    t: Time between updates [s]
    Returns:
    q_next: The next position in the trajectory
    v: The updated velocity
    """
    # Calculate desired velocity
    v = omega + kn * a + kd * d_a
    v = np.clip(v, -v_max, v_max)

    # Update position
    scale = _boundary_scaling(imu_q, a, theta_min, theta_max, margin_ratio=MARGIN_RATIO)
    q_next = imu_q + t * v * scale

    return q_next, v