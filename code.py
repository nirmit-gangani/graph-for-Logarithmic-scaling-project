import numpy as np
import matplotlib.pyplot as plt

g = 9.81
L1 = L2 = 1.0
m1 = m2 = 1.0

epsilon = 0.1
T = 20

dt_values = [0.02, 0.015, 0.01, 0.007, 0.005, 0.003, 0.002]
agreement_times = []

theta1_0 = np.pi / 2
theta2_0 = np.pi / 2
omega1_0 = 0
omega2_0 = 0

def derivatives(state):
    theta1, omega1, theta2, omega2 = state
    
    delta = theta2 - theta1
    
    denom1 = (m1 + m2)*L1 - m2*L1*np.cos(delta)**2
    denom2 = (L2/L1)*denom1

    domega1 = (m2*L1*omega1**2*np.sin(delta)*np.cos(delta) +
               m2*g*np.sin(theta2)*np.cos(delta) +
               m2*L2*omega2**2*np.sin(delta) -
               (m1+m2)*g*np.sin(theta1)) / denom1

    domega2 = (-m2*L2*omega2**2*np.sin(delta)*np.cos(delta) +
               (m1+m2)*(g*np.sin(theta1)*np.cos(delta) -
               L1*omega1**2*np.sin(delta) -
               g*np.sin(theta2))) / denom2

    return np.array([omega1, domega1, omega2, domega2])

def rk4_step(state, dt):
    k1 = derivatives(state)
    k2 = derivatives(state + 0.5*dt*k1)
    k3 = derivatives(state + 0.5*dt*k2)
    k4 = derivatives(state + dt*k3)
    return state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

def verlet_step(state, dt):
    theta1, omega1, theta2, omega2 = state
    deriv = derivatives(state)

    omega1_half = omega1 + 0.5 * dt * deriv[1]
    omega2_half = omega2 + 0.5 * dt * deriv[3]

    theta1_new = theta1 + dt * omega1_half
    theta2_new = theta2 + dt * omega2_half

    new_state = np.array([theta1_new, omega1_half, theta2_new, omega2_half])
    deriv_new = derivatives(new_state)

    omega1_new = omega1_half + 0.5 * dt * deriv_new[1]
    omega2_new = omega2_half + 0.5 * dt * deriv_new[3]

    return np.array([theta1_new, omega1_new, theta2_new, omega2_new])

for dt in dt_values:
    steps = int(T / dt)
    time = np.linspace(0, T, steps)

    state_rk4 = np.array([theta1_0, omega1_0, theta2_0, omega2_0])
    state_verlet = state_rk4.copy()

    traj_rk4 = []
    traj_verlet = []

    Ta = None

    for i in range(steps):
        traj_rk4.append(state_rk4.copy())
        traj_verlet.append(state_verlet.copy())

        diff = np.linalg.norm(state_rk4 - state_verlet)

        if Ta is None and diff > epsilon:
            Ta = i * dt

        state_rk4 = rk4_step(state_rk4, dt)
        state_verlet = verlet_step(state_verlet, dt)

    if Ta is None:
        Ta = T

    agreement_times.append(Ta)

    # SAVE ONE TRAJECTORY FOR GRAPH 1 (choose dt = 0.005)
    if dt == 0.005:
        traj_rk4_plot = np.array(traj_rk4)
        traj_verlet_plot = np.array(traj_verlet)
        time_plot = time

separation = np.linalg.norm(traj_rk4_plot - traj_verlet_plot, axis=1)
separation = np.where(separation < 1e-10, 1e-10, separation)
log_sep = np.log(separation)

plt.figure()
plt.plot(time_plot, log_sep)
plt.xlabel("Time (s)")
plt.ylabel("log Separation")
plt.title("Figure 1: Log Separation vs Time")
plt.grid()
plt.savefig("figure1_log_separation.png", dpi=300)
plt.show()

plt.figure()
plt.plot(dt_values, agreement_times, marker='o')
plt.xlabel("Timestep (dt)")
plt.ylabel("Agreement Time (s)")
plt.title("Figure 2: Agreement Time vs Timestep")
plt.grid()
plt.savefig("figure2_Ta_vs_dt.png", dpi=300)
plt.show()

log_dt = np.log(1/np.array(dt_values))

coeffs = np.polyfit(log_dt, agreement_times, 1)
a, b = coeffs

x_fit = np.linspace(min(log_dt), max(log_dt), 100)
y_fit = a * x_fit + b

plt.figure()
plt.plot(log_dt, agreement_times, 'o', label="Data")
plt.plot(x_fit, y_fit, '-', label=f"Fit: Ta = {a:.3f} ln(1/dt) + {b:.3f}")
plt.xlabel("ln(1/dt)")
plt.ylabel("Agreement Time (s)")
plt.title("Figure 3: Logarithmic Scaling")
plt.legend()
plt.grid()
plt.savefig("figure3_scaling.png", dpi=300)
plt.show()
