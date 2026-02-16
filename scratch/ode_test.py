import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def carbonate_kinetics(t, y, kf, kb, k_out, C_eq):
    """
    y = [C, H, B]
    C = [CO2(aq)]  (mol/L)
    H = [H+]       (mol/L)
    B = [HCO3-]    (mol/L)

    Reaction: CO2 <-> H+ + HCO3-
      forward  rate = kf * C
      reverse  rate = kb * H * B

    Slow physical process: CO2 outgassing / re-equilibration with air:
      dC/dt includes -k_out*(C - C_eq)
    """
    C, H, B = y
    r_f = kf * C
    r_b = kb * H * B
    r_net = r_f - r_b

    dC = -r_net - k_out * (C - C_eq)
    dH = +r_net
    dB = +r_net
    return [dC, dH, dB]

# -------------------------
# Parameters (chosen to be intentionally stiff)
# -------------------------
kf = 1e6        # 1/s  (very fast forward)
kb = 1e12       # 1/(M*s) (very fast reverse; makes equilibrium extremely fast)
k_out = 1e-3    # 1/s  (slow CO2 exchange with air)
C_eq = 1e-5     # mol/L equilibrium CO2 with air (illustrative)

# Initial conditions (acidified water slug with dissolved CO2)
C0 = 1e-3       # mol/L CO2(aq)
H0 = 1e-7       # mol/L (pH 7)
B0 = 1e-12      # tiny bicarbonate seed to avoid exact zero product
y0 = [C0, H0, B0]

t_span = (0.0, 2e4)  # seconds (~5.6 hours)
t_eval = np.linspace(t_span[0], t_span[1], 2000)

# -------------------------
# 1) Try RK45 (explicit) with tight tolerances -> likely failure
# -------------------------
sol_rk = solve_ivp(
    fun=lambda t, y: carbonate_kinetics(t, y, kf, kb, k_out, C_eq),
    t_span=t_span,
    y0=y0,
    method="RK45",
    t_eval=t_eval,
    rtol=1e-12,
    atol=1e-18
)

print("RK45 success:", sol_rk.success)
print("RK45 message:", sol_rk.message)

# -------------------------
# 2) Fix A: use an implicit stiff solver (Radau or BDF)
# -------------------------
sol_radau = solve_ivp(
    fun=lambda t, y: carbonate_kinetics(t, y, kf, kb, k_out, C_eq),
    t_span=t_span,
    y0=y0,
    method="Radau",
    t_eval=t_eval,
    rtol=1e-9,
    atol=1e-15
)

print("\nRadau success:", sol_radau.success)
print("Radau message:", sol_radau.message)

# -------------------------
# Plot only if Radau succeeded
# -------------------------
if sol_radau.success:
    C, H, B = sol_radau.y
    pH = -np.log10(H)

    plt.figure()
    plt.plot(sol_radau.t, C, label="[CO2(aq)]")
    plt.plot(sol_radau.t, B, label="[HCO3-]")
    plt.xlabel("Time (s)")
    plt.ylabel("Concentration (mol/L)")
    plt.legend()
    plt.title("Stiff carbonate kinetics solved with Radau")
    plt.show()

    plt.figure()
    plt.plot(sol_radau.t, pH, label="pH")
    plt.xlabel("Time (s)")
    plt.ylabel("pH")
    plt.legend()
    plt.title("pH evolution")
    plt.show()

# -------------------------
# 3) Fix B (modeling fix): assume fast equilibrium instead of explicit kinetics
# -------------------------
# For CO2 <-> H+ + HCO3-, equilibrium constant K = kf/kb (for this toy kinetic model)
K = kf / kb

def reduced_model_equilibrium(t, y, k_out, C_eq, K):
    """
    Reduced model:
    - Treat acid-base as instantaneous equilibrium: H*B = K*C
    - Keep slow CO2 exchange: dC/dt = -k_out*(C - C_eq)
    - Conserve "acid equivalents": A = H - B is constant for this reaction stoichiometry
      (because both H and B rise/fall together via r_net).
    Then solve algebraically for H and B each time from:
      H*B = K*C
      H - B = A_const
    """
    C, = y
    dC = -k_out * (C - C_eq)
    return [dC]

# Compute A_const from initial conditions (invariant under H,B changing equally)
A_const = H0 - B0

def solve_H_B_from_equilibrium(C, K, A):
    # Solve: H*(H - A) = K*C  (since B = H - A)
    # => H^2 - A*H - K*C = 0
    disc = A*A + 4*K*C
    H = 0.5*(A + np.sqrt(disc))
    B = H - A
    return H, B

sol_red = solve_ivp(
    fun=lambda t, y: reduced_model_equilibrium(t, y, k_out, C_eq, K),
    t_span=t_span,
    y0=[C0],
    t_eval=t_eval,
    method="RK45"  # now fine, because we removed the stiff fast kinetics
)

print("\nReduced equilibrium model success:", sol_red.success)

if sol_red.success:
    C_red = sol_red.y[0]
    H_red, B_red = solve_H_B_from_equilibrium(C_red, K, A_const)
    pH_red = -np.log10(H_red)

    plt.figure()
    plt.plot(sol_red.t, C_red, label="[CO2(aq)] reduced")
    plt.plot(sol_red.t, B_red, label="[HCO3-] reduced")
    plt.xlabel("Time (s)")
    plt.ylabel("Concentration (mol/L)")
    plt.legend()
    plt.title("Reduced (equilibrium) model")
    plt.show()

    plt.figure()
    plt.plot(sol_red.t, pH_red, label="pH reduced")
    plt.xlabel("Time (s)")
    plt.ylabel("pH")
    plt.legend()
    plt.title("pH (reduced model)")
    plt.show()
