from arc import *
from scipy.constants import speed_of_light, hbar, e
from scipy.constants import physical_constants
bohr_radius = physical_constants['Bohr radius'][0]
bohr_magneton = physical_constants['Bohr magneton in Hz/T'][0]
import json
import numpy as np
from scipy.integrate import solve_ivp
import os
import sys
from Code import SinglePhotonSim, SinglePhotonSim2AC, SinglePhoton, RydbergLeakage, find_FS_state
import matplotlib.pyplot as plt
def l_t(L,w,t):
    return np.exp(1j*w*t)*L

cs = Cesium()

# Capture command-line arguments
if __name__ == "__main__":
    arg = eval('[' + sys.argv[1] + ']')
    folder = 'results'  # f'2AC-delta{detuning/1e6}-energy{energy_space}dl{dl}'
    os.makedirs(folder, exist_ok=True)

    # ### atomic property #########
    # Rydberg state
    n_r = 60  # int(arg[0]) #
    l_r = 1  # int(arg[1])
    j_r = 3 / 2  # float(arg[2])
    mj_r = 3 / 2  # float(arg[3])

    # Magnetic field
    Bz = 1  # float(arg[4])  # G

    # Nearby Rydberg state
    energy_space = 15 # float(arg[1])  # GHz
    dl = 1  # int(arg[2])
    # atomic_states = find_FS_state(n_r, l_r, j_r, energy_space, dl, folder)
    atomic_states = [
        [n_r, l_r, j_r, mj_r],
        [n_r, l_r, j_r, 1 / 2],
        [n_r+1, 0, 1 / 2, 1 / 2]]
    # ### atomic property #########

    q_dc = {-1: 0.0, 0: 1.0, 1: 0.0}

    # ########### property of E1 shiftout###############
    # Transition parameters
    n_1 = n_r  # int(arg[8])
    l_1 = l_r  # int(arg[9])
    j_1 = j_r  # float(arg[10])
    mj_1 = 1 / 2  # float(arg[11])

    n_2 = n_r + 1  # int(arg[12])
    l_2 = 0  # int(arg[13])
    j_2 = 1 / 2  # float(arg[14])
    mj_2 = 1 / 2  # float(arg[15])
    detuning = -20e6  # float(arg[0])

    # Calculate first laser frequency accounting for Zeeman shift and transition frequency
    Zeeman_shift_ryd = cs.getZeemanEnergyShift(l=l_1, j=j_1, mj=mj_1, magneticFieldBz=Bz / 10000, s=0.5) / (
            hbar * 2 * np.pi)
    fine_frequency = cs.getTransitionFrequency(n1=n_1, l1=l_1, j1=j_1, n2=n_2, l2=l_2, j2=j_2, s=0.5)
    Zeeman_shift = cs.getZeemanEnergyShift(l=l_2, j=j_2, mj=mj_2, magneticFieldBz=Bz / 10000, s=0.5) / (
            hbar * 2 * np.pi)
    w_ac_r = fine_frequency + Zeeman_shift - Zeeman_shift_ryd
    w_ac_1 = w_ac_r + detuning

    # Polarization components of E1
    sigma_min = 0  # float(arg[5])
    pi = 1  # float(arg[6])
    sigma_plus = 0  # float(arg[7])
    pol_total = np.sqrt(sigma_plus ** 2 + pi ** 2 + sigma_min ** 2)
    q_ac_1 = {
        -1: np.sqrt(sigma_min / pol_total),
        0: np.sqrt(pi / pol_total),
        1: np.sqrt(sigma_plus / pol_total)
    }

    # Fourier component
    delta_ms = 2  # int(arg[3])
    ms1 = list(range(-delta_ms, delta_ms + 1))

    # Field strength scan parameters
    # v_ac_min = 0  # float(arg[5])
    # v_ac_max = 100  # float(arg[6])
    # v_ac_point = 40  # int(arg[7])
    # E_ac_is = np.linspace(v_ac_min, v_ac_max, v_ac_point)
    v_ac1 = float(arg[0])
    # ########### property of E1 shiftout###############

    # ########### property of E polarizability  ###############
    # Transition parameters
    n_1 = n_r  # int(arg[8])
    l_1 = l_r  # int(arg[9])
    j_1 = j_r  # float(arg[10])
    mj_1 = mj_r  # float(arg[11])

    n_2 = n_r - 1  # int(arg[12])
    l_2 = 2  # int(arg[13])
    j_2 = 5 / 2  # float(arg[14])
    mj_2 = mj_r  # float(arg[15])
    detuning2 = -20e6  # float(arg[1])

    # Calculate laser frequency accounting for Zeeman shift and transition frequency
    Zeeman_shift_ryd = cs.getZeemanEnergyShift(l=l_1, j=j_1, mj=mj_1, magneticFieldBz=Bz / 10000, s=0.5) / (
            hbar * 2 * np.pi)
    fine_frequency = cs.getTransitionFrequency(n1=n_1, l1=l_1, j1=j_1, n2=n_2, l2=l_2, j2=j_2, s=0.5)
    Zeeman_shift = cs.getZeemanEnergyShift(l=l_2, j=j_2, mj=mj_2, magneticFieldBz=Bz / 10000, s=0.5) / (
            hbar * 2 * np.pi)
    w_ac_r2 = fine_frequency + Zeeman_shift - Zeeman_shift_ryd
    w_ac_2 = w_ac_r2 + detuning2

    # Polarization components
    sigma_min = 0  # float(arg[5])
    pi = 1  # float(arg[6])
    sigma_plus = 0  # float(arg[7])
    pol_total = np.sqrt(sigma_plus ** 2 + pi ** 2 + sigma_min ** 2)
    q_ac_2 = {
        -1: np.sqrt(sigma_min / pol_total),
        0: np.sqrt(pi / pol_total),
        1: np.sqrt(sigma_plus / pol_total)
    }

    # Fourier component
    delta_ms2 = 2  # int(arg[4])
    ms2 = list(range(-delta_ms, delta_ms + 1))

    # Field strength scan parameters

    # E_ac_is2 = np.linspace(v_ac_min2, v_ac_max2, v_ac_point2)
    v_ac2 = float(arg[1])
    if v_ac2 == 0:
        v_ac_min2 = 0  # float(arg[8])
        v_ac_max2 = 25  # float(arg[9])
        v_ac_point2 = 26  # int(arg[10])
    elif v_ac2 == 1:
        v_ac_min2 = 26  # float(arg[8])
        v_ac_max2 = 50  # float(arg[9])
        v_ac_point2 = 26  # int(arg[10])
    else:
        raise ValueError(f'v_ac2 is neither 1 or 2 -> {v_ac2}')
    E_ac_is2 = np.linspace(v_ac_min2, v_ac_max2, v_ac_point2)
    # ########### property of E2 ###############

    rabi_rydberg = 1e6
    # Create settings dictionary
    setting_dict = {
        'n_r': n_r,
        'l_r': l_r,
        'j_r': j_r,
        'mj_r': mj_r,
        'Bz': Bz,
        'polarization': {
            'sigma_min': sigma_min,
            'pi': pi,
            'sigma_plus': sigma_plus,
            'q_ac_1': q_ac_1,
            'q_dc': q_dc
        },
        'transition_E1': {
            'n_1': n_1,
            'l_1': l_1,
            'j_1': j_1,
            'mj_1': mj_1,
            'n_2': n_2,
            'l_2': l_2,
            'j_2': j_2,
            'mj_2': mj_2,
            'detuning': detuning,
            'w_ac_0': w_ac_r,
            'w_ac': w_ac_1
        },
        'transition_E2': {
            'detuning': detuning2,
            'w_ac_0': w_ac_r2,
            'w_ac': w_ac_2
        },
        'nearby_rydberg': {
            'energy_space': energy_space,
            'dl': dl,
            'atomic_states': atomic_states
        },
        'SF_Hamiltonian': {
            'delta_ms': delta_ms,
            'ms': ms1,
            'ms2': ms2
        },
        'ac_voltage': {
            'v_ac1': v_ac1,
            'v_ac2': v_ac2
            # 'v_ac_min': v_ac_min,
            # 'v_ac_max': v_ac_max,
            # 'v_ac_point': v_ac_point,
            # 'v_ac_min2': v_ac_min2,
            # 'v_ac_max2': v_ac_max2,
            # 'v_ac_point2': v_ac_point2
        },
        'Rydberg_rabi': rabi_rydberg
    }

    # Dump settings to a JSON file
    with open(f'{folder}/settings.json', 'w') as json_file:
        json.dump(setting_dict, json_file, indent=4)

    ground_state = [
        [[n_r, l_r, j_r, 1 / 2], rabi_rydberg / np.sqrt(3)],
        [[n_r, l_r, j_r, 3 / 2], rabi_rydberg],
    ]

    sim = RydbergLeakage(atom=cs, atom_FS_states=atomic_states, ground_state=ground_state, delay=1e-5,
                              expand_zeeman=False)
    comp_atomic_states = sim.atom_states
    # sim.atom_states = [comp_atomic_states[1], comp_atomic_states[4]]
    total_state = sim.comp_atomic_states_num
    #### Generate H0
    H0 = sim.generate_H0(Bz)
    print('H0:', np.around(np.diag(H0) / 1e6, 2).astype(int))
    result_list = []
    d, H_g = sim.generate_Edipole_matrix(q_ac_1)
    L = -d
    L_dag = np.conjugate(L.T)
    H = H0 + H_g
    for row, E_ac_i1 in enumerate([v_ac1]):
        for col, E_ac_i2 in enumerate([50]):
        # for col, E_ac_i2 in enumerate(E_ac_is2):
            # Drive Hamiltonian
            def linbald(t, y):
                psi = y[:total_state * total_state].reshape(total_state, total_state) + 1j * y[total_state ** 2:].reshape(
                    total_state, total_state)
                evo = -1j * (H @ psi - psi @ H)
                evo += E_ac_i1 * (l_t(L, w_ac_1, t) @ psi @ l_t(L_dag, -w_ac_1, t) -
                                  1 / 2 * (l_t(L_dag, -w_ac_1, t) @ l_t(L, w_ac_1, t) @ psi +
                                           psi @ l_t(L_dag, -w_ac_1, t) @ l_t(L, w_ac_1,t)))
                evo += E_ac_i2 * (l_t(L, w_ac_2, t) @ psi @ l_t(L_dag, -w_ac_2, t) -
                                  1 / 2 * (l_t(L_dag, -w_ac_2, t) @ l_t(L, w_ac_2, t) @ psi +
                                           psi @ l_t(L_dag, -w_ac_2, t) @ l_t(L, w_ac_2,t)))
                return np.concatenate((evo.real.flatten(), evo.imag.flatten()))

            # Initial state |0>
            psi0 = np.zeros((total_state, total_state), dtype=complex)
            psi0[0, 0] = 1
            y0 = np.concatenate((psi0.real.flatten(), psi0.imag.flatten()))
            t_span = (0, 10e-6)
            t_eval = np.linspace(*t_span, 100)
            sol = solve_ivp(linbald, t_span, y0, t_eval=t_eval, method='RK45', rtol=1e-11, atol=1e-12)
            # print(sol.y.shape)
            # # Probabilities
            # popu = sol.y[:total_state, :]**2 + sol.y[total_state:, :]**2  # |ψ0|²

            # Extract state amplitudes
            c = sol.y
            steps = c.shape[-1]
            cg = c[:total_state * total_state, :].reshape(total_state, total_state, steps) + 1j * c[total_state ** 2:,
                                                                                                  :].reshape(total_state,
                                                                                                             total_state,
                                                                                                             steps)

            Pg = cg[0, 0]
            Pe = cg[1, 1]
            Pr = cg[2, 2]
            Prr = cg[3,3]
            # Plot populations
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(sol.t, Pg, label=r'$g$', lw=2)
            ax.plot(sol.t, Pe, label=r'$r$', lw=2)
            ax.plot(sol.t, Pr, label=r'$r_{1/2}$', lw=2)
            ax.plot(sol.t, Prr, label=r'$rs$', lw=2)
            ax.set_xlabel('Time')
            ax.set_ylabel('Population')
            ax.set_title('Three-Level Raman System (Λ Configuration, RWA)')
            ax.legend()
            # plt.grid(True)
            fig.tight_layout()
            # fig.show()
            fig.savefig('leakage.png')
            with open(f"{folder}/result.txt", "a") as f:
                f.write(f'{E_ac_i1},{E_ac_i2},{cg}\n')

