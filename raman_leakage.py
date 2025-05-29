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
from Code import SinglePhotonSim, SinglePhotonSim2AC, SinglePhoton, RydbergSinglePhoton, find_FS_state


# Time-dependent Hamiltonian
def H_t(t):
    return H0 + H_prime * np.cos(w_ac * t) + H_g
# Schrödinger equation in real + imag form
def schrodinger(t, y):
    psi = y[:total_state] + 1j * y[total_state:]
    dpsi_dt = -1j * H_t(t) @ psi
    return np.concatenate((dpsi_dt.real, dpsi_dt.imag))


# Capture command-line arguments
if __name__ == "__main__":
    # Cesium Atom object in Arc Akali
    cs = Cesium()
    arg = eval('['+sys.argv[1]+']')
    os.makedirs('results', exist_ok=True)

    # Rydberg state
    n_r = int(arg[0])
    l_r = int(arg[1])
    j_r = float(arg[2])
    mj_r = float(arg[3])

    # Magnetic field
    Bz = float(arg[4])  # G

    # Polarization components
    sigma_min = float(arg[5])
    pi = float(arg[6])
    sigma_plus = float(arg[7])

    pol_total = np.sqrt(sigma_plus ** 2 + pi ** 2 + sigma_min ** 2)
    q_ac = {
        -1: np.sqrt(sigma_min / pol_total),
        0: np.sqrt(pi / pol_total),
        1: np.sqrt(sigma_plus / pol_total)
    }

    # Transition parameters
    n_1 = int(arg[8])
    l_1 = int(arg[9])
    j_1 = float(arg[10])
    mj_1 = float(arg[11])

    n_2 = int(arg[12])
    l_2 = int(arg[13])
    j_2 = float(arg[14])
    mj_2 = float(arg[15])

    detuning = float(arg[16])

    # Placeholder functions for Zeeman shift and transition frequency
    # Replace with actual function calls as needed
    Zeeman_shift_ryd = cs.getZeemanEnergyShift(l=l_1, j=j_1, mj=mj_1, magneticFieldBz=Bz / 10000, s=0.5) / (
            hbar * 2 * np.pi)
    fine_frequency = cs.getTransitionFrequency(n1=n_1, l1=l_1, j1=j_1, n2=n_2, l2=l_2, j2=j_2, s=0.5)
    Zeeman_shift = cs.getZeemanEnergyShift(l=l_2, j=j_2, mj=mj_2, magneticFieldBz=Bz / 10000, s=0.5) / (
            hbar * 2 * np.pi)

    w_ac_0 = fine_frequency + Zeeman_shift - Zeeman_shift_ryd
    w_ac = w_ac_0 + detuning

    # Nearby Rydberg state
    energy_space = float(arg[17])  # GHz
    dl = int(arg[18])
    atomic_states = find_FS_state(n_r, l_r, j_r, energy_space, dl, 'results')

    delta_ms = int(arg[19])
    ms = list(range(-delta_ms, delta_ms + 1))

    v_ac_min = float(arg[20])
    v_ac_max = float(arg[21])
    v_ac_point = int(arg[22])

    rabi_rydberg = float(arg[23])

    ground_state = [
        [[6, 0, 1/2, -1/2], [n_r, l_r, j_r, 1/2], rabi_rydberg],
        [[6, 0, 1/2, 1/2], [n_r, l_r, j_r, 3/2], rabi_rydberg],
         ]

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
            'q_ac': q_ac
        },
        'transition': {
            'n_1': n_1,
            'l_1': l_1,
            'j_1': j_1,
            'mj_1': mj_1,
            'n_2': n_2,
            'l_2': l_2,
            'j_2': j_2,
            'mj_2': mj_2,
            'detuning': detuning,
            'w_ac_0': w_ac_0,
            'w_ac': w_ac
        },
        'nearby_rydberg': {
            'energy_space': energy_space,
            'dl': dl,
            'atomic_states': atomic_states
        },
        'SF_Hamiltonian': {
            'delta_ms': [],
            'ms': []
        },
        'ac_voltage': {
            'v_ac_min': v_ac_min,
            'v_ac_max': v_ac_max,
            'v_ac_point': v_ac_point
        },
        'ground_state': ground_state
    }

    # # Dump settings to a JSON file
    # with open('results/settings.json', 'w') as json_file:
    #     json.dump(setting_dict, json_file, indent=4)

    sim = RydbergSinglePhoton(atom=cs, atom_FS_states=atomic_states, ground_state=ground_state, delay=1e-5,
                              expand_zeeman=True)
    comp_atomic_states = sim.atom_states
    # sim.atom_states = [comp_atomic_states[1], comp_atomic_states[4]]
    total_state = sim.comp_atomic_states_num


    # w_ac
    print('w_ac1:', int(w_ac / 1e6))
    result_list = []
    #### Generate H0
    H0 = sim.generate_H0(Bz)
    print('H0:', np.around(np.diag(H0) / 1e6, 2).astype(int))
    E_ac_list = np.linspace(v_ac_min, v_ac_max, v_ac_point)
    for E_ac in E_ac_list:
        print('=================== E_ac ==================')
        # Drive Hamiltonian
        d, H_g = sim.generate_Edipole_matrix(q_ac)
        H_prime = -d * E_ac
        Rabi = np.abs(H_prime.max()) / 2 / np.pi
        Rabi_rydberg = np.abs(H_g.max()) / 2 / np.pi
        print('Microwave Rabi:', Rabi / 1e6, 'MHz')
        print('Rydberg Rabi:', Rabi_rydberg / 1e6, 'MHz')
        # Time settings
        dt = 2 * np.pi / w_ac / 500
        t_span = (0, 2 / Rabi_rydberg)
        ts = np.arange(*t_span, dt)

        # Initial state |0>
        psi0 = np.zeros(total_state, dtype=complex)
        psi0[-2] = 1
        y0 = np.concatenate((psi0.real, psi0.imag))  # real + imag
        # Solve the ODE
        sol = solve_ivp(schrodinger, t_span, y0, t_eval=ts,rtol=1e-11, atol=1e-12)
        # print(sol.y.shape)
        # # Probabilities
        popu = sol.y[:total_state, :]**2 + sol.y[total_state:, :]**2  # |ψ0|²

        print('min ground state population', popu[-2].min())
        result_list.append(popu[-2].min())

    setting_dict['result'] = result_list
    setting_dict['E_ac'] = E_ac_list
    # Dump settings to a JSON file
    with open('results/raman_leakage_result.json', 'w') as json_file:
        json.dump(setting_dict, json_file, indent=4)