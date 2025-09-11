from Code import SinglePhotonSim2AC, find_FS_state, hbar, polarizability_fit
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages
import sys
import os
import numpy as np
from arc import *
import json
import matplotlib.pyplot as plt
import seaborn as sns
cs = Cesium()
import pickle

def main(H0, H_dc, H_ac1, H_ac2, w_ac_1, w_ac_2):
    H_f = sim.generate_shirley_floquet_hamiltonian(H0, H_dc, H_ac1, H_ac2, w_ac_1, w_ac_2)
    eig_val, eig_vec = np.linalg.eig(H_f)
    prob = np.abs(eig_vec ** 2)
    prob_state = prob[sf_look_states]
    state_idx = np.argmax(prob_state, axis=1)
    energy = eig_val[state_idx]
    return energy.astype(np.float64)


if __name__ == "__main__":
    # Capture command-line arguments
    # arg = eval('[' + sys.argv[1] + ']')
    folder = 'results'  # f'2AC-delta{detuning/1e6}-energy{energy_space}dl{dl}'
    os.makedirs(folder, exist_ok=True)

    # ### atomic property #########
    # Rydberg state
    n_r = 60  # int(arg[0]) #
    l_r = 1  # int(arg[1])
    j_r = 3/2  # float(arg[2])
    mj_r = 3/2  # float(arg[3])

    # Magnetic field
    Bz = 1  # float(arg[4])  # G

    # Nearby Rydberg state
    energy_space = 60  # float(arg[1])  # GHz
    dl = 2  # int(arg[2])
    atomic_states = find_FS_state(n_r, l_r, j_r, energy_space, dl, folder)
    # ### atomic property #########

    q_dc = {-1: 0.0, 0: 1.0, 1: 0.0}

    # ########### property of E1 shiftout###############
    # Transition parameters
    n_1 = n_r  # int(arg[8])
    l_1 = l_r  # int(arg[9])
    j_1 = j_r  # float(arg[10])
    mj_1 = 1/2  # float(arg[11])

    n_2 = n_r+1  # int(arg[12])
    l_2 = 0  # int(arg[13])
    j_2 = 1/2  # float(arg[14])
    mj_2 = 1/2  # float(arg[15])
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
    delta_ms = 3  # int(arg[3])
    ms1 = list(range(-delta_ms, delta_ms + 1))

    # Field strength scan parameters
    v_ac_min = 0  # float(arg[5])
    v_ac_max = 100  # float(arg[6])
    v_ac_point = 40  # int(arg[7])
    E_ac_is = np.linspace(v_ac_min, v_ac_max, v_ac_point)
    # ########### property of E1 shiftout###############

    # ########### property of E polarizability  ###############
    # Transition parameters
    n_1 = n_r  # int(arg[8])
    l_1 = l_r  # int(arg[9])
    j_1 = j_r  # float(arg[10])
    mj_1 = mj_r  # float(arg[11])

    n_2 = n_r-1  # int(arg[12])
    l_2 = 2  # int(arg[13])
    j_2 = 5/2  # float(arg[14])
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
    delta_ms2 = 3  # int(arg[4])
    ms2 = list(range(-delta_ms, delta_ms + 1))

    # Field strength scan parameters
    v_ac_min2 = 0  # float(arg[8])
    v_ac_max2 = 60  # float(arg[9])
    v_ac_point2 = 30  # int(arg[10])
    E_ac_is2 = np.linspace(v_ac_min2, v_ac_max2, v_ac_point2)
    # ########### property of E2 ###############


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
            'v_ac_min': v_ac_min,
            'v_ac_max': v_ac_max,
            'v_ac_point': v_ac_point,
            'v_ac_min2': v_ac_min2,
            'v_ac_max2': v_ac_max2,
            'v_ac_point2': v_ac_point2
        }
    }

    # Dump settings to a JSON file
    with open(f'{folder}/settings.json', 'w') as json_file:
        json.dump(setting_dict, json_file, indent=4)

    sim = SinglePhotonSim2AC(atom=cs, ms1=ms1, ms2=ms2, atom_FS_states=atomic_states)
    comp_atomic_states = sim.atom_states
    total_state = sim.total_state

    # Generate H0
    H0 = sim.generate_H0(Bz)

    # Generate Hamiltonian from DC polariziblity ###
    d_dc = sim.generate_Edipole_matrix(q_dc)


    # Generate AC Hamiltonian###
    d_ac_1 = sim.generate_Edipole_matrix(q_ac_1)
    d_ac_2 = sim.generate_Edipole_matrix(q_ac_2)

    scan_dict = {'all_result': {},
                 'alpha': [],
                 'beta': [],
                 'shift': [],
                 'p_prop': [],
                 'd_prop': []}

    look_states = np.array([0, 1, 2, 3])
    photon_states_1 = 0
    photon_states_2 = 0
    photon_idx_1 = np.where(np.array(ms1) == photon_states_1)[0][0]
    photon_idx_2 = np.where(np.array(ms2) == photon_states_2)[0][0]
    sf_look_states = ((photon_idx_2*len(ms1)+photon_idx_1) * len(comp_atomic_states) + look_states).astype(int)

    figs = []
    shiftout = np.zeros((v_ac_point, v_ac_point2))
    alpha = np.zeros((v_ac_point, v_ac_point2))
    pdf = PdfPages(f'{folder}/DC_fit.pdf')
    for row, E_ac_i1 in enumerate(E_ac_is):
        H_ac1 = -E_ac_i1 * d_ac_1
        for col, E_ac_i2 in enumerate(E_ac_is2):
            print(f'E1={E_ac_i1}({row/len(E_ac_is)}), E2={E_ac_i2} ({col/len(E_ac_is2)})')
            H_ac2 = -E_ac_i2 * d_ac_2

            # Zero-DC field
            H_dc = -0*d_dc
            energy = main(H0, H_dc, H_ac1, H_ac2, w_ac_1, w_ac_2)
            base_frequency = energy[3]
            shiftout[row, col] = energy[3]-energy[2]

            # find alpha(polarizability)
            E_dc_list = np.linspace(-1, 1, 30)
            energy_dc = np.zeros(len(E_dc_list))
            i = 0
            for E_dc in E_dc_list:
                H_dc = -E_dc * d_dc
                energy = main(H0, H_dc, H_ac1, H_ac2, w_ac_1, w_ac_2)
                energy_dc[i] = energy[3]
                i += 1

            fig, ax = plt.subplots()
            label = []
            ax.scatter(E_dc_list, energy_dc/1e6)
            ax.set_ylabel('$\Delta U$ [MHz]', fontsize=15)
            ax.set_xlabel('$E_{dc}$ [V/m]', fontsize=15)
            ax.set_title('E1={},E2={}'.format(E_ac_i1, E_ac_i2))
            param, err = curve_fit(polarizability_fit, E_dc_list, energy_dc, p0=[-6e4, 7e1, base_frequency])
            ax.plot(E_dc_list, polarizability_fit(E_dc_list, *param) / 1e6, '-', linewidth=3,
                    label=r'$\alpha=$' + '{:.2f} GHz'.format(
                        param[0] / 1e5) + r'$(V/cm)^{-2}$' + '\n' + r'$\beta=$' + '{:.2f} MHz'.format(
                        param[1] / 1e2) + r'$(V/cm)^{-4}$')
            pdf.savefig(fig)
            alpha[row, col] = param[0] / 1e6 * 1e4  # 1e6-> MHz, 1e4 -> cm^2
            del(fig)
    pdf.close()
    fig2, ax2 = plt.subplots(ncols=2, figsize=(10, 5))

    # First heatmap (Shiftout)
    sns.heatmap(
        shiftout/1e6,
        ax=ax2[0],
        cmap="Greens",
        cbar=True,
        xticklabels=np.round(E_ac_is2, 2),  # X-axis ticks
        yticklabels=np.round(E_ac_is, 2)  # Y-axis ticks
    )
    ax2[0].set_title("Shiftout[MHz]")
    ax2[0].set_xlabel("$E_{pol}$ (V/m)")
    ax2[0].set_ylabel("$E_{shift}$ (V/m)")
    ax2[0].invert_yaxis()  # so low field is at bottom

    # Second heatmap (Polarizability)
    sns.heatmap(
        alpha,
        ax=ax2[1],
        cmap="Blues",
        cbar=True,
        xticklabels=np.round(E_ac_is2, 2),
        yticklabels=np.round(E_ac_is, 2)
    )
    ax2[1].set_title("Polarizability$ [MHz/cm^2]$")
    ax2[1].set_xlabel("$E_{pol}$ (V/m)")
    ax2[1].set_ylabel("$E_{shift}$ (V/m)")
    ax2[1].invert_yaxis()

    # Improve layout (so tick labels don't overlap)
    fig2.tight_layout()
    fig2.savefig(f"{folder}/result.pdf", dpi=300)

    with open(f"{folder}/result.pkl", "wb") as f:
        pickle.dump({"shiftout": shiftout, "polarizability": alpha}, f)

    # print('bad')
    # with PdfPages('results/DC_fit.pdf') as pdf:
    #     for fig in figs:
    #         pdf.savefig(fig)