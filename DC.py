from Code import *
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages

# Capture command-line arguments
arg = eval('[' + sys.argv[1] + ']')
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

dc_minus = float(arg[23])
dc_pi = float(arg[24])
dc_plus = int(arg[25])
dc_total = np.sqrt(dc_plus ** 2 + dc_pi ** 2 + dc_min ** 2)

q_dc = {-1: np.sqrt(dc_minus/dc_total),
        0: np.sqrt(dc_pi/dc_total),
        1: np.sqrt(dc_plus/dc_total)}


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
        'q_ac': q_ac,
        'q_dc': q_dc
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
        'delta_ms': delta_ms,
        'ms': ms
    },
    'ac_voltage': {
        'v_ac_min': v_ac_min,
        'v_ac_max': v_ac_max,
        'v_ac_point': v_ac_point
    }
}

# Dump settings to a JSON file
with open('results/settings.json', 'w') as json_file:
    json.dump(setting_dict, json_file, indent=4)

sim = SinglePhotonSim(atom=cs, ms=ms, atom_FS_states=atomic_states)
comp_atomic_states = sim.atom_states
total_state = sim.total_state

# Generate H0
H0 = sim.generate_H0(Bz)

# Generate Hamiltonian from DC polariziblity ###
d_dc = sim.generate_Edipole_matrix(q_dc)
H_dc = -0 * d_dc

# Generate AC Hamiltonian###
d_ac = sim.generate_Edipole_matrix(q_ac)
H_ac = -0 * d_ac

scan_dict = {'all_result': {},
             'alpha': [],
             'beta': [],
             'shift': [],
             'p_prop': [],
             'd_prop': []}

look_states = np.array([0, 1, 2, 3])
photon_states = 0
photon_idx = np.where(np.array(ms) == photon_states)[0][0]
sf_look_states = (photon_idx * len(comp_atomic_states) + look_states).astype(int)
E_ac_is = np.linspace(v_ac_min, v_ac_max, v_ac_point)
prob_result = np.zeros((len(E_ac_is), len(sf_look_states), total_state))
energy_result = np.zeros((len(E_ac_is), total_state))
figs = []

with PdfPages('results/DC_fit.pdf') as pdf:
    for E_ac_i in E_ac_is:
        H_ac = -E_ac_i * d_ac
        E_dc_is = np.linspace(-20, 20, 50)
        print('Evaluating E_ac:', np.where(E_ac_is == E_ac_i)[0][0], '/', len(E_ac_is))
        prob_result = np.zeros((len(E_dc_is), len(sf_look_states), total_state))
        energy_result = np.zeros((len(E_dc_is), total_state))
        for i, E_dc_i in enumerate(E_dc_is):
            # print('Evaluating E_dc:', np.where(E_dc_is==E_dc_i)[0][0],'/',len(E_dc_is) )
            H_dc = -E_dc_i * d_dc
            H_f = sim.generate_shirley_floquet_hamiltonian(H0, H_dc, H_ac, w_ac)
            eig_val, eig_vec = np.linalg.eig(H_f)
            probs = np.abs(eig_vec ** 2)
            prob_result[i] = probs[sf_look_states]
            energy_result[i] = eig_val

        fig, ax = plt.subplots()
        max_band = np.argmax(prob_result[:, 3, :], axis=1)
        band_energy = []
        E_dc_valid = []
        for j, b in enumerate(max_band):
            if energy_result[j, b] < 0:
                band_energy.append(energy_result[j, b])
                E_dc_valid.append(E_dc_is[j])
        E_dc_is = np.array(E_dc_valid)
        band_energy = np.array(band_energy)
        ax.scatter(E_dc_is / 100, band_energy / 1e6, marker='.')
        param, err = curve_fit(polarizability_fit, E_dc_is, band_energy, p0=[3e5, 1e1, 17e6])
        ax.plot(E_dc_is / 100, polarizability_fit(E_dc_is, *param) / 1e6, label=r'$\alpha=$' + '{:.2f} GHz'.format(
            param[0] / 1e5) + r'$(V/cm)^{-2}$' + '\n' + r'$\beta=$' + '{:.2f} MHz'.format(
            param[1] / 1e2) + r'$(V/cm)^{-4}$')
        ax.set_xlabel(r'$E_{dc}$ [V/cm]', fontsize=16)
        ax.set_ylabel(r'$\Delta U$ [MHz]', fontsize=16)
        ax.legend()
        pdf.savefig(fig)
        del fig
        scan_dict['alpha'].append(param[0] / 1e5)
        scan_dict['beta'].append(param[1] / 1e2)
        scan_dict['shift'].append(param[2] / 1e6)

fig, ax = plt.subplots()
ax.plot(E_ac_is, np.array(scan_dict['alpha']) * 1000, 'o')
ax.set_ylabel(r'$\alpha$' + '  [MHz' + '$(V/cm)^{-2}$]', fontsize=15)
ax.set_xlabel(r'V_ac (V/m)', fontsize=15)
# ax.set_ylim([-10000,4000])
ax.axhline(0, c='k')
fig.savefig('results/polarizibility.png')
fig.savefig('results/polarizibility.pdf')