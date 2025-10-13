from arc import *
from scipy.constants import speed_of_light, hbar, e
from scipy.constants import physical_constants
bohr_radius = physical_constants['Bohr radius'][0]
bohr_magneton = physical_constants['Bohr magneton in Hz/T'][0]
import json
import cupy as np
from scipy.integrate import solve_ivp
import os
import sys
from Code import SinglePhotonSim, SinglePhotonSim2AC, SinglePhoton, RydbergLeakage, find_FS_state
import matplotlib.pyplot as plt

cs = Cesium()


class RydbergLeakage(SinglePhoton):
    """Original Ramen calculation"""
    def __init__(self, atom, atom_FS_states, ground_state, delay, ms, expand_zeeman=True):
        super().__init__(atom, atom_FS_states, expand_zeeman=expand_zeeman)
        self.ground_state = ground_state
        self.delay = delay
        self.ms = np.array(ms)
        # total dressed state. This will be the dimension of the SF Hamiltonian
        self.ms_num = len(self.ms)
        self.total_state = self.ms_num * self.comp_atomic_states_num

        self.total_state += 1  # len(ground_state)
        self.len_ground_state = 1  # len(self.ground_state)
        ms_idx = np.where(self.ms == 0)[0][0]
        self.target_fourier_state = 1+(ms_idx*self.comp_atomic_states_num)
    def generate_H0(self, Bz):
        n_r, l_r, j_r, mj_r = self.atom_states[0]
        Zeeman_shift_ryd = self.atom.getZeemanEnergyShift(l=l_r, j=j_r, mj=mj_r, magneticFieldBz=Bz / 10000,
                                                          s=0.5) / (hbar * 2 * np.pi)
        state_frequencies = []
        # for state in self.ground_state:
            # n_r, l_r, j_r, mj = state[0]
            # rabi = state[1] * 2 * np.pi
        # fine_frequency = self.atom.getTransitionFrequency(n1=n_r, l1=l_r, j1=j_r, n2=n_r, l2=l, j2=j, s=0.5)
        Zeeman_shift = self.atom.getZeemanEnergyShift(l=l_r, j=j_r, mj=mj_r, magneticFieldBz=Bz / 10000, s=0.5) / \
                       (hbar * 2 * np.pi)
        Eg = Zeeman_shift - Zeeman_shift_ryd
        for (n, l, j, mj) in self.atom_states:
            fine_frequency = self.atom.getTransitionFrequency(n1=n_r, l1=l_r, j1=j_r, n2=n, l2=l, j2=j, s=0.5)
            Zeeman_shift = self.atom.getZeemanEnergyShift(l=l, j=j, mj=mj, magneticFieldBz=Bz / 10000, s=0.5) / \
                           (hbar * 2 * np.pi)
            state_frequencies.append(fine_frequency + Zeeman_shift - Zeeman_shift_ryd)

        H0 = np.diag(state_frequencies)
        print('H0 shape:', H0.shape)
        return H0, Eg

    def generate_Edipole_matrix(self, qs):
        """
        Generate electric dipole matrix element from list of atomic state

        """
        d = np.zeros((self.comp_atomic_states_num, self.comp_atomic_states_num), dtype=np.complex64)
        for i, (a_n, a_l, a_j, a_mj) in enumerate(self.atom_states):
            for j, (b_n, b_l, b_j, b_mj) in enumerate(self.atom_states):
                q = b_mj - a_mj
                # Filter state that doesn't meet the Dipole selection rules i.e. Delta mj >1, delta l != 1, or Delta j>1
                if abs(q) > 1 or abs(a_l - b_l) != 1 or abs(a_j - b_j) > 1:
                    d[i, j] = 0
                elif i > j:
                    # <n_a, l_a, j_a, mj_a| er | n_b, l_b, j_b mj_b> E_q
                    d[i, j] \
                        = self.atom.getDipoleMatrixElement(n1=a_n, l1=a_l, j1=a_j, mj1=a_mj, n2=b_n, l2=b_l,
                                                               j2=b_j,
                                                               mj2=b_mj, q=q) * qs[
                                  q] * bohr_radius * e / hbar / 2 / np.pi

                else:
                    d[i, j] = 0

        # d += np.conjugate(d.T)
        d *= -1
        print('electric dipole matrix size:', d.shape)

        return d

    def generate_shirley_floquet_hamiltonian(self, H0, H_dc, H_ac, w_ac, E_g):
        """
         Generate Shirley-Floquet Hamiltonian
         :param H0:
         :param H_dc:
         :param H_ac:
         :param w_ac:
         :return:
        """
        H_f = np.zeros((self.total_state, self.total_state), dtype=complex)  # Initialize Hamiltonian
        L_f = np.zeros((self.total_state, self.total_state), dtype=complex)  # Initialize Hamiltonian

        for m_idx, m in enumerate(self.ms):
            for n_idx, n in enumerate(self.ms):
                block = np.zeros((self.comp_atomic_states_num, self.comp_atomic_states_num),
                                 dtype=complex)  # Each block of the Floquet matrix

                if m == n:
                    block += H0 + H_dc + np.eye(self.comp_atomic_states_num) * m * w_ac  # Diagonal blocks
                    # Insert the computed block into the corresponding position in H_f
                    row_start, row_end = m_idx * self.comp_atomic_states_num + 1, (
                                m_idx + 1) * self.comp_atomic_states_num + 1
                    col_start, col_end = n_idx * self.comp_atomic_states_num + 1, (
                                n_idx + 1) * self.comp_atomic_states_num + 1
                    H_f[row_start:row_end, col_start:col_end] = block
                elif n == m - 1:
                    block += 0.5 * H_ac  # Off-diagonal blocks
                    row_start, row_end = m_idx * self.comp_atomic_states_num + 1, (
                                m_idx + 1) * self.comp_atomic_states_num + 1
                    col_start, col_end = n_idx * self.comp_atomic_states_num + 1, (
                                n_idx + 1) * self.comp_atomic_states_num + 1
                    L_f[row_start:row_end, col_start:col_end] = block

        # add ground state energy
        H_f[0, 0] = E_g

        # add coupling between the
        for i, state in enumerate(self.ground_state):
            ground_state = state[0]
            rabi = state[1] * 2 * np.pi
            for j, s in enumerate(self.atom_states):
                if s == ground_state:
                    print(j + len(self.atom_states), j)
                    H_f[0, j + self.target_fourier_state] = rabi
                    H_f[j + self.target_fourier_state, 0] = np.conjugate(rabi)

        return H_f, L_f

# ---------- Core RHS (operator form, no vectorization to d^2) ----------
def lindblad_rhs(rho, H, Ls):
    """
    rho: (d, d) density matrix on GPU (complex64/complex128)
    H:   (d, d) Hamiltonian (Hermitian)
    Ls:  list/tuple of collapse operators L_k (each (d,d))
    Returns drho/dt (d, d)
    """
    i = 1j
    comm = H @ rho - rho @ H  # [H, rho]
    drho = -i * comm

    for L in Ls:
        Lrho = L @ rho
        Ldag = L.conj().T
        term = Lrho @ Ldag
        A = Ldag @ L
        drho += term - 0.5 * (A @ rho + rho @ A)
    return drho

# ---------- RK4 integrator on GPU ----------
def rk4_step(f, y, dt, *args):
    k1 = f(y, *args)
    k2 = f(y + 0.5*dt*k1, *args)
    k3 = f(y + 0.5*dt*k2, *args)
    k4 = f(y + dt*k3, *args)
    return y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

 def evolve_lindblad(H, Ls, rho0, t0, t1, dt, symmetrize_every=25, renorm=True):
        """
        Evolve rho from t0 to t1 with fixed-step RK4 on GPU.
        Returns times (cp.ndarray) and rhos (list of cp.ndarray or cp.stack if memory permits)
        """
        assert rho0.shape[0] == rho0.shape[1] == H.shape[0]
        nsteps = int(np.round((t1 - t0) / dt).get())
        t = t0
        rho = rho0.astype(np.complex64 if rho0.dtype == np.complex64 else np.complex128, copy=True)

        times = [t0]
        rhos = [rho.copy()]

        for n in range(1, nsteps + 1):
            rho = rk4_step(lindblad_rhs, rho, dt, H, Ls)

            # Optional maintenance to control numerical drift
            if symmetrize_every and (n % symmetrize_every == 0):
                rho = 0.5 * (rho + rho.conj().T)  # enforce Hermiticity
                if renorm:
                    tr = cp.trace(rho)
                    # re-normalize only if slightly off
                    rho = rho / tr

            t = t0 + n * dt
            times.append(t)
            rhos.append(rho.copy())

        return np.asarray(times), rhos


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
    delta_ms = 1  # int(arg[3])
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

    total_state = sim.total_state


    sim = RydbergLeakage(atom=cs, atom_FS_states=atomic_states, ground_state=ground_state, delay=1e-6, ms=ms1,
                         expand_zeeman=False)
    H0, E_g = sim.generate_H0(Bz)
    # print(np.around(H0))
    d = sim.generate_Edipole_matrix(q_ac_1)
    for row, E_ac_i1 in enumerate([v_ac1]):
        for col, E_ac_i2 in enumerate([50]):
        # for col, E_ac_i2 in enumerate(E_ac_is2):

            # Initial state |0>
            psi0 = np.zeros((total_state, total_state), dtype=np.complex64)
            psi0[0, 0] = 1

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

