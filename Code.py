from arc import *
from scipy.constants import hbar, e
from scipy.constants import physical_constants
import numpy as np
import sys
import matplotlib.colors as c
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import json
import os
import pickle
bohr_radius = physical_constants['Bohr radius'][0]
bohr_magneton = physical_constants['Bohr magneton in Hz/T'][0]


# Cesium Atom object in Arc Akali
cs = Cesium()

## ground state
n_g = 6
l_g = 0
j_g = 1 / 2
f_g = 4
mf_g = 0
i_cs = cs.I

orbital = {0: 's',
           1: 'p',
           2: 'd',
           3: 'f',
           4: 'g',
           5: 'h',
           6: 'i'}

# Cesium Atom object in Arc Akali
cs = Cesium()

class SinglePhoton:
    def __init__(self, atom, atom_FS_states,expand_zeeman=True):
        self.atom = atom
        self.atom_FS_states = atom_FS_states
        # add Zeeman states to the atomic FS structure #####
        if expand_zeeman:
            self.atom_states = []
            for a_n, a_l, a_j in self.atom_FS_states:
                for a_mj in np.arange(-a_j, a_j + 1, 1):
                    self.atom_states.append([a_n, a_l, a_j, a_mj])
        else:
            self.atom_states = atom_FS_states
        # count the number of states
        self.comp_atomic_states_num = len(self.atom_states)

    def generate_H0(self, Bz):
        n_r, l_r, j_r, mj_r = self.atom_states[0]
        Zeeman_shift_ryd = self.atom.getZeemanEnergyShift(l=l_r, j=j_r, mj=mj_r, magneticFieldBz=Bz / 10000, s=0.5) / (
                hbar * 2 * np.pi)
        state_frequencies = []
        for (n, l, j, mj) in self.atom_states:
            fine_frequency = self.atom.getTransitionFrequency(n1=n_r, l1=l_r, j1=j_r, n2=n, l2=l, j2=j, s=0.5)
            Zeeman_shift = self.atom.getZeemanEnergyShift(l=l, j=j, mj=mj, magneticFieldBz=Bz / 10000, s=0.5) / \
                           (hbar * 2 * np.pi)
            state_frequencies.append(fine_frequency + Zeeman_shift - Zeeman_shift_ryd)
        H0 = np.diag(state_frequencies)
        print('H0 shape:', H0.shape)
        return H0

    def generate_H_t(self, qs, w_ac, t, E_ac, Bz):
        return -self.generate_H_t(qs, w_ac, t) * E_ac + self.generate_H0(Bz)

    def generate_Edipole_matrix(self, qs):
        """
        Generate electric dipole matrix element from list of atomic state

        """
        d = []
        for i, (a_n, a_l, a_j, a_mj) in enumerate(self.atom_states):
            row = []
            for j, (b_n, b_l, b_j, b_mj) in enumerate(self.atom_states):
                q = b_mj - a_mj
                # Filter state that doesn't meet the Dipole selection rules i.e. Delta mj >1, delta l != 1, or Delta j>1
                if abs(q) > 1 or abs(a_l - b_l) != 1 or abs(a_j - b_j) > 1:
                    row.append(0)
                elif i < j:
                    # <n_a, l_a, j_a, mj_a| er | n_b, l_b, j_b mj_b> E_q
                    row.append(
                        self.atom.getDipoleMatrixElement(n1=a_n, l1=a_l, j1=a_j, mj1=a_mj, n2=b_n, l2=b_l, j2=b_j,
                                                         mj2=b_mj, q=q)
                        * qs[q] * bohr_radius * e / hbar / 2 / np.pi)

                else:
                    row.append(0)

            d.append(row)

        d = np.array(d)
        d += np.conjugate(d.T)
        d *= -1
        print('electric dipole matrix size:', d.shape)
        return d

    @staticmethod
    def print_FS_state(label):
        return '{}{}_{}, mf={}'.format(label[0], orbital[label[1]], label[2], label[3])

    @staticmethod
    def print_atomic_state(label):
        return '{}{}_{}'.format(label[0], orbital[label[1]], label[2])

    def find_idx(self, state):
        # idx = []
        for i, s in enumerate(self.atom_states):
            if s == state:
                return i
                # idx.append(i)


class RydbergSinglePhoton(SinglePhoton):
    def __init__(self, atom, atom_FS_states, ground_state, delay, expand_zeeman=True):
        super().__init__(atom, atom_FS_states, expand_zeeman=expand_zeeman)
        self.ground_state = ground_state
        self.delay = delay
        self.comp_atomic_states_num += len(ground_state)

    def generate_H0(self, Bz):
        n_r, l_r, j_r, mj_r = self.atom_states[0]
        Zeeman_shift_ryd = self.atom.getZeemanEnergyShift(l=l_r, j=j_r, mj=mj_r, magneticFieldBz=Bz / 10000,
                                                          s=0.5) / (
                                   hbar * 2 * np.pi)
        state_frequencies = []
        for (n, l, j, mj) in self.atom_states:
            fine_frequency = self.atom.getTransitionFrequency(n1=n_r, l1=l_r, j1=j_r, n2=n, l2=l, j2=j, s=0.5)
            Zeeman_shift = self.atom.getZeemanEnergyShift(l=l, j=j, mj=mj, magneticFieldBz=Bz / 10000, s=0.5) / \
                           (hbar * 2 * np.pi)
            state_frequencies.append(fine_frequency + Zeeman_shift - Zeeman_shift_ryd)
        for state in self.ground_state:
            ground_state = state[0]
            n, l, j, mj = state[1]
            rabi = state[2] * 2 * np.pi
            fine_frequency = self.atom.getTransitionFrequency(n1=n_r, l1=l_r, j1=j_r, n2=n, l2=l, j2=j, s=0.5)
            Zeeman_shift = self.atom.getZeemanEnergyShift(l=l, j=j, mj=mj, magneticFieldBz=Bz / 10000, s=0.5) / \
                           (hbar * 2 * np.pi)
            state_frequencies.append(fine_frequency + Zeeman_shift - Zeeman_shift_ryd)
        H0 = np.diag(state_frequencies)
        print('H0 shape:', H0.shape)
        return H0

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
                elif i < j:
                    # <n_a, l_a, j_a, mj_a| er | n_b, l_b, j_b mj_b> E_q
                    d[i, j] = self.atom.getDipoleMatrixElement(n1=a_n, l1=a_l, j1=a_j, mj1=a_mj, n2=b_n, l2=b_l,
                                                               j2=b_j,
                                                               mj2=b_mj, q=q) * qs[
                                  q] * bohr_radius * e / hbar / 2 / np.pi

                else:
                    d[i, j] = 0

        d += np.conjugate(d.T)
        d *= -1
        print('electric dipole matrix size:', d.shape)

        H_ground = np.zeros((self.comp_atomic_states_num, self.comp_atomic_states_num), dtype=np.complex64)
        for i, state in enumerate(self.ground_state):
            ground_state = state[1]
            rabi = state[2] * 2 * np.pi
            for j, s in enumerate(self.atom_states):
                if s == ground_state:
                    # print(i + len(self.atom_states), j)
                    H_ground[i + len(self.atom_states), j] = rabi
        H_ground += np.conjugate(H_ground.T)
        return d, H_ground

class SinglePhotonSim(SinglePhoton):
    def __init__(self, atom, ms, atom_FS_states):
        super().__init__(atom, atom_FS_states)
        self.ms = ms
        # total dressed state. This will be the dimension of the SF Hamiltonian
        self.ms_num = len(self.ms)
        self.total_state = self.ms_num * self.comp_atomic_states_num

    def generate_shirley_floquet_hamiltonian(self, H0, H_dc, H_ac, w_ac):
        """
         Generate Shirley-Floquet Hamiltonian
         :param H0:
         :param H_dc:
         :param H_ac:
         :param w_ac:
         :return:
        """
        H_f = np.zeros((self.total_state, self.total_state), dtype=complex)  # Initialize Hamiltonian

        for m_idx, m in enumerate(self.ms):
            for n_idx, n in enumerate(self.ms):
                block = np.zeros((self.comp_atomic_states_num, self.comp_atomic_states_num),
                                 dtype=complex)  # Each block of the Floquet matrix

                if m == n:
                    block += H0 + H_dc + np.eye(self.comp_atomic_states_num) * m * w_ac  # Diagonal blocks
                elif n == m + 1 or n == m - 1:
                    block += 0.5 * H_ac  # Off-diagonal blocks

                # Insert the computed block into the corresponding position in H_f
                row_start, row_end = m_idx * self.comp_atomic_states_num, (m_idx + 1) * self.comp_atomic_states_num
                col_start, col_end = n_idx * self.comp_atomic_states_num, (n_idx + 1) * self.comp_atomic_states_num
                H_f[row_start:row_end, col_start:col_end] = block
        return H_f

    def find_energy_bands(self, eig_val, eig_vec, target_state=0, target_photon_count=0,
                          target_num_bands=4, resolve_error=False):
        if (np.imag(eig_val) != 0).any():
            raise Exception("Eigen values contain non-zero imaginary values")
        band_energy = []
        idn = []
        prop = []
        # iterate through column.
        for j in range(self.total_state):
            # generate column of probabilty
            col_probability = abs(eig_vec[:, j] ** 2)

            # initialized zeros array to track dressed state without Zeeman sublevel probability
            P_dressed_state = np.zeros(self.ms_num * len(self.atom_FS_states))

            index = 0
            i = 0
            # loop over all dressed states
            for _ in self.ms:
                for state in self.atom_FS_states:
                    # sum over the mj state
                    total_mj = int(2 * state[2] + 1)
                    P_dressed_state[i] = np.sum(col_probability[index:index + total_mj])
                    index += total_mj
                    i += 1

            # find the state with the max probability
            arg = np.argmax(P_dressed_state)
            state = arg % len(self.atom_FS_states)
            photon_count = arg // len(self.atom_FS_states)

            # get the top 2 states population
            ind = col_probability.argsort()[-5:][::-1]
            state2 = ind % self.comp_atomic_states_num
            photon_count2 = ind // self.comp_atomic_states_num
            # filter to Rydberg state and 0-photon
            if state == target_state and self.ms[photon_count] == target_photon_count:
                band_energy.append(eig_val[j])
                idn.append(ind)
                prop.append(col_probability[ind])

        p = np.argsort(band_energy)
        band_energy = np.array(band_energy)[p]
        prop = [prop[i] for i in p]
        idn = [idn[i] for i in p]

        if len(band_energy) != target_num_bands:
            if resolve_error:
                return None, None, None
            else:
                raise Exception("ERROR number of bands found({}) doesn't equal to target band({})"
                                .format(len(band_energy), target_num_bands))

        return band_energy, idn, prop

    def find_nearest_energy_bands(self, eig_val, eig_vec, target_num_bands=4, target_photon_count=0):

        # find the bare atomic dressed states(fourier component m=0) that closest to the target states
        if (np.imag(eig_val) != 0).any():
            raise Exception("Eigen values contain non-zero imaginary values")

        # calculate absolute distance eignal value matrix
        abs_eig_val = np.abs(eig_val)

        # find the target fourier component m=target_photon_count index
        # target_ms_idn = np.where(np.array(ms)==target_photon_count)[0][0]

        # sort by energy shift
        inds = abs_eig_val.argsort()

        target_states = inds[:target_num_bands]

        # find energy, index and components of top bare states,
        band_energy = []
        idn = []
        prop = []
        for j in target_states:
            # generate column of probabilty
            col_probability = abs(eig_vec[:, j] ** 2)

            # get the top 5 states population
            ind = col_probability.argsort()[-5:][::-1]

            #
            band_energy.append(eig_val[j])
            idn.append(ind)
            prop.append(col_probability[ind])

        p = np.argsort(band_energy)
        band_energy = np.array(band_energy)[p]
        prop = [prop[i] for i in p]
        idn = [idn[i] for i in p]

        return band_energy, idn, prop

    def find_energy_bands_no_mixing(self, eig_val, eig_vec, target_state=[0, 1, 2, 3], target_photon_count=0):
        if (np.imag(eig_val) != 0).any():
            raise Exception("Eigen values contain non-zero imaginary values")

        target_num_bands = len(target_state)
        band_energy = np.zeros(target_num_bands)
        idn = []
        prop = []
        # iterate through column.
        for j in range(self.total_state):
            # generate column of probabilty
            col_probability = abs(eig_vec[:, j] ** 2)

            # get the top 2 states population
            ind = col_probability.argsort()[-5:][::-1]
            state2 = ind % self.comp_atomic_states_num
            photon_count2 = ind // self.comp_atomic_states_num

            # check whether the state with highest probability is in the target_state.
            target_state_index = np.where(target_state == state2[0])[0]
            # filter to target state and 0-photon
            if len(target_state_index) == 1 and self.ms[photon_count2[0]] == target_photon_count:
                band_energy[target_state_index[0]] = eig_val[j]
                idn.append(ind[0])
                prop.append(col_probability[ind])

        if len(idn) != target_num_bands:
            raise Exception("ERROR found more than 4 bands:{}".format(len(idn)))

        return band_energy, idn, prop


class SinglePhotonSim2AC(SinglePhotonSim):
    def __init__(self, atom, ms1, ms2, atom_FS_states):
        super().__init__(atom,ms1,atom_FS_states)
        self.ms2 = ms2

        # total dressed state. This will be the dimension of the SF Hamiltonian
        self.ms_num = len(self.ms)
        self.ms2_num = len(self.ms2)

        self.total_state = self.ms_num * self.ms2_num * self.comp_atomic_states_num

    def generate_shirley_floquet_hamiltonian(self, H0, H_dc, H_ac1, H_ac2, w_ac1, w_ac2):
        H_f = np.zeros((self.total_state, self.total_state), dtype=complex)  # Initialize Hamiltonian

        for m2_idx, m2 in enumerate(self.ms2):
            for n2_idx, n2 in enumerate(self.ms2):
                block1 = np.zeros((self.comp_atomic_states_num*self.ms_num,
                                   self.comp_atomic_states_num*self.ms_num), dtype=complex)
                if m2 == n2:
                    block1 += np.eye(self.comp_atomic_states_num*self.ms_num) * m2 * w_ac2
                    for m_idx, m in enumerate(self.ms):
                        for n_idx, n in enumerate(self.ms):
                            block = np.zeros((self.comp_atomic_states_num, self.comp_atomic_states_num),
                                             dtype=complex)  # Each block of the Floquet matrix
                            if m == n:
                                block += H0 + H_dc +np.eye(self.comp_atomic_states_num) * m * w_ac1# Diagonal element
                            elif n == m + 1 or n == m - 1:
                                block += 0.5 * H_ac1  # Off-diagonal blocks

                            # Insert the computed block into the corresponding position in H_f
                            row_start, row_end = m_idx * self.comp_atomic_states_num, (
                                    m_idx + 1) * self.comp_atomic_states_num
                            col_start, col_end = n_idx * self.comp_atomic_states_num, (
                                    n_idx + 1) * self.comp_atomic_states_num
                            block1[row_start:row_end, col_start:col_end] = block

                elif n2 == m2 + 1 or n2 == m2 - 1:
                    for m_idx, m in enumerate(self.ms):
                        for n_idx, n in enumerate(self.ms):
                            block = np.zeros((self.comp_atomic_states_num, self.comp_atomic_states_num),
                                             dtype=complex)  # Each block of the Floquet matrix
                            if n == m:
                                block += 0.5 * H_ac2  # Off-diagonal blocks

                            # Insert the computed block into the corresponding position in H_f
                            row_start, row_end = m_idx * self.comp_atomic_states_num, (
                                    m_idx + 1) * self.comp_atomic_states_num
                            col_start, col_end = n_idx * self.comp_atomic_states_num, (
                                    n_idx + 1) * self.comp_atomic_states_num
                            block1[row_start:row_end, col_start:col_end] = block

                # Insert the computed block into the corresponding position in H_f
                row_start = m2_idx * self.ms_num * self.comp_atomic_states_num
                row_end = (m2_idx + 1) * self.ms_num * self.comp_atomic_states_num
                col_start = n2_idx * self.ms_num * self.comp_atomic_states_num
                col_end = (n2_idx + 1) * self.ms_num * self.comp_atomic_states_num
                H_f[row_start:row_end, col_start:col_end] = block1
        return H_f

def polarizability_fit(E, a, b, c):
    return -1 / 2 * a * E ** 2 - 1 / (4 * 3 * 2) * b * E ** 4 + c


def find_FS_state(n_r, l_r, j_r, energy_space, dl, pathname, verbose=True):
    atomic_states = [[n_r, l_r, j_r]]
    for l in range(dl + 1):
        # shift down
        shift_n = 0
        run_down = True
        while shift_n < 100 and run_down:
            for s in [-1 / 2, 1 / 2]:
                n = n_r - shift_n
                j = l + s
                if j > 0 and abs(cs.getTransitionFrequency(n1=n_r, l1=l_r, j1=j_r, n2=n, l2=l, j2=j,
                                                           s=0.5)) < energy_space * 1e9:
                    if verbose:
                        print(n, l, j, np.around(
                            cs.getTransitionFrequency(n1=n_r, l1=l_r, j1=j_r, n2=n, l2=l, j2=j, s=0.5) / 1e9, 2))
                    if n != n_r or l != l_r or j != j_r:
                        atomic_states.append([n, l, j])
                elif cs.getTransitionFrequency(n1=n_r, l1=l_r, j1=j_r, n2=n, l2=l, j2=j,
                                               s=0.5) < -energy_space / 2 * 1e9:
                    # print('stop',n,l,j,np.around(cs.getTransitionFrequency(n1=n_r, l1=l_r, j1=j_r, n2=n, l2=l,
                    # j2=j, s=0.5)/1e9,2))
                    run_down = False
                # else:
                # print('stop',n,l,j,np.around(cs.getTransitionFrequency(n1=n_r, l1=l_r, j1=j_r, n2=n,
                # l2=l, j2=j, s=0.5)/1e9,2))

            shift_n += 1

        # shift up
        shift_n = 1
        run_up = True
        while shift_n < 100 and run_up:
            for s in [-1 / 2, 1 / 2]:
                n = n_r + shift_n
                j = l + s
                if j > 0 and abs(cs.getTransitionFrequency(n1=n_r, l1=l_r, j1=j_r, n2=n, l2=l, j2=j,
                                                           s=0.5)) < energy_space * 1e9:
                    if verbose:
                        print(n, l, j, np.around(
                            cs.getTransitionFrequency(n1=n_r, l1=l_r, j1=j_r, n2=n, l2=l, j2=j, s=0.5) / 1e9, 2))
                    atomic_states.append([n, l, j])
                elif cs.getTransitionFrequency(n1=n_r, l1=l_r, j1=j_r, n2=n, l2=l, j2=j,
                                               s=0.5) > energy_space / 2 * 1e9:
                    run_up = False
                    # print('stop',n,l,j,np.around(cs.getTransitionFrequency(n1=n_r, l1=l_r, j1=j_r, n2=n, l2=l,
                    # j2=j, s=0.5)/1e9,2))
            shift_n += 1

    nmin = int(np.min(np.array(atomic_states)[:, 0])) - 2
    nmax = int(np.max(np.array(atomic_states)[:, 0])) + 2

    lvlplot = LevelPlot(cs)
    lvlplot.makeLevels(nmin, nmax, 0, dl, sList=[0.5])
    lvlplot.drawLevels(units='GHz')

    datum_energy = cs.getEnergy(n_r, l_r, j_r) * e / (hbar * 2 * np.pi) / 1e9
    lvlplot.ax.axhline(datum_energy + energy_space, linestyle=':', c='k')
    lvlplot.ax.axhline(datum_energy - energy_space, linestyle=':', c='k')
    lvlplot.ax.set_ylim([datum_energy - (energy_space + 10), datum_energy + (energy_space + 10)])
    for state in atomic_states:
        (n, l, j) = state
        energy = cs.getEnergy(n, l, l + 1 / 2) * e / (hbar * 2 * np.pi) / 1e9
        if np.abs(datum_energy - energy) < energy_space:
            # print(energy )
            lvlplot.ax.text(l - 0.12, energy + 1, '$' + SinglePhotonSim.print_atomic_state(state)[:3] + '$')
    lvlplot.fig.savefig(os.path.join(pathname,
        'lvlplot{}-{:d}-{:d}.pdf'.format(SinglePhotonSim.print_atomic_state([n_r, l_r, j_r]), int(energy_space), dl)))

    return atomic_states


if __name__ == "__main__":
    # Capture command-line arguments
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

    q_dc = {-1: 0.0, 0: 0.0, 1: 0.0}

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

    result_dict = {'band0': [],
                   'band1': [],
                   'band2': [],
                   'band3': [],
                   'eig_vec': [],
                   'prop': [],
                   'scan_x': []}

    look_states = np.array([0, 1, 2, 3])
    photon_states = np.array([0, 0, 0, 0])
    sf_look_states = []
    for st, ph in zip(look_states, photon_states):
        photon_idx = np.where(np.array(ms) == ph)[0][0]
        sf_look_states.append((photon_idx * len(comp_atomic_states) + st).astype(int))

    sf_look_states = np.array(sf_look_states)
    E_ac_is = np.linspace(v_ac_min, v_ac_max, v_ac_point)
    prob_result = np.zeros((len(E_ac_is), len(sf_look_states), total_state))
    energy_result = np.zeros((len(E_ac_is), total_state))
    for i, E_ac_i in enumerate(E_ac_is):
        print('Evaluating E_ac:', np.where(E_ac_is == E_ac_i)[0][0], '/', len(E_ac_is))
        H_ac = -E_ac_i * d_ac
        H_f = sim.generate_shirley_floquet_hamiltonian(H0, H_dc, H_ac, w_ac)
        eig_val, eig_vec = np.linalg.eig(H_f)
        probs = np.abs(eig_vec ** 2)
        prob_result[i] = probs[sf_look_states]
        energy_result[i] = eig_val

    color = ['Blues', 'Oranges', 'Greens', 'Reds', 'Purples']

    fig, ax = plt.subplots(figsize=(8, 5))
    label = []
    norm = c.Normalize(vmin=0, vmax=1, clip=False)
    for j in range(len(sf_look_states[:])):
        for state in range(total_state):
            axis = ax.scatter(E_ac_is[:], energy_result[:, state] / 1e6, c=prob_result[:, j, state],
                              alpha=prob_result[:, j, state], cmap=color[j])  # color[j])
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=color[j]), ax=ax, fraction=0.04, pad=0.02)

        label.append('$mj=' + '{}/2'.format(-3 + 2 * i))
    # ax.legend(label)
    ax.set_ylim([-1e3, 1e3])

    # ax.set_xlim([0,100])
    ax.set_ylabel('$\Delta U$ [MHz]', fontsize=15)
    ax.set_xlabel('$E_{ac}$ [V/m]', fontsize=15)
    fig.tight_layout()

    name = 'results/shiftout'
    fig.savefig(name+'.png')
    fig.savefig(name+'.pdf')

    result_dict = {'V_ac': E_ac_is,
                   'energy_result': energy_result,
                   'prob_result': prob_result,
                   'look_state': sf_look_states,
                   }
    
    with open('results/sf_result.pkl', 'wb') as json_file:
        pickle.dump(result_dict,json_file)
