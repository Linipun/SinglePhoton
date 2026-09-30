"""
polarizability_reduction.py
============================
Reduce the DC polarizability of the Cs nP_3/2 (m_j=3/2) gate state by microwave-dressing
it to the dominant nearby D state, and extract the (reduced) polarizability by scanning a
DC field E_dc and fitting the quadratic Stark shift.

Why it works
------------
For Cs 70P_3/2 (m_j=3/2) the scalar+tensor DC polarizability alpha ~= 1.39e6 Hz/(V/m)^2
(= 13.9 GHz/(V/cm)^2) is ~78% due to a single state, 69D_5/2 at +2.08 GHz, plus ~22% from
69D_3/2 at +1.87 GHz.  alpha = 2 sum_k |<k|d_z|t>|^2 / (E_k - E_t); the 69D term is positive
(D above P).  A pi microwave that couples gate<->69D_5/2 moves that state's Floquet photon
replica to E_D - w_mw - E_t = -Delta_c relative to the gate.  Tuning Delta_c flips the sign
and magnitude of the dominant term, so alpha_eff(Delta_c) sweeps through ZERO -- a magic
dressing that cancels the polarizability.

Method (exact, non-RWA Shirley-Floquet -- reuses Code.py)
--------------------------------------------------------
  - microwave (pi, freq w_mw = f[gate->coupled D] + Delta_c, Rabi Omega) enters as the AC
    field H_ac that connects adjacent photon blocks;
  - a static DC field E_dc (pi) enters as H_dc = -E_dc * d_dc on every block diagonal;
  - for each E_dc we diagonalize, track the dressed-gate quasi-energy (argmax overlap with
    the bare gate in the m=0 block), and fit
        dU(E_dc) = -1/2 alpha E_dc^2 - 1/24 beta E_dc^4 + c        [polarizability_fit]
    to extract alpha_eff.

Omega=0 recovers the bare polarizability (single m=0 block = H0 + H_dc), validated against
ARC's StarkMap to <3% (n=60: 4431 vs 4402; n=70: 13875 vs 13500 MHz/(V/cm)^2).

Run:  python polarizability_reduction.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.constants import physical_constants, e, hbar
from arc import Cesium

from Code import SinglePhotonSim2AC, polarizability_fit

cs = Cesium()
a0 = physical_constants['Bohr radius'][0]
Q_PI = {-1: 0.0, 0: 1.0, 1: 0.0}                 # pi polarization (both DC and MW along B)

# ----------------------------------------------------------------------------- config
N       = 70                                     # gate principal quantum number
BZ      = 10.0                                    # axial B [G] (sets m_j; alpha is B-independent)
ES_WIN  = 100.0                                   # basis energy window [GHz] (>=50 converges alpha)
COUPLE  = (N - 1, 2, 2.5)                          # D state to dress to: (n, l, j) -> 69D_5/2
M1      = 2                                        # MW photon sidebands (+-M1); 2 converges (=M1=4)
EDC_MAX = 2.0                                      # DC scan half-range [V/m] (quadratic regime)
EDC_PTS = 9
OUT     = 'polarizability_reduction'
os.makedirs(OUT, exist_ok=True)

HZ_PER_VM2_TO_MHZ_PER_VCM2 = 1e-2                  # a[Hz/(V/m)^2] -> MHz/(V/cm)^2  (x1e-6, x1e4)


def make_basis(n_r, l_r, j_r, es, dl=2):
    """FS states (n,l,j) within +-es GHz of the target, l in 0..dl (S,P,D)."""
    ats = [[n_r, l_r, j_r]]
    for l in range(dl + 1):
        for direction in (-1, +1):
            sh = 0 if direction < 0 else 1
            run = True
            while sh < 150 and run:
                for s in (-0.5, 0.5):
                    n = n_r + direction * sh
                    j = l + s
                    if j <= 0:
                        continue
                    f = cs.getTransitionFrequency(n1=n_r, l1=l_r, j1=j_r, n2=n, l2=l, j2=j, s=0.5)
                    if abs(f) < es * 1e9:
                        if not (n == n_r and l == l_r and abs(j - j_r) < 1e-9):
                            ats.append([n, l, j])
                    elif (direction < 0 and f < -es * 1e9) or (direction > 0 and f > es * 1e9):
                        run = False
                sh += 1
    return ats


def build(n, Bz, es):
    """Return (sim, H0, d_dc, d_mw, gate_idx, coupled_D_idx, ms1_list, comp)."""
    ats = make_basis(n, 1, 1.5, es)
    ms1 = list(range(-M1, M1 + 1))
    sim = SinglePhotonSim2AC(atom=cs, ms1=ms1, ms2=[0], atom_FS_states=ats)
    H0 = sim.generate_H0(Bz)
    d_dc = sim.generate_Edipole_matrix(Q_PI)          # pi DC dipole (Hz/(V/m))
    d_mw = sim.generate_Edipole_matrix(Q_PI)          # pi MW dipole (same structure)
    comp = len(sim.atom_states)

    def idx(nn, ll, jj, mj=1.5):
        return next(i for i, a in enumerate(sim.atom_states)
                    if a[0] == nn and a[1] == ll and abs(a[2] - jj) < 1e-6 and abs(a[3] - mj) < 1e-6)

    gi = idx(n, 1, 1.5)
    di = idx(*COUPLE)
    return sim, H0, d_dc, d_mw, gi, di, ms1, comp


def extract_alpha(sim, H0, d_dc, d_mw, ms1, comp, gi, w_mw, E_mw,
                  Edc_max=EDC_MAX, npts=EDC_PTS):
    """Scan E_dc, track the (dressed) gate quasi-energy, fit -> alpha [MHz/(V/cm)^2]."""
    base = ms1.index(0) * comp                         # gate lives in the m1=0 photon block
    tgt = np.zeros(sim.total_state); tgt[base + gi] = 1.0
    H_ac = -E_mw * d_mw
    zero = 0.0 * d_mw
    Edc = np.linspace(-Edc_max, Edc_max, npts)
    en = np.zeros(npts); pur = np.zeros(npts)
    for i, E in enumerate(Edc):
        H_dc = -E * d_dc
        Hf = sim.generate_shirley_floquet_hamiltonian(H0, H_dc, H_ac, zero, w_mw, w_mw)
        Ev, V = np.linalg.eigh(Hf)
        w = np.abs(tgt @ V) ** 2
        k = int(np.argmax(w)); en[i] = Ev[k]; pur[i] = w[k]
    p, _ = curve_fit(polarizability_fit, Edc, en, p0=[-1e6, 1e2, en[npts // 2]])
    alpha = p[0] * HZ_PER_VM2_TO_MHZ_PER_VCM2
    resid = np.max(np.abs(en - polarizability_fit(Edc, *p)))
    return dict(alpha=alpha, beta=p[1], params=p, Edc=Edc, en=en,
                gate_purity=float(pur.min()), fit_resid_Hz=float(resid))


def bare_alpha(n, Bz, es):
    """Fast bare polarizability (no MW -> single atomic block, no Floquet)."""
    ats = make_basis(n, 1, 1.5, es)
    sim = SinglePhotonSim2AC(atom=cs, ms1=[0], ms2=[0], atom_FS_states=ats)
    H0 = sim.generate_H0(Bz); d_dc = sim.generate_Edipole_matrix(Q_PI)
    gi = next(i for i, a in enumerate(sim.atom_states)
              if a[0] == n and a[1] == 1 and abs(a[2] - 1.5) < 1e-6 and abs(a[3] - 1.5) < 1e-6)
    tgt = np.zeros(sim.total_state); tgt[gi] = 1.0
    Edc = np.linspace(-EDC_MAX, EDC_MAX, EDC_PTS); en = np.zeros(EDC_PTS)
    for i, E in enumerate(Edc):
        Ev, V = np.linalg.eigh(H0 - E * d_dc)
        en[i] = Ev[int(np.argmax(np.abs(tgt @ V) ** 2))]
    p, _ = curve_fit(polarizability_fit, Edc, en, p0=[-1e6, 1e2, en[EDC_PTS // 2]])
    return p[0] * HZ_PER_VM2_TO_MHZ_PER_VCM2, Edc, en, p


def main():
    prog = open(os.path.join(OUT, 'progress.txt'), 'w')
    def log(*a):
        print(*a); print(*a, file=prog); prog.flush()

    cn, cl, cj = COUPLE
    d_gateD = abs(cs.getDipoleMatrixElement(N, 1, 1.5, 1.5, cn, cl, cj, 1.5, 0)) * a0 * e / hbar / 2 / np.pi
    f_gateD = cs.getTransitionFrequency(N, 1, 1.5, cn, cl, cj)                     # Hz gate->D
    log(f'Cs {N}P3/2 m_j=3/2 -> {cn}D{cj}:  f={f_gateD/1e9:.3f} GHz, dipole={d_gateD/1e6:.2f} MHz/(V/m)')

    a_bare, Edc_b, en_b, p_b = bare_alpha(N, BZ, ES_WIN)
    log(f'BARE alpha = {a_bare:.1f} MHz/(V/cm)^2')

    sim, H0, d_dc, d_mw, gi, di, ms1, comp = build(N, BZ, ES_WIN)
    det = np.linspace(-2.0e9, 2.0e9, 41)             # MW detuning Delta_c about gate->D [Hz]
    OMEGAS = [0.5e9, 1.0e9, 2.0e9]                    # Rabi frequencies to compare [Hz]

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(Edc_b, (en_b - en_b[EDC_PTS//2]) / 1e6, 'o', ms=6, label='data (bare)')
    ax[0].plot(Edc_b, (polarizability_fit(Edc_b, *p_b) - p_b[2]) / 1e6, '-',
               label=fr"fit $\alpha$={a_bare:.0f} MHz/(V/cm)$^2$")
    ax[0].set_xlabel('$E_{dc}$ [V/m]'); ax[0].set_ylabel(r'$\Delta U$ [MHz]')
    ax[0].set_title(f'Bare Stark shift & fit (Cs {N}P$_{{3/2}}$ $m_j$=3/2)'); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].axhline(0, color='gray', lw=.8)
    ax[1].axhline(a_bare, color='k', ls=':', lw=1.3, label=f'bare $\\alpha$={a_bare:.0f}')

    for oi, OMEGA in enumerate(OMEGAS):
        E_mw = OMEGA / d_gateD
        alpha_d = np.zeros(len(det)); pur_d = np.zeros(len(det))
        for i, dc in enumerate(det):
            r = extract_alpha(sim, H0, d_dc, d_mw, ms1, comp, gi, w_mw=f_gateD + dc, E_mw=E_mw)
            alpha_d[i] = r['alpha']; pur_d[i] = r['gate_purity']
            log(f'  Omega={OMEGA/1e9:.1f}GHz det={dc/1e9:+.2f}GHz: alpha={alpha_d[i]:8.0f} pur={pur_d[i]:.2f}')
        # zero crossings (magic detunings), only where the dressed gate is still mostly gate
        good = pur_d > 0.5
        sign = np.sign(np.where(good, alpha_d, np.nan))
        magic = []
        for i in range(len(det) - 1):
            if np.isfinite(sign[i]) and np.isfinite(sign[i+1]) and sign[i] != sign[i+1]:
                magic.append(det[i] - alpha_d[i]*(det[i+1]-det[i])/(alpha_d[i+1]-alpha_d[i]))
        log(f'Omega={OMEGA/1e9:.1f} GHz: alpha_eff in [{alpha_d.min():.0f}, {alpha_d.max():.0f}]; '
            f'magic (alpha=0, purity>0.5): {[f"{m/1e9:+.3f} GHz" for m in magic]}')
        ax[1].plot(det/1e9, alpha_d, '-o', ms=2.5, color=f'C{oi}', label=fr'$\Omega$={OMEGA/1e9:.1f} GHz')

    ax[1].set_xlabel(r'MW detuning $\Delta_c$ from gate$\to${}D$_{{{}}}$ [GHz]'.format(cn, cj))
    ax[1].set_ylabel(r'$\alpha_{\rm eff}$ [MHz/(V/cm)$^2$]')
    ax[1].set_ylim(-1.5*abs(a_bare), 1.6*abs(a_bare))
    ax[1].set_title(f'Polarizability reduction by dressing {cn}D$_{{{cj}}}$ (Cs {N}P$_{{3/2}}$)')
    ax[1].legend(loc='upper right', fontsize=9); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f'pol_reduction_n{N}.png'), dpi=130)
    log(f'saved {OUT}/pol_reduction_n{N}.png')
    prog.close()


if __name__ == '__main__':
    main()
