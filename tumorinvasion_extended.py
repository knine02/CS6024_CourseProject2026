"""
=============================================================================
Hybrid PDE-ABM Model: Tumor Invasion with Gene Expression Vectors  v3
=============================================================================
Based on:
  Franssen et al. (2019) -- PDE framework for tumor invasion
  Compucell3D / Aguilar et al. -- MMP-TIMP R-D system & dual ECM fields

Extensions over v1/v2:
  1. Dual ECM: w_f (fibrillar, drives haptotaxis) + w_nf (nonfibrillar, quiescence)
  2. MMP-TIMP Turing R-D system
       dA/dt = D_A*lap(A) + a - delta_A*A        A = MMP (activator)
       dI/dt = D_I*lap(I) + b - delta_I*I        I = TIMP (inhibitor)
       a = b = K_eff(h_bar) - (c*I - d*A)        [only at cell-occupied nodes]
     Gene 2 (MMP Secretion) sets K_eff = gain of the autocatalytic loop.
     ECM threshold lysis: [A]/[I] > 2.0 -> pixel lysed.
     Lysed pixels regenerate after ecm_regen_delay steps (cancer matrisome).
  3. Two new adhesion genes:
       Gene 6: Cell-Cell Adhesion  J_cc   -- suppresses D when neighbours present
       Gene 7: Cell-ECM Adhesion   J_cECM -- gates haptotaxis + quiescence signal
  4. Per-gene mutation probabilities + magnitudes (small for structural genes)
  5. Hard carrying capacity per grid node
  6. Phenotype presets replicating paper parameter regimes
  7. Invasion area (AU) metric + comparison utilities

PHENOTYPE REGIMES:
  homeostatic         high J_cc, high J_cECM, no R-D
  carcinoma_in_situ   high J_cc, high J_cECM, no R-D
  apolar_cluster      low  J_cc, low  J_cECM, no R-D
  multiscale_invasion low  J_cc, low  J_cECM, R-D ON

NOTEBOOK USAGE:
    sim = TumorSimulation.from_phenotype("multiscale_invasion")
    sim.run()
    sim.plot_snapshots()
    sim.animate(save_path="tumor.mp4", fps=5)

    results = run_phenotype_comparison()
    plot_invasion_comparison(results)
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import pandas as pd
import warnings
warnings.filterwarnings("ignore")



# Keys are 0-based gene indices in the expression vector h_i.
# 'P_mut'   : per-gene mutation probability on division
# 'mut_mag' : std of Gaussian mutation step (keep small for structural genes)

GENE_ROLES: Dict[int, Dict] = {
    0: {"name": "Motility (EMT)",       "function": "motility",       "baseline": 0.3,  "P_mut": 0.05, "mut_mag": 0.12},
    1: {"name": "Haptotaxis",           "function": "haptotaxis",     "baseline": 0.3,  "P_mut": 0.05, "mut_mag": 0.12},
    2: {"name": "MMP Secretion",        "function": "mmp_secretion",  "baseline": 0.4,  "P_mut": 0.10, "mut_mag": 0.15},
    3: {"name": "ECM Degradation",      "function": "ecm_degradation","baseline": 0.3,  "P_mut": 0.05, "mut_mag": 0.10},
    4: {"name": "Proliferation",        "function": "proliferation",  "baseline": 0.6,  "P_mut": 0.07, "mut_mag": 0.12},
    5: {"name": "Apoptosis Resistance", "function": "apoptosis",      "baseline": 0.4,  "P_mut": 0.04, "mut_mag": 0.08},
    6: {"name": "Cell-Cell Adhesion",   "function": "adhesion_cc",    "baseline": 0.7,  "P_mut": 0.10, "mut_mag": 0.20},
    7: {"name": "Cell-ECM Adhesion",    "function": "adhesion_ecm",   "baseline": 0.7,  "P_mut": 0.10, "mut_mag": 0.20},
    # Add more genes here -- the rest of the code picks them up automatically.
    # Example:
    # 8: {"name": "Angiogenesis", "function": "angiogenesis", "baseline": 0.2, "P_mut": 0.03, "mut_mag": 0.08},
}
N_GENES = len(GENE_ROLES)



PHENOTYPE_PRESETS: Dict[str, Dict] = {
    "homeostatic": {
        # High adhesion, near-zero motility/MMP -> growth arrest
        0: 0.1, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.3, 5: 0.2, 6: 0.9, 7: 0.9,
        "rd_ecm": False,
    },
    "carcinoma_in_situ": {
        # High adhesion, moderate MMP -> contained proliferation, no invasion
        0: 0.2, 1: 0.2, 2: 0.4, 3: 0.3, 4: 0.7, 5: 0.5, 6: 0.8, 7: 0.8,
        "rd_ecm": False,
    },
    "apolar_cluster": {
        # Low adhesion -> loss of polarity, occasional single-cell subinvasion
        0: 0.4, 1: 0.4, 2: 0.4, 3: 0.4, 4: 0.7, 5: 0.5, 6: 0.2, 7: 0.2,
        "rd_ecm": False,
    },
    "multiscale_invasion": {
        # Low adhesion + R-D ECM kinetics -> full multiscale invasion
        0: 0.5, 1: 0.5, 2: 0.6, 3: 0.5, 4: 0.7, 5: 0.6, 6: 0.2, 7: 0.2,
        "rd_ecm": True,
    },
}

GENE_THRESHOLDS = {
    "motility":    {"low": 0.33, "high": 0.66},   # gene 0
    "haptotaxis":  {"low": 0.33, "high": 0.66},   # gene 1
    "mmp":         {"low": 0.33, "high": 0.66},   # gene 2
    "ecm_deg":     {"low": 0.33, "high": 0.66},   # gene 3
    "apoptosis":   {"low": 0.33, "high": 0.66},   # gene 5
    "cc_adhesion": {"low": 0.33, "high": 0.66},   # gene 6
    "ecm_adhesion":{"low": 0.33, "high": 0.66},   # gene 7
}
CELL_TYPE_COLORS = {
    "Adhesive / Quiescent":   "#4C72B0",   # high adhesion, low motility
    "Adhesive / Motile":      "#64B5CD",   # high adhesion, high motility — EMT initiating
    "Motile / Low Adhesion":  "#FF7C00",   # lost adhesion, moving — EMT intermediate
    "Invasive / Degradative": "#C44E52",   # low adhesion + high motility + MMP + ECM deg
    "Mixed / Intermediate":   "#8172B2",   # no dominant trait
}
def _classify_cell(h: np.ndarray) -> str:
    g_mot  = h[0]; g_hap  = h[1]
    g_mmp  = h[2]; g_ecmd = h[3]
    g_cc   = h[6]; g_ecma = h[7]

    hi = {k: v["high"] for k, v in GENE_THRESHOLDS.items()}
    lo = {k: v["low"]  for k, v in GENE_THRESHOLDS.items()}

    high_adhesion = (g_cc  >= hi["cc_adhesion"]) and (g_ecma >= hi["ecm_adhesion"])
    low_adhesion  = (g_cc  <  lo["cc_adhesion"]) or  (g_ecma <  lo["ecm_adhesion"])
    high_motility = (g_mot >= hi["motility"])     or  (g_hap  >= hi["haptotaxis"])
    high_deg      = (g_mmp >= hi["mmp"])          and (g_ecmd >= hi["ecm_deg"])

    if high_adhesion and not high_motility:
        return "Adhesive / Quiescent"
    elif high_adhesion and high_motility:
        return "Adhesive / Motile"
    elif low_adhesion and high_motility and high_deg:
        return "Invasive / Degradative"
    elif low_adhesion and high_motility:
        return "Motile / Low Adhesion"
    else:
        return "Mixed / Intermediate"

def _sigmoid(x: float, k: float = 8.0, x0: float = 0.5) -> float:
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def _laplacian(f: np.ndarray, dx: float) -> np.ndarray:
    """2-D central-difference Laplacian with zero-flux (Neumann) boundaries."""
    lap = np.zeros_like(f)
    lap[1:-1, 1:-1] = (
        f[2:, 1:-1] + f[:-2, 1:-1] + f[1:-1, 2:] + f[1:-1, :-2]
        - 4 * f[1:-1, 1:-1]
    ) / dx**2
    for axis in [0, 1]:
        lap = np.swapaxes(lap, 0, axis)
        fs  = np.swapaxes(f,   0, axis)
        lap[0,  1:-1] = (fs[1,  1:-1] - fs[0,  1:-1]) / dx**2
        lap[-1, 1:-1] = (fs[-2, 1:-1] - fs[-1, 1:-1]) / dx**2
        lap = np.swapaxes(lap, 0, axis)
    return lap



class GeneExpressionFunctions:
    """
    Maps gene expression to PDE / agent parameters.

    Convention
    ----------
    h_i   : individual cell vector  (movement, division, death decisions)
    h_bar : spatial mean at a grid node  (PDE source terms -- collective effect)
    """

    def __init__(self, params: "SimulationParams"):
        self.p = params

    # ---- Gene 0 + 6:  Random motility D(h_i, local_density) ----------------
    def D(self, h_i: np.ndarray, local_density: float) -> float:
        """
        D = base_motility * adhesion_suppression
        High cell-cell adhesion (g6) suppresses motility when neighbours present.
          adhesion_suppression = 1 - g6 * tanh(rho / rho_half)
        This models contact inhibition of locomotion.
        """
        base_D   = self.p.D_min + (self.p.D_max - self.p.D_min) * _sigmoid(h_i[0], k=8)
        rho_half = self.p.carrying_capacity / 2.0
        supp     = 1.0 - h_i[6] * np.tanh(local_density / rho_half)
        return max(self.p.D_min * 0.1, base_D * supp)

    #  Gene 1 + 7:  Haptotaxis toward fibrillar ECM 
    def Phi_fibrillar(self, h_i: np.ndarray) -> float:
        """
        Phi = Phi_max * sigmoid(g1) * g7
        Cell-ECM adhesion gene (g7) gates whether the cell can bind
        and follow fibrillar ECM gradients at all.
        """
        return self.p.Phi_max * _sigmoid(h_i[1], k=8) * h_i[7]

    # Gene 7:  Quiescence signal from nonfibrillar ECM 
    def quiescence_signal(self, h_i: np.ndarray, w_nf_local: float) -> float:
        """
        High g7 (cell-ECM adhesion) + high w_nf -> strong quiescence.
        Nonfibrillar ECM signals anoikis / growth arrest in epithelial-like cells.
        q = g7 * w_nf_local   enters proliferation_prob as a suppressor.
        """
        return h_i[7] * w_nf_local

    #  Gene 2 (collective):  MMP-TIMP system gain K_eff(h_bar)
    def K_eff(self, h_bar: np.ndarray) -> float:
        """
        Baseline constitutive secretion rate for both A (MMP) and I (TIMP).
          K_eff = K_base * sigmoid(g2, threshold=0.4)

        This is 'K' in:  a = b = K_eff - (c*[I] - d*[A])
        High g2 -> high K_eff -> autocatalytic MMP loop runs faster ->
        [A]/[I] crosses the ECM lysis threshold (2.0) more readily.
        Collective: mean g2 over all cells at the node sets the loop gain.
        """
        return self.p.K_base * _sigmoid(h_bar[2], k=6, x0=0.4)

    # Gene 3 (collective):  Direct contact ECM degradation 
    def Gamma1(self, h_bar: np.ndarray) -> float:
        """
        Contact-mediated direct fibrillar ECM degradation.
          Gamma1 = Gamma1_max * g3^2
        Quadratic: needs high expression to have significant effect.
        """
        return self.p.Gamma1_max * (h_bar[3] ** 2)

    # Gene 4 + 7:  Proliferation probability 

    def proliferation_prob(self, h_i: np.ndarray,
                           local_density: float, 
                           w_nf_local: float, 
                           w_f_local: float) -> float:
        """
        P_div = P_max * sigmoid(g4) * volume_factor * quiescence_factor
        Volume factor: ECM acts as a physical solid. Cells + ECM cannot exceed 1.0.
        """
        # Calculate physical volume occupied by the thickest ECM layer
        ecm_volume = max(w_f_local, w_nf_local) 
        cell_volume = local_density / self.p.carrying_capacity
        
        # The remaining empty space for a new cell
        volume_f = max(0.0, 1.0 - (cell_volume + ecm_volume))
        
        quiesc_f = max(0.0, 1.0 - self.quiescence_signal(h_i, w_nf_local))
        return self.p.P_div_max * _sigmoid(h_i[4], k=6) * volume_f * quiesc_f

    # Gene 5:  Apoptosis resistance 
    def apoptosis_prob(self, h_i: np.ndarray) -> float:
        """P_apop = P_apop_base * (1 - 0.9 * g5)"""
        return self.p.P_apop_base * (1.0 - 0.9 * h_i[5])


# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================
@dataclass
class SimulationParams:
    # Grid
    grid_size: int        = 80
    dx: float             = 1.0
    dt: float             = 0.01
    n_steps: int          = 500
    save_every: int       = 25

    # MMP-TIMP R-D system (Aguilar et al.)
    use_rd_ecm: bool      = True    # False -> simple MMP (Franssen fallback)
    D_A: float            = 0.01   # MMP activator diffusion  (slow, localised)
    D_I: float            = 0.8    # TIMP inhibitor diffusion (fast, delocal)
    delta_A: float        = 0.003  # MMP decay
    delta_I: float        = 0.003  # TIMP decay
    K_base: float         = 2.0    # baseline secretion scaled by gene 2
    c_TIMP: float         = 4.0    # TIMP inhibition coefficient
    d_MMP: float          = 2.0    # MMP autocatalysis coefficient
    AI_threshold: float   = 2.0    # [A]/[I] ratio for threshold ECM lysis
    ecm_regen_delay: int  = 20     # steps before lysed pixel regenerates

    # Simple MMP fallback (use_rd_ecm=False)
    D_m: float            = 0.005
    Lambda: float         = 0.2
    Theta_max: float      = 0.8

    # Dual ECM fields
    Gamma1_max: float     = 0.3    # max contact-mediated fibrillar degradation
    Gamma2_f: float       = 0.2    # MMP-mediated fibrillar degradation rate
    Gamma_nf: float       = 0.05   # nonfibrillar degradation (contact only)
    w_nf_init: float      = 0.8    # initial nonfibrillar ECM density
    w_f_init: float       = 1.0    # initial fibrillar ECM density

    # Cell motility
    D_min: float          = 0.0005
    D_max: float          = 0.012
    Phi_max: float        = 0.06

    # Cell lifecycle
    P_div_max: float      = 0.08
    P_apop_base: float    = 0.005
    carrying_capacity: float = 3.0   # hard cap: max cells per grid node
    max_cells_total: int  = 15000     # global cell ceiling

    # Mutation fallbacks 
    P_mut_global: float   = 0.04
    mut_mag_global: float = 0.10

    # Misc
    n_initial_cells: int  = 20
    seed: int             = 42



# CANCER CELL AGENT

class CancerCell:
    _id_counter = 0

    def __init__(self, x: int, y: int, h: np.ndarray):
        CancerCell._id_counter += 1
        self.id  = CancerCell._id_counter
        self.x   = x
        self.y   = y
        self.h   = np.clip(h.copy(), 0.0, 1.0)
        self.age = 0

    def mutate(self, p: SimulationParams) -> np.ndarray:
        """
        Per-gene independent stochastic mutation on division.
        Each gene uses its own P_mut and mut_mag from GENE_ROLES.
        Adhesion genes (6, 7) have lower P_mut and mut_mag by default,
        keeping phenotype more stable while activity genes evolve freely.
        Output is clamped to [0, 1].
        """
        h_new = self.h.copy()
        for i, role in GENE_ROLES.items():
            if np.random.rand() < role.get("P_mut", p.P_mut_global):
                h_new[i] += np.random.normal(0, role.get("mut_mag", p.mut_mag_global))
        return np.clip(h_new, 0.0, 1.0)


# MAIN SIMULATION CLASS

class TumorSimulation:
    """
    Hybrid PDE-ABM tumor invasion simulation.

    Quick-start
    -----------
    sim = TumorSimulation.from_phenotype("multiscale_invasion")
    sim.run()
    sim.plot_snapshots()
    sim.plot_AI_ratio()
    sim.animate(save_path="tumor.mp4", fps=5)

    Phenotype comparison (replicates paper Fig 1F / Fig 2D):
    ---------------------------------------------------------
    results = run_phenotype_comparison()
    plot_invasion_comparison(results)
    plot_spatial_comparison(results)
    """

    def __init__(self, params: SimulationParams = None,
                 gene_baselines: Optional[Dict[int, float]] = None):
        self.p   = params or SimulationParams()
        self.gef = GeneExpressionFunctions(self.p)
        np.random.seed(self.p.seed)
        CancerCell._id_counter = 0

        N = self.p.grid_size
        self.A    = np.zeros((N, N))                      # MMP activator
        self.I    = np.ones((N, N)) * 0.1                 # TIMP inhibitor
        self.rho  = np.zeros((N, N))                      # cell density
        
        # MUST BE np.zeros, otherwise the grid defaults to solid walls
        self.w_f  = np.zeros((N, N))
        self.w_nf = np.zeros((N, N))

        self._lysed_at: Dict[Tuple[int, int], int] = {}

        self._gene_baselines = {
            i: (gene_baselines.get(i, role["baseline"]) if gene_baselines
                else role["baseline"])
            for i, role in GENE_ROLES.items()
        }

        self.cells:             List[CancerCell] = []
        self.snapshots:         List[Dict]       = []
        self.transcriptome_log: List[Dict]       = []
        self.invasion_log:      List[Dict]       = []
        self.step_count = 0


        self._init_environment()
        

        self._init_cells()

    def _init_environment(self):
        N = self.p.grid_size
        cx, cy = N // 2, N // 2
        
        r_tumor = max(2, N // 14)
        r_capsule = r_tumor + 3

        # 1. Generate Fibrous Stroma (Fibrillar ECM)
        fiber_density = 0.15 
        fiber_length = 6
        num_fibers = int(N * N * fiber_density)
        
        for _ in range(num_fibers):
            x0, y0 = np.random.randint(0, N, 2)
            angle = np.random.uniform(0, 2 * np.pi)
            for step in range(fiber_length):
                x = int(x0 + step * np.cos(angle))
                y = int(y0 + step * np.sin(angle))
                if 0 <= x < N and 0 <= y < N:
                    self.w_f[x, y] = self.p.w_f_init

        # 2. Add Basement Membrane Capsule (Non-fibrillar ECM)
        for x in range(N):
            for y in range(N):
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                if dist <= r_tumor:
                    self.w_f[x, y] = 0.0
                    self.w_nf[x, y] = 0.0
                elif dist <= r_capsule:
                    self.w_f[x, y] = 0.0
                    self.w_nf[x, y] = self.p.w_nf_init
                else:
                    self.w_nf[x, y] = 0.0
    @classmethod
    def from_phenotype(cls, name: str,
                       params: SimulationParams = None) -> "TumorSimulation":
        """
        Build simulation pre-configured for a named phenotype.
        name: 'homeostatic' | 'carcinoma_in_situ' | 'apolar_cluster' | 'multiscale_invasion'
        """
        assert name in PHENOTYPE_PRESETS, \
            f"Unknown phenotype '{name}'. Options: {list(PHENOTYPE_PRESETS)}"
        preset = PHENOTYPE_PRESETS[name]
        p = params or SimulationParams()
        p.use_rd_ecm = preset["rd_ecm"]
        return cls(params=p, gene_baselines={i: preset[i] for i in range(N_GENES)})


    def _init_cells(self):
        N = self.p.grid_size
        cx, cy = N // 2, N // 2
        radius = max(2, N // 14)
        for _ in range(self.p.n_initial_cells):
            x = int(np.clip(cx + np.random.randint(-radius, radius + 1), 0, N - 1))
            y = int(np.clip(cy + np.random.randint(-radius, radius + 1), 0, N - 1))
            if self.rho[x, y] < self.p.carrying_capacity:
                h = np.array([
                    np.clip(self._gene_baselines[i] + np.random.normal(0, 0.04), 0, 1)
                    for i in range(N_GENES)
                ])
                self.cells.append(CancerCell(x, y, h))
                self.rho[x, y] += 1

    def _compute_mean_field_h(self) -> np.ndarray:
        N = self.p.grid_size
        h_sum   = np.zeros((N, N, N_GENES))
        h_count = np.zeros((N, N))
        for c in self.cells:
            h_sum[c.x, c.y]   += c.h
            h_count[c.x, c.y] += 1
        h_bar        = np.zeros((N, N, N_GENES))
        mask         = h_count > 0
        h_bar[mask]  = h_sum[mask] / h_count[mask, np.newaxis]
        return h_bar

    # ==== PDE STEP ===========================================================

    def _pde_step(self, h_bar: np.ndarray):
        if self.p.use_rd_ecm:
            self._step_rd(h_bar)
        else:
            self._step_simple(h_bar)
        self._ecm_step(h_bar)
        self._ecm_regeneration()

    def _step_rd(self, h_bar: np.ndarray):
        """
        MMP-TIMP Turing R-D:
          dA/dt = D_A*lap(A) + a - delta_A*A
          dI/dt = D_I*lap(I) + b - delta_I*I
          a = b = K_eff(h_bar) - (c*I - d*A)   [only at cell-occupied nodes]

        K_eff is controlled by Gene 2 (MMP Secretion):
          high g2 -> K_eff large -> autocatalytic loop drives [A] up fast
          -> [A]/[I] crosses lysis threshold 2.0 -> ECM lysed.
        """
        N = self.p.grid_size
        K_field   = np.array([[self.gef.K_eff(h_bar[i, j])
                                for j in range(N)] for i in range(N)])
        cell_mask = (self.rho > 0).astype(float)
        source    = np.clip(
            (K_field - (self.p.c_TIMP * self.I - self.p.d_MMP * self.A)) * cell_mask,
            -5.0, 5.0)
        self.A = np.clip(
            self.A + self.p.dt * (self.p.D_A * _laplacian(self.A, self.p.dx)
                                  + source - self.p.delta_A * self.A), 0, None)
        self.I = np.clip(
            self.I + self.p.dt * (self.p.D_I * _laplacian(self.I, self.p.dx)
                                  + source - self.p.delta_I * self.I), 0, None)

    def _step_simple(self, h_bar: np.ndarray):
        """Fallback: simple MMP diffusion + decay (Franssen). I unused."""
        N = self.p.grid_size
        K_field = np.array([[self.gef.K_eff(h_bar[i, j])
                              for j in range(N)] for i in range(N)])
        self.A = np.clip(
            self.A + self.p.dt * (
                self.p.D_m * _laplacian(self.A, self.p.dx)
                + K_field * self.rho - self.p.Lambda * self.A), 0, None)

    def _ecm_step(self, h_bar: np.ndarray):
        """
        Fibrillar ECM degradation (two routes):
          1. Continuous: dw_f/dt = -(Gamma1(h_bar)*rho + Gamma2*A) * w_f
          2. Threshold lysis (R-D only): [A]/[I] > 2.0 -> pixel set to 0
             Cost: A and I each reduced by 1.5 at lysed pixel (paper eq)

        Nonfibrillar ECM (slower, contact-only):
          dw_nf/dt = -Gamma_nf * rho * w_nf
          Drives quiescence via Gene 7 (Cell-ECM Adhesion).
        """
        N = self.p.grid_size
        G1 = np.array([[self.gef.Gamma1(h_bar[i, j])
                         for j in range(N)] for i in range(N)])
        self.w_f = np.clip(
            self.w_f + self.p.dt * (
                -(G1 * self.rho + self.p.Gamma2_f * self.A) * self.w_f),
            0.0, 1.0)

        if self.p.use_rd_ecm:
            safe_I     = np.where(self.I > 1e-9, self.I, 1e-9)
            lysis_mask = (self.A / safe_I > self.p.AI_threshold) & (self.w_f > 0.05)
            xs, ys     = np.where(lysis_mask)
            for ix, iy in zip(xs, ys):
                if (int(ix), int(iy)) not in self._lysed_at:
                    self._lysed_at[(int(ix), int(iy))] = self.step_count
                    self.w_f[ix, iy]  = 0.0
                    self.w_nf[ix, iy] = max(0.0, self.w_nf[ix, iy] - 0.1)
                    self.A[ix, iy]    = max(0.0, self.A[ix, iy] - 1.5)
                    self.I[ix, iy]    = max(0.0, self.I[ix, iy] - 1.5)

        self.w_nf = np.clip(
            self.w_nf + self.p.dt * (-self.p.Gamma_nf * self.rho * self.w_nf),
            0.0, 1.0)

    def _ecm_regeneration(self):
        """
        Lysed fibrillar pixels regenerate after ecm_regen_delay steps.
        Models cancer matrisome: tumor cells secrete new fibrillar collagen.
        """
        done = [k for k, t0 in self._lysed_at.items()
                if self.step_count - t0 >= self.p.ecm_regen_delay]
        for k in done:
            ix, iy = k
            self.w_f[ix, iy] = min(1.0, self.w_f[ix, iy] + 0.3)
            del self._lysed_at[k]

   # AGENT MOVEMENT 

    def _move_cells(self):
        N = self.p.grid_size
        for cell in self.cells:
            x, y  = cell.x, cell.y
            D_i   = self.gef.D(cell.h, self.rho[x, y])
            Phi_i = self.gef.Phi_fibrillar(cell.h)
            
            # Get cell's adhesion gene (Gene 6)
            g6_adhesion = cell.h[6] 
            
            probs = []
            for nx, ny in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]:
                if 0 <= nx < N and 0 <= ny < N:
               
                    ecm_vol = max(self.w_f[nx, ny], self.w_nf[nx, ny])
                    cell_vol = self.rho[nx, ny] / self.p.carrying_capacity
                    if cell_vol + ecm_vol >= 1.0:
                        probs.append((nx, ny, 0.0)); continue
                    
                
                    target_rho = self.rho[nx, ny]
                    current_rho = self.rho[x, y]
                    adhesion_penalty = 1.0
                  
                    if target_rho < current_rho:
                        # High g6 = closer to 0.0 multiplier (movement blocked)
                        adhesion_penalty = max(0.01, 1.0 - g6_adhesion)
                        
                    
                    p_diff = D_i / self.p.dx**2
                    p_hapt = max(0.0, Phi_i * (self.w_f[nx, ny] - self.w_f[x, y]) / self.p.dx)
                    
                    # Multiply the total desire to move by the Velcro penalty
                    final_prob = (p_diff + p_hapt) * adhesion_penalty
                    probs.append((nx, ny, final_prob))
                else:
                    probs.append((nx, ny, 0.0))
            p_move = sum(v for _, _, v in probs) * self.p.dt
            if p_move > 1e-9 and np.random.rand() < min(p_move, 0.85):
                w = np.array([v for _, _, v in probs])
                if w.sum() > 0:
                    idx = np.random.choice(len(probs), p=w / w.sum())
                    nx, ny, _ = probs[idx]
                    self.rho[x, y]   -= 1
                    cell.x, cell.y    = nx, ny
                    self.rho[nx, ny] += 1

   #CELL LIFECYCLE 

    def _update_cells(self):
        new_cells = []; dead_ids = set()
        N = self.p.grid_size

        for cell in self.cells:
            cell.age += 1
            rho_l = self.rho[cell.x, cell.y]
            wnf_l = self.w_nf[cell.x, cell.y]
            wf_l  = self.w_f[cell.x, cell.y]

            # Apoptosis
            if np.random.rand() < self.gef.apoptosis_prob(cell.h):
                dead_ids.add(cell.id)
                self.rho[cell.x, cell.y] = max(0, self.rho[cell.x, cell.y] - 1)
                continue

            # Division
            if len(self.cells) + len(new_cells) < self.p.max_cells_total:
                # Pass the wf_l parameter into the updated probability function
                if np.random.rand() < self.gef.proliferation_prob(cell.h, rho_l, wnf_l, wf_l):
                    h_d = cell.mutate(self.p)
                    cands = [(cell.x+dx, cell.y+dy)
                             for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]
                             if 0 <= cell.x+dx < N and 0 <= cell.y+dy < N]
                    np.random.shuffle(cands)
                    placed = False
                    
                    # Try to place daughter in neighbor, checking physical ECM volume
                    for cx2, cy2 in cands:
                        ecm_vol = max(self.w_f[cx2, cy2], self.w_nf[cx2, cy2])
                        cell_vol = self.rho[cx2, cy2] / self.p.carrying_capacity
                        if cell_vol + ecm_vol < 1.0:  # Physical space exists!
                            new_cells.append(CancerCell(cx2, cy2, h_d))
                            self.rho[cx2, cy2] += 1
                            placed = True; break
                    
                    # Try to place in own spot if neighbors are physically full
                    if not placed:
                        ecm_vol = max(self.w_f[cell.x, cell.y], self.w_nf[cell.x, cell.y])
                        cell_vol = self.rho[cell.x, cell.y] / self.p.carrying_capacity
                        if cell_vol + ecm_vol < 1.0:
                            new_cells.append(CancerCell(cell.x, cell.y, h_d))
                            self.rho[cell.x, cell.y] += 1

        self.cells = [c for c in self.cells if c.id not in dead_ids] + new_cells

    # ==== METRICS ============================================================

    def _invasion_area(self) -> float:
        """
        Cells outside the initial seed radius.
        Matches paper's 'Area of Invasion (AU)'.
        """
        N = self.p.grid_size
        cx, cy  = N // 2, N // 2
        r_seed  = max(2, N // 14) + 2
        return float(sum(
            1 for c in self.cells
            if np.sqrt((c.x - cx)**2 + (c.y - cy)**2) > r_seed
        ))

    #  LOGGING 
    def _log_transcriptomics(self, t: int):
        for c in self.cells:
            rec = {"timestep": t, "cell_id": c.id,
                   "x": c.x, "y": c.y, "age": c.age}
            for i, role in GENE_ROLES.items():
                rec[role["name"]] = c.h[i]
            self.transcriptome_log.append(rec)

    def _save_snapshot(self, t: int):
        ia = self._invasion_area()
        self.snapshots.append({
            "t": t, "rho": self.rho.copy(),
            "A": self.A.copy(), "I": self.I.copy(),
            "w_f": self.w_f.copy(), "w_nf": self.w_nf.copy(),
            "cells": [(c.x, c.y, c.h.copy()) for c in self.cells],
            "n_cells": len(self.cells), "invasion_area": ia,
        })
        self.invasion_log.append({"t": t, "n_cells": len(self.cells),
                                  "invasion_area": ia})

    # MAIN LOOP 

    def run(self, verbose: bool = True):
        mode = "R-D MMP-TIMP" 
        print(f"Simulation | {self.p.n_steps} steps | "
              f"{self.p.grid_size}x{self.p.grid_size} | {N_GENES} genes | {mode}")
        self._save_snapshot(0)
        self._log_transcriptomics(0)

        for t in range(1, self.p.n_steps + 1):
            h_bar = self._compute_mean_field_h()
            self._pde_step(h_bar)
            self._move_cells()
            self._update_cells()
            self.step_count = t

            if t % self.p.save_every == 0:
                self._save_snapshot(t)
                self._log_transcriptomics(t)
                if verbose:
                    s = self.snapshots[-1]
                    print(f"  t={t:4d} | cells={s['n_cells']:5d} | "
                          f"maxA={self.A.max():.3f} | "
                          f"minWf={self.w_f.min():.3f} | "
                          f"inv={s['invasion_area']:.0f}")

        print(f"Done. cells={len(self.cells)} inv={self._invasion_area():.0f}")


    # VISUALISATION
 

    def plot_snapshots(self, figsize_col: Tuple = (3.2, 3.0)):
        """5-row snapshot grid: rho | A | I | w_f | w_nf."""
        snaps   = self.snapshots
        ncols   = len(snaps)
        configs = [
            ("rho",  "Cell Density",     "hot_r", None),
            ("A",    "MMP [A]",          "Reds",  None),
            ("I",    "TIMP [I]",         "Blues", None),
            ("w_f",  "Fibrillar ECM",    "YlGn",  1.0),
            ("w_nf", "Nonfibrillar ECM", "BuGn",  1.0),
        ]
        fig, axes = plt.subplots(5, ncols,
                                 figsize=(figsize_col[0] * ncols,
                                          figsize_col[1] * 5))
        if ncols == 1:
            axes = axes[:, np.newaxis]
        for col, snap in enumerate(snaps):
            for row, (key, label, cmap, vmax) in enumerate(configs):
                ax = axes[row, col]
                d  = snap[key]
                vm = vmax or (d.max() if d.max() > 0 else 1.0)
                im = ax.imshow(d.T, origin="lower", cmap=cmap, vmin=0, vmax=vm)
                ax.set_title(f"t={snap['t']}", fontsize=7)
                if col == 0: ax.set_ylabel(label, fontsize=7)
                ax.set_xticks([]); ax.set_yticks([])
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.suptitle("Tumor Invasion Fields", fontsize=11, fontweight="bold")
        plt.tight_layout(); plt.show()

    def plot_population_dynamics(self):
        """Cell count, invasion area, and mean gene expression over time."""
        inv   = pd.DataFrame(self.invasion_log)
        snaps = self.snapshots
        times = [s["t"] for s in snaps]
        gene_means = {GENE_ROLES[i]["name"]: [] for i in range(N_GENES)}
        for s in snaps:
            if s["cells"]:
                hs = np.stack([h for (_, _, h) in s["cells"]])
                for i, role in GENE_ROLES.items():
                    gene_means[role["name"]].append(hs[:, i].mean())
            else:
                for n in gene_means: gene_means[n].append(np.nan)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(inv["t"], inv["n_cells"], "k-o", ms=3, lw=2)
        axes[0].set_title("Population"); axes[0].set_xlabel("t")
        axes[0].set_ylabel("Cell count"); axes[0].grid(alpha=0.3)

        axes[1].plot(inv["t"], inv["invasion_area"], "r-o", ms=3, lw=2)
        axes[1].set_title("Invasion Area (AU)"); axes[1].set_xlabel("t")
        axes[1].set_ylabel("Cells outside seed"); axes[1].grid(alpha=0.3)

        cmap = plt.cm.tab10
        for idx, (name, vals) in enumerate(gene_means.items()):
            axes[2].plot(times, vals, "-o", ms=2, lw=1.5,
                         color=cmap(idx / N_GENES), label=name)
        axes[2].set_title("Mean Gene Expression"); axes[2].set_xlabel("t")
        axes[2].set_ylim(0, 1); axes[2].legend(fontsize=6, ncol=2)
        axes[2].grid(alpha=0.3)
        plt.suptitle("Dynamics", fontsize=11, fontweight="bold")
        plt.tight_layout(); plt.show()

    def plot_gene_distributions(self, snapshot_idx: int = -1):
        snap = self.snapshots[snapshot_idx]
        if not snap["cells"]: return
        hs    = np.stack([h for (_, _, h) in snap["cells"]])
        ncols = min(N_GENES, 4)
        nrows = (N_GENES + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.5*ncols, 3*nrows))
        axes_flat = list(np.array(axes).flat)
        for i, (ax, role) in enumerate(zip(axes_flat, GENE_ROLES.values())):
            ax.hist(hs[:, i], bins=20, range=(0, 1),
                    color=plt.cm.tab10(i / N_GENES), edgecolor="white", alpha=0.85)
            ax.set_title(role["name"], fontsize=8)
            ax.set_xlim(0, 1)
        for ax in axes_flat[N_GENES:]: ax.set_visible(False)
        plt.suptitle(f"Gene Distributions  t={snap['t']}  n={snap['n_cells']}",
                     fontsize=10, fontweight="bold")
        plt.tight_layout(); plt.show()

    def plot_AI_ratio(self, snapshot_idx: int = -1):
        """Spatial [A]/[I] map with lysis-threshold contour (dashed white)."""
        s      = self.snapshots[snapshot_idx]
        safe_I = np.where(s["I"] > 1e-9, s["I"], 1e-9)
        ratio  = s["A"] / safe_I
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, data, title, cmap, vmax in zip(
            axes,
            [s["A"], s["I"], ratio],
            ["MMP [A]", "TIMP [I]", "[A]/[I] ratio"],
            ["Reds", "Blues", "RdYlGn_r"],
            [None, None, 4.0]
        ):
            vm = vmax or (data.max() if data.max() > 0 else 1.0)
            im = ax.imshow(data.T, origin="lower", cmap=cmap, vmin=0, vmax=vm)
            if title == "[A]/[I] ratio":
                ax.contour(ratio.T, levels=[self.p.AI_threshold],
                           colors="white", linewidths=1.5, linestyles="--")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"{title}  t={s['t']}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        plt.suptitle("MMP-TIMP Fields  (white dashed = lysis threshold [A]/[I]=2)",
                     fontsize=10, fontweight="bold")
        plt.tight_layout(); plt.show()

    # ---- animations ---------------------------------------------------------

    def animate(self, save_path: Optional[str] = None,
                fps: int = 5, interval: int = 250) -> FuncAnimation:
        """
        Creates a single-panel RGB composite animation:
        Red = Cells (rho), Green = Fibrillar ECM (w_f), Blue = Capsule (w_nf).
        """
        if not self.snapshots:
            raise RuntimeError("No snapshots to animate.")

        N = self.p.grid_size
        snaps = self.snapshots
        rho_max = max(s["rho"].max() for s in snaps) or 1.0

        fig, ax = plt.subplots(figsize=(6, 6), facecolor='black')
        # Tweak margins to make the image fill the frame
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1) 
        
        def _build_rgb(snap):
            rgb = np.zeros((N, N, 3))
            # RED: Cells (normalized so max density is bright red)
            rgb[:, :, 0] = np.clip(snap["rho"] / rho_max, 0, 1)
            # GREEN: Fibrillar ECM (Stroma)
            rgb[:, :, 1] = np.clip(snap["w_f"], 0, 1)
            # BLUE: Non-fibrillar ECM (Capsule)
            rgb[:, :, 2] = np.clip(snap["w_nf"], 0, 1)
            return rgb

        im = ax.imshow(_build_rgb(snaps[0]).transpose((1, 0, 2)), origin="lower")
        ax.axis('off')  \
        
   
        time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, 
                            color="white", fontsize=11, fontweight="bold", 
                            va="top", ha="left", bbox=dict(facecolor='black', alpha=0.6))

        def _update(frame_idx):
            snap = snaps[frame_idx]
            im.set_data(_build_rgb(snap).transpose((1, 0, 2)))
            time_text.set_text(f"t = {snap['t']} | cells = {snap['n_cells']}")
            return im, time_text

        anim = FuncAnimation(fig, _update, frames=len(snaps),
                             interval=interval, blit=True, repeat=True)

        if save_path:
            print(f"Saving animation to {save_path} ...")
            writer = FFMpegWriter(fps=fps) if save_path.endswith(".mp4") \
                     else PillowWriter(fps=fps)
            anim.save(save_path, writer=writer)
            plt.close(fig)
            print("Saved.")
        else:
            plt.show()

        return anim

    def animate_gene_expression(self, save_path: Optional[str] = None,
                                fps: int = 4, interval: int = 250) -> FuncAnimation:
        """Animate spatial maps of all genes side-by-side."""
        N = self.p.grid_size; snaps = self.snapshots
        ncols = min(N_GENES, 4); nrows = (N_GENES + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.5*ncols, 3.5*nrows))
        axes_flat = list(np.array(axes).flat)

        def _gmap(snap, gi):
            g = np.full((N,N), np.nan); cnt = np.zeros((N,N))
            for (x,y,h) in snap["cells"]:
                g[x,y] = (np.nan_to_num(g[x,y])*cnt[x,y]+h[gi])/(cnt[x,y]+1)
                cnt[x,y] += 1
            return g

        images = []
        for i, (ax, role) in enumerate(zip(axes_flat, GENE_ROLES.values())):
            im = ax.imshow(_gmap(snaps[0], i).T, origin="lower",
                           cmap="RdYlBu_r", vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
            ax.set_title(role["name"], fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            images.append(im)
        for ax in axes_flat[N_GENES:]: ax.set_visible(False)
        tt = fig.text(0.5, 0.97, "", ha="center", fontsize=10, fontweight="bold")
        plt.tight_layout(rect=[0,0,1,0.93])

        def _update(fi):
            s = snaps[fi]
            for i, im in enumerate(images): im.set_data(_gmap(s, i).T)
            tt.set_text(f"Gene Expression  t={s['t']}  n={s['n_cells']}")
            return images + [tt]

        anim = FuncAnimation(fig, _update, frames=len(snaps),
                             interval=interval, blit=True, repeat=True)
        if save_path:
            print(f"Saving -> {save_path}")
            writer = FFMpegWriter(fps=fps) if save_path.endswith(".mp4") \
                     else PillowWriter(fps=fps)
            anim.save(save_path, writer=writer); plt.close(fig)
        else:
            plt.show()
        return anim

    # ---- transcriptomics export ---------------------------------------------

    def get_transcriptomic_dataframe(self) -> pd.DataFrame:
        if not self.transcriptome_log:
            print("No data -- run first."); return pd.DataFrame()
        return pd.DataFrame(self.transcriptome_log)

    def get_final_expression_matrix(self) -> pd.DataFrame:
        df = self.get_transcriptomic_dataframe()
        if df.empty: return df
        gene_cols = [GENE_ROLES[i]["name"] for i in range(N_GENES)]
        final_t   = df["timestep"].max()
        return df[df["timestep"] == final_t].set_index("cell_id")[["x","y"]+gene_cols]


# PHENOTYPE COMPARISON UTILITIES

def run_phenotype_comparison(
        n_steps: int = 400,
        grid_size: int = 80,
        save_every: int = 40,
        seed: int = 42,
        phenotypes: Optional[List[str]] = None
) -> Dict[str, TumorSimulation]:
    """
    Run all phenotype regimes and return completed simulation objects.
    Replicates the paper's Fig 1F / Fig 2D experimental design.

    Usage
    -----
    results = run_phenotype_comparison()
    plot_invasion_comparison(results)
    plot_spatial_comparison(results)
    """
    if phenotypes is None:
        phenotypes = ["carcinoma_in_situ", "apolar_cluster", "multiscale_invasion"]
    results = {}
    for name in phenotypes:
        print(f"\n{'='*50}\n  {name.upper()}\n{'='*50}")
        p = SimulationParams(grid_size=grid_size, n_steps=n_steps,
                             save_every=save_every, seed=seed)
        sim = TumorSimulation.from_phenotype(name, params=p)
        sim.run(verbose=True)
        results[name] = sim
    return results


def animate_phenotype_scatter(
        sim: "TumorSimulation",
        save_path: Optional[str] = None,
        fps: int = 4,
        interval: int = 250,
        color_by: str = "phenotype"   # "phenotype" | "dist" (distance from centre)
) -> FuncAnimation:
    """
    Animate each cell as a point in 2-D phenotype space:
        X axis = Adhesion level      mean(g6, g7)
        Y axis = Invasion potential  mean(g0, g1, g2)
 
    Points are coloured by their current phenotype regime classification
    (or by distance from tumour centre if color_by="dist").
    Regime boundary lines are drawn as fixed dashed lines.
    A marginal histogram bar on each axis shows the population distribution.
 
    This reveals the continuous drift of the population through phenotype
    space driven by mutation and selection -- the key advantage of this
    model over the paper's discrete CPM parameter sweeps.
 
    Parameters
    ----------
    sim        : completed TumorSimulation
    save_path  : e.g. "phenotype_scatter.mp4" or "phenotype_scatter.gif"
    fps        : frames per second when saving
    interval   : ms between frames for Jupyter display
    color_by   : "phenotype" colours by regime; "dist" colours by spatial
                 distance from tumour centre (shows front vs core separation)
 
    Jupyter usage
    -------------
        anim = animate_phenotype_scatter(sim)
        from IPython.display import HTML; HTML(anim.to_jshtml())
    """
    snaps = sim.snapshots
    N     = sim.p.grid_size
    cx, cy = N // 2, N // 2
 
    # Pre-compute per-snapshot point clouds

    def _snap_points(snap):
        pts = []
        for (x, y, h) in snap["cells"]:
            adh = (h[6] + h[7]) / 2.0
            inv = (h[0] + h[1] + h[2]) / 3.0
            ph  = _classify_cell(h)
            dist = np.sqrt((x - cx)**2 + (y - cy)**2) / (N / 2.0)   # normalised
            pts.append((adh, inv, ph, dist))
        return pts
 
    all_pts = [_snap_points(s) for s in snaps]
 
    # ── Figure layout: main scatter + two marginal histograms ────────────────
    fig = plt.figure(figsize=(8, 7))
    gs  = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                           hspace=0.05, wspace=0.05)
    ax_main  = fig.add_subplot(gs[1, 0])
    ax_top   = fig.add_subplot(gs[0, 0], sharex=ax_main)   # adhesion histogram
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)   # invasion histogram
 
    ax_main.set_xlim(-0.02, 1.02)
    ax_main.set_ylim(-0.02, 1.02)
    ax_main.set_xlabel("Adhesion  mean(g6, g7)", fontsize=10)
    ax_main.set_ylabel("Invasion potential  mean(g0, g1, g2)", fontsize=10)
    plt.setp(ax_top.get_xticklabels(),   visible=False)
    plt.setp(ax_right.get_yticklabels(), visible=False)
    
    # Replace the shading + corner text block with:
    ax_main.axvline(0.5, color="gray", lw=1.0, ls="--", alpha=0.4)
    ax_main.axhline(0.5, color="gray", lw=1.0, ls="--", alpha=0.4)
    ax_main.text(0.02, 0.97, "← low adhesion", fontsize=7, color="gray",
                transform=ax_main.transAxes, va="top")
    ax_main.text(0.98, 0.97, "high adhesion →", fontsize=7, color="gray",
                transform=ax_main.transAxes, va="top", ha="right")
    ax_main.text(0.02, 0.02, "low invasion ↓", fontsize=7, color="gray",
                transform=ax_main.transAxes, va="bottom")
    ax_main.text(0.02, 0.52, "high invasion ↑", fontsize=7, color="gray",
                transform=ax_main.transAxes, va="bottom")
    # Legend
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor=color,
            markersize=7, label=label)
        for label, color in CELL_TYPE_COLORS.items()
    ]
    ax_main.legend(handles=legend_handles, fontsize=8, loc="upper left",
                   framealpha=0.85)
 
    # Scatter collections -- one per regime
    scatter_objs = {}
    for label, color in CELL_TYPE_COLORS.items():
        sc = ax_main.scatter([], [], c=color, marker="o",
                            s=12, alpha=0.55, linewidths=0, zorder=3)
        scatter_objs[label] = sc
    # Centroid marker
    centroid_sc = ax_main.scatter([], [], c="black", marker="+", s=120,
                                  linewidths=2, zorder=5, label="Population centroid")
 
    # Time / stats text
    tt = ax_main.text(0.98, 0.03, "", transform=ax_main.transAxes,
                      ha="right", va="bottom", fontsize=8,
                      bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
 
    # Colour-by-distance colormap
    cmap_dist = plt.cm.plasma
 
    n_bins = 25
 
    def _update(fi):
        pts = all_pts[fi]
        snap = snaps[fi]
 
        # Clear marginals
        ax_top.cla();   ax_top.set_xlim(-0.02, 1.02)
        ax_right.cla(); ax_right.set_ylim(-0.02, 1.02)
        plt.setp(ax_top.get_xticklabels(),   visible=False)
        plt.setp(ax_right.get_yticklabels(), visible=False)
 
        if not pts:
            for sc in scatter_objs.values(): sc.set_offsets(np.empty((0,2)))
            centroid_sc.set_offsets(np.empty((0,2)))
            tt.set_text(f"t={snap['t']}  n=0")
            return list(scatter_objs.values()) + [centroid_sc, tt]
 
        adhs  = np.array([p[0] for p in pts])
        invs  = np.array([p[1] for p in pts])
        dists = np.array([p[3] for p in pts])
        labels = [p[2] for p in pts]
 
        if color_by == "phenotype":
            for regime, sc in scatter_objs.items():
                mask = np.array([l == regime for l in labels])
                if mask.any():
                    sc.set_offsets(np.column_stack([adhs[mask], invs[mask]]))
                else:
                    sc.set_offsets(np.empty((0, 2)))
        else:
            # All points in one scatter coloured by distance
            for sc in scatter_objs.values():
                sc.set_offsets(np.empty((0, 2)))
            # Reuse the first scatter object for all points
            list(scatter_objs.values())[0].set_offsets(
                np.column_stack([adhs, invs]))
            list(scatter_objs.values())[0].set_array(dists)
            list(scatter_objs.values())[0].set_cmap(cmap_dist)
            list(scatter_objs.values())[0].set_clim(0, 1)
 
        # Centroid
        centroid_sc.set_offsets([[adhs.mean(), invs.mean()]])
 
        # Marginal histograms
     
        ax_top.hist(adhs,  bins=n_bins, range=(0,1), color="#4C72B0", alpha=0.6)
        ax_right.hist(invs, bins=n_bins, range=(0,1), color="#C44E52", alpha=0.6,
                      orientation="horizontal")
        ax_top.set_ylabel("n", fontsize=7)
        ax_right.set_xlabel("n", fontsize=7)
 
        # Regime breakdown
        counts = {label: sum(1 for l in labels if l == label)
                for label in CELL_TYPE_COLORS}
        count_str = "  ".join(f"{k.split('/')[0].strip()}={v}"
                            for k, v in counts.items() if v > 0)
        tt.set_text(f"t={snap['t']}  n={len(pts)}\n{count_str}")
 
        return list(scatter_objs.values()) + [centroid_sc, tt]
 
    fig.suptitle("Phenotype Space Evolution (single simulation)",
                 fontsize=11, fontweight="bold")
 
    anim = FuncAnimation(fig, _update, frames=len(snaps),
                         interval=interval, blit=False, repeat=True)
    if save_path:
        print(f"Saving -> {save_path}")
        writer = FFMpegWriter(fps=fps) if save_path.endswith(".mp4") \
                 else PillowWriter(fps=fps)
        anim.save(save_path, writer=writer); plt.close(fig); print("Saved.")
    else:
        plt.show()
    return anim
 
 
# =============================================================================
# VARIATION 2 — Spatial animation with cells coloured by phenotype regime
# =============================================================================
 
def animate_spatial_phenotype(
        sim: "TumorSimulation",
        save_path: Optional[str] = None,
        fps: int = 4,
        interval: int = 250,
        show_field: str = "w_f"   # background field: "w_f" | "w_nf" | "rho" | "A"
) -> FuncAnimation:
    """
    Animate the spatial layout of cells coloured by their current phenotype
    regime, overlaid on a chosen continuum field.
 
    Each cell is drawn as a coloured dot:
        Blue   = Carcinoma in-situ   (high adhesion, low invasion)
        Orange = Apolar cluster      (low adhesion, low invasion)
        Red    = Multiscale invasion (low adhesion, high invasion)
 
    This shows how the tumour spatially segregates into subclones with
    different phenotypes as mutations accumulate -- the invasion front
    progressively turning red while the core stays blue/orange.
 
    Parameters
    ----------
    sim        : completed TumorSimulation
    save_path  : e.g. "spatial_phenotype.mp4"
    fps        : frames per second when saving
    interval   : ms between frames for Jupyter display
    show_field : continuum field shown as greyscale background
 
    Jupyter usage
    -------------
        anim = animate_spatial_phenotype(sim)
        from IPython.display import HTML; HTML(anim.to_jshtml())
    """
    snaps = sim.snapshots
    N     = sim.p.grid_size
 
    field_labels = {"w_f": "Fibrillar ECM", "w_nf": "Nonfibrillar ECM",
                    "rho": "Cell Density",   "A":    "MMP [A]"}
    field_cmaps  = {"w_f": "YlGn", "w_nf": "BuGn", "rho": "Greys", "A": "Reds"}
    assert show_field in field_labels, f"show_field must be one of {list(field_labels)}"
 
    # Fixed colour scale for background
    field_max = max(s[show_field].max() for s in snaps) or 1.0
    field_max = 1.0 if show_field in ("w_f", "w_nf") else field_max
 
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, N); ax.set_ylim(0, N)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
 
    # Background field image
    bg_im = ax.imshow(snaps[0][show_field].T, origin="lower",
                      cmap=field_cmaps[show_field],
                      vmin=0, vmax=field_max,
                      extent=[0, N, 0, N], alpha=0.75, zorder=1)
    plt.colorbar(bg_im, ax=ax, fraction=0.046, pad=0.04,
                 label=field_labels[show_field])
 
    # One scatter per regime so legend is clean
    scatter_objs = {}
    for label, color in CELL_TYPE_COLORS.items():
        sc = ax.scatter([], [], c=color, marker="o",
                        s=8, alpha=0.75, linewidths=0,
                        label=label, zorder=3)
        scatter_objs[label] = sc
 
    # Seed radius circle
    cx, cy   = N / 2, N / 2
    r_seed   = max(2, N // 14) + 2
    theta    = np.linspace(0, 2*np.pi, 200)
    ax.plot(cx + r_seed*np.cos(theta), cy + r_seed*np.sin(theta),
            "w--", lw=1.2, alpha=0.5, zorder=4, label="Seed radius")
 
    ax.legend(fontsize=8, loc="upper right", framealpha=0.85,
              markerscale=2.0)
 
    tt = ax.set_title("", fontsize=10, fontweight="bold")
    it = fig.text(0.5, 0.01, "", ha="center", fontsize=8, color="#444")
 
    def _update(fi):
        snap = snaps[fi]
 
        # Update background
        bg_im.set_data(snap[show_field].T)
 
        # Sort cells into regime buckets
        buckets: Dict[str, Tuple[List, List]] = {r: ([], []) for r in CELL_TYPE_COLORS}
        for (x, y, h) in snap["cells"]:
            regime = _classify_cell(h)
            buckets[regime][0].append(x + 0.5)   # offset 0.5 to centre in pixel
            buckets[regime][1].append(y + 0.5)
 
        for regime, sc in scatter_objs.items():
            xs, ys = buckets[regime]
            if xs:
                sc.set_offsets(np.column_stack([xs, ys]))
            else:
                sc.set_offsets(np.empty((0, 2)))
 
        # Regime counts
        counts = {label: len(buckets[label][0]) for label in CELL_TYPE_COLORS}
        total  = sum(counts.values())
        count_str = "   ".join(
            f"{k.split('/')[0].strip()}={v} ({100*v//max(total,1)}%)"
            for k, v in counts.items() if v > 0
        )
        it.set_text(count_str)
        return list(scatter_objs.values()) + [bg_im, tt, it]
 
    fig.suptitle(
        f"Spatial Phenotype Evolution  |  bg: {field_labels[show_field]}",
        fontsize=11, fontweight="bold"
    )
 
    anim = FuncAnimation(fig, _update, frames=len(snaps),
                         interval=interval, blit=False, repeat=True)
    if save_path:
        print(f"Saving -> {save_path}")
        writer = FFMpegWriter(fps=fps) if save_path.endswith(".mp4") \
                 else PillowWriter(fps=fps)
        anim.save(save_path, writer=writer); plt.close(fig); print("Saved.")
    else:
        plt.show()
    return anim
 

def plot_invasion_comparison(results: Dict[str, "TumorSimulation"]):
    """
    Bar chart of final invasion area per phenotype.
    Replicates paper Fig 1F / Fig 2D style.
    """
    names  = list(results.keys())
    areas  = [r.snapshots[-1]["invasion_area"] for r in results.values()]
    colors = ["#4C72B0","#C44E52","#55A868","#8172B2"][:len(names)]
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(range(len(names)), areas, color=colors,
                  edgecolor="black", width=0.55, alpha=0.88)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("_","\n") for n in names], fontsize=9)
    ax.set_ylabel("Area of Invasion (AU)", fontsize=10)
    ax.set_title("Invasion Area by Phenotype", fontsize=12, fontweight="bold")
    if areas:
        ax.set_ylim(0, max(areas) * 1.25)
        for bar, val in zip(bars, areas):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+max(areas)*0.02,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.show()


def plot_spatial_comparison(results: Dict[str, "TumorSimulation"],
                            snapshot_idx: int = -1):
    """Side-by-side cell density and fibrillar ECM across phenotypes."""
    names = list(results.keys()); ncols = len(names)
    fig, axes = plt.subplots(2, ncols, figsize=(4.5*ncols, 8))
    if ncols == 1: axes = axes[:, np.newaxis]
    for col, name in enumerate(names):
        sim  = results[name]
        snap = sim.snapshots[snapshot_idx]
        for row, (key, label, cmap, vmax) in enumerate([
            ("rho",  "Cell Density",  "hot_r", None),
            ("w_f",  "Fibrillar ECM", "YlGn",  1.0),
        ]):
            ax = axes[row, col]
            d  = snap[key]
            vm = vmax or (d.max() if d.max() > 0 else 1.0)
            im = ax.imshow(d.T, origin="lower", cmap=cmap, vmin=0, vmax=vm)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"{name.replace('_',' ')}\nt={snap['t']}",
                         fontsize=9, fontweight="bold")
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0: ax.set_ylabel(label, fontsize=9)
    plt.suptitle("Phenotype Comparison -- Cell Density & Fibrillar ECM",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(); plt.show()


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":

    # ---- Single phenotype run -----------------------------------------------
    params = SimulationParams(
        grid_size=80, n_steps=400, save_every=25,
        use_rd_ecm=True, n_initial_cells=20, seed=42
    )
    sim = TumorSimulation.from_phenotype("multiscale_invasion", params=params)
    sim.run()

    sim.plot_snapshots()
    sim.plot_population_dynamics()
    sim.plot_gene_distributions()
    sim.plot_AI_ratio()

    # Save animations (mp4 needs ffmpeg; gif needs Pillow)
    sim.animate(save_path="tumor_multiscale.mp4", fps=5, gene_idx=0)
    sim.animate_gene_expression(save_path="tumor_genes.mp4", fps=5)

    # ---- Phenotype comparison (replicates paper Fig 1F / Fig 2D) -----------
    print("\nRunning phenotype comparison ...")
    results = run_phenotype_comparison(n_steps=400, grid_size=80, save_every=40)
    plot_invasion_comparison(results)
    plot_spatial_comparison(results)