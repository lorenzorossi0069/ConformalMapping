"""
Conformal Mapping Visualizer
============================
Visualizza trasformazioni conformi di funzioni complesse con matplotlib.

Dipendenze:
    pip install numpy matplotlib

Uso:
    python conformal_mapping.py
"""

import matplotlib
#matplotlib.use("Qt5Agg")   # backend interattivo su Ubuntu (usa Qt5Agg se TkAgg non funziona)
matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import RadioButtons, Slider, Button
from matplotlib.lines import Line2D

# ── Funzioni complesse ──────────────────────────────────────────────────────

def f_z2(z):      return z**2
def f_z3(z):      return z**3
def f_exp(z):     return np.exp(z)
def f_log(z):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(z) < 1e-10, np.nan+0j, np.log(z))
def f_sin(z):     return np.sin(z)
def f_cos(z):     return np.cos(z)
def f_sinh(z):    return np.sinh(z)
def f_moebius(z):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(z - 1) < 1e-10, np.nan+0j, (z + 1) / (z - 1))
def f_inv(z):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(z) < 1e-10, np.nan+0j, 1.0 / z)
def f_joukowski(z):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(z) < 1e-10, np.nan+0j, z + 1.0 / z)
def f_sqrt(z):    return np.sqrt(z)
def f_ez2(z):     return np.exp(-z**2)

FUNCTIONS = {
    "z²":                f_z2,
    "z³":                f_z3,
    "exp(z)":            f_exp,
    "log(z)":            f_log,
    "sin(z)":            f_sin,
    "cos(z)":            f_cos,
    "sinh(z)":           f_sinh,
    "Möbius (z+1)/(z-1)": f_moebius,
    "1/z":               f_inv,
    "z + 1/z  (Joukowski)": f_joukowski,
    "√z":                f_sqrt,
    "exp(−z²)":          f_ez2,
}

# ── Generatori di griglia ───────────────────────────────────────────────────

N_LINES  = 14
N_POINTS = 400

def make_rect_grid(R=2.0, n=N_LINES):
    """Griglia rettangolare nel rettangolo [−R,R]×[−R,R]."""
    ts = np.linspace(-R, R, n + 1)
    pts = np.linspace(-R, R, N_POINTS)
    h_lines = [pts + 1j * t for t in ts]           # rette Im = cost
    v_lines = [t + 1j * pts for t in ts]            # rette Re = cost
    return h_lines, v_lines

def make_polar_grid(R=2.0, n=N_LINES):
    """Griglia polare: cerchi concentrici + raggi."""
    th = np.linspace(0, 2 * np.pi, N_POINTS)
    radii  = np.linspace(R / n, R, n)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rs = np.linspace(0, R, N_POINTS)
    h_lines = [r * np.exp(1j * th) for r in radii]          # cerchi
    v_lines = [rs * np.exp(1j * a) for a in angles]          # raggi
    return h_lines, v_lines

def make_strip_grid(R=2.0, n=N_LINES):
    """Striscia −π < Im(z) < π, utile per exp e sin."""
    xs = np.linspace(-R * 2, R * 2, N_POINTS)
    ys = np.linspace(-np.pi, np.pi, N_POINTS)
    h_ts = np.linspace(-np.pi, np.pi, n + 1)
    v_ts = np.linspace(-R * 2, R * 2, n + 1)
    h_lines = [xs + 1j * t for t in h_ts]
    v_lines = [t + 1j * ys for t in v_ts]
    return h_lines, v_lines

def make_disk_grid(n=N_LINES):
    """Griglia polare nel disco unitario |z| ≤ 1."""
    th = np.linspace(0, 2 * np.pi, N_POINTS)
    radii  = np.linspace(1 / n, 1.0, n)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rs = np.linspace(0, 1.0, N_POINTS)
    h_lines = [r * np.exp(1j * th) for r in radii]
    v_lines = [rs * np.exp(1j * a) for a in angles]
    return h_lines, v_lines

DOMAINS = {
    "Rettangolare":  make_rect_grid,
    "Polare":        make_polar_grid,
    "Striscia":      make_strip_grid,
    "Disco unitario": make_disk_grid,
}

# ── Colorazione per indice di linea ────────────────────────────────────────

def line_color(idx, total, is_horizontal):
    """Colore graduato per distinguere le linee."""
    cmap = plt.cm.Blues if is_horizontal else plt.cm.Oranges
    return cmap(0.35 + 0.55 * idx / max(total - 1, 1))

# ── Plot principale ─────────────────────────────────────────────────────────

MAX_W = 1e4   # soglia per filtrare valori esplosivi

def clip_curve(w):
    """Sostituisce con NaN i punti fuori range o non finiti."""
    w = w.copy()
    bad = ~np.isfinite(w) | (np.abs(w.real) > MAX_W) | (np.abs(w.imag) > MAX_W)
    w[bad] = np.nan
    return w


def draw_grid(ax, h_lines, v_lines, fn=None, title="", lw=0.9, alpha=0.8):
    """Disegna le curve (o le loro immagini) su ax."""
    ax.cla()
    ax.set_facecolor("#f9f9f7")
    ax.set_title(title, fontsize=11, pad=6)
    ax.axhline(0, color="#aaa", lw=0.5, zorder=0)
    ax.axvline(0, color="#aaa", lw=0.5, zorder=0)
    ax.set_xlabel("Re", fontsize=9)
    ax.set_ylabel("Im", fontsize=9)
    ax.tick_params(labelsize=8)

    n_h = len(h_lines)
    n_v = len(v_lines)

    for i, z in enumerate(h_lines):
        w = fn(z) if fn else z
        w = clip_curve(w)
        ax.plot(w.real, w.imag, color=line_color(i, n_h, True),
                lw=lw, alpha=alpha, solid_capstyle='round')

    for i, z in enumerate(v_lines):
        w = fn(z) if fn else z
        w = clip_curve(w)
        ax.plot(w.real, w.imag, color=line_color(i, n_v, False),
                lw=lw, alpha=alpha, solid_capstyle='round')

    ax.set_aspect('equal', adjustable='datalim')


def auto_zoom(h_lines, v_lines, fn, pad=1.15):
    """Calcola limiti automatici dal codominio (escludi outlier)."""
    vals = []
    for z in h_lines + v_lines:
        w = fn(z)
        w = w[np.isfinite(w) & (np.abs(w) < MAX_W)]
        if w.size:
            vals.append(w)
    if not vals:
        return -3, 3, -3, 3
    all_w = np.concatenate(vals)
    p_lo, p_hi = np.percentile(np.abs(all_w.real), [1, 99])
    q_lo, q_hi = np.percentile(np.abs(all_w.imag), [1, 99])
    M = max(p_hi, q_hi) * pad
    M = max(M, 0.5)
    return -M, M, -M, M


# ── App interattiva ─────────────────────────────────────────────────────────

class ConformalApp:
    def __init__(self):
        self.func_name   = "z²"
        self.domain_name = "Rettangolare"
        self.n_lines     = 12
        self.zoom_dom    = 2.0
        self.auto_zoom   = True

        self._build_ui()
        self.update(None)
        plt.show()

    # ── costruzione figura ────────────────────────────────────────────────

    def _build_ui(self):
        self.fig = plt.figure(figsize=(14, 8), facecolor="#ffffff")
        self.fig.canvas.manager.set_window_title("Conformal Mapping Visualizer")

        gs = gridspec.GridSpec(
            1, 3,
            width_ratios=[1, 1, 0.32],
            wspace=0.35,
            left=0.05, right=0.98,
            top=0.93, bottom=0.08,
        )
        self.ax_dom = self.fig.add_subplot(gs[0])
        self.ax_cod = self.fig.add_subplot(gs[1])
        ctrl_ax     = self.fig.add_subplot(gs[2])
        ctrl_ax.axis("off")

        self.fig.text(0.5, 0.97, "Conformal Mapping Visualizer",
                      ha="center", va="top", fontsize=14, fontweight="bold", color="#222")

        # legenda colori
        legend_elems = [
            Line2D([0],[0], color=plt.cm.Blues(0.6),   lw=2, label="Re(z) = cost  /  cerchi"),
            Line2D([0],[0], color=plt.cm.Oranges(0.6), lw=2, label="Im(z) = cost  /  raggi"),
        ]
        self.fig.legend(handles=legend_elems, loc="lower center",
                        ncol=2, fontsize=8, framealpha=0.6,
                        bbox_to_anchor=(0.38, 0.0))

        # ── widget posizioni (in figura-fraction) ────────────────────────
        left = 0.72

        # Radio: funzione
        ax_rf = self.fig.add_axes([left, 0.52, 0.25, 0.43])
        self.radio_func = RadioButtons(
            ax_rf, list(FUNCTIONS.keys()), active=0,
            label_props={"fontsize": [8]*len(FUNCTIONS)},
        )
        ax_rf.set_title("f(z)", fontsize=9, pad=4)
        self.radio_func.on_clicked(self._on_func)

        # Radio: dominio
        ax_rd = self.fig.add_axes([left, 0.33, 0.25, 0.17])
        self.radio_dom = RadioButtons(
            ax_rd, list(DOMAINS.keys()), active=0,
            label_props={"fontsize": [8]*len(DOMAINS)},
        )
        ax_rd.set_title("Dominio", fontsize=9, pad=4)
        self.radio_dom.on_clicked(self._on_dom)

        # Slider: numero linee
        ax_sl = self.fig.add_axes([left, 0.24, 0.25, 0.03])
        self.sl_lines = Slider(ax_sl, "Linee", 4, 24, valinit=12, valstep=2)
        self.sl_lines.on_changed(self.update)

        # Slider: zoom dominio
        ax_sd = self.fig.add_axes([left, 0.18, 0.25, 0.03])
        self.sl_zoom = Slider(ax_sd, "Zoom dom.", 0.5, 5.0, valinit=2.0, valstep=0.1)
        self.sl_zoom.on_changed(self.update)

        # Slider: zoom codominio (0 = auto)
        ax_sc = self.fig.add_axes([left, 0.12, 0.25, 0.03])
        self.sl_zoom_cod = Slider(ax_sc, "Zoom cod.", 0.0, 8.0, valinit=0.0, valstep=0.1)
        self.sl_zoom_cod.label.set_text("Zoom cod.\n(0=auto)")
        self.sl_zoom_cod.on_changed(self.update)

        # Pulsante salva
        ax_btn = self.fig.add_axes([left, 0.04, 0.12, 0.05])
        self.btn_save = Button(ax_btn, "Salva PNG", color="#e8e8e8", hovercolor="#d0d0d0")
        self.btn_save.on_clicked(self._save)

        # Pulsante reset
        ax_rst = self.fig.add_axes([left + 0.13, 0.04, 0.12, 0.05])
        self.btn_reset = Button(ax_rst, "↺ Reset", color="#e8e8e8", hovercolor="#d0d0d0")
        self.btn_reset.on_clicked(self._reset)

    # ── callbacks ────────────────────────────────────────────────────────

    def _on_func(self, label):
        self.func_name = label
        self.update(None)

    def _on_dom(self, label):
        self.domain_name = label
        self.update(None)

    def update(self, _val):
        self.n_lines   = int(self.sl_lines.val)
        self.zoom_dom  = float(self.sl_zoom.val)
        zoom_cod_raw   = float(self.sl_zoom_cod.val)

        fn    = FUNCTIONS[self.func_name]
        dom_f = DOMAINS[self.domain_name]

        # Griglia dominio
        if self.domain_name == "Disco unitario":
            h_lines, v_lines = dom_f(self.n_lines)
        elif self.domain_name == "Polare":
            h_lines, v_lines = dom_f(self.zoom_dom, self.n_lines)
        else:
            h_lines, v_lines = dom_f(self.zoom_dom, self.n_lines)

        # Disegna dominio
        draw_grid(self.ax_dom, h_lines, v_lines,
                  fn=None, title=f"Dominio  ({self.domain_name})")
        if self.domain_name == "Disco unitario":
            self.ax_dom.set_xlim(-1.2, 1.2)
            self.ax_dom.set_ylim(-1.2, 1.2)
        else:
            lim = self.zoom_dom * 1.05
            if self.domain_name == "Striscia":
                self.ax_dom.set_xlim(-self.zoom_dom * 2.1, self.zoom_dom * 2.1)
                self.ax_dom.set_ylim(-np.pi * 1.1, np.pi * 1.1)
            else:
                self.ax_dom.set_xlim(-lim, lim)
                self.ax_dom.set_ylim(-lim, lim)

        # Disegna codominio
        draw_grid(self.ax_cod, h_lines, v_lines,
                  fn=fn, title=f"w = {self.func_name}")

        # Zoom codominio
        if zoom_cod_raw < 0.15:   # auto
            x0, x1, y0, y1 = auto_zoom(h_lines, v_lines, fn)
            self.ax_cod.set_xlim(x0, x1)
            self.ax_cod.set_ylim(y0, y1)
        else:
            self.ax_cod.set_xlim(-zoom_cod_raw, zoom_cod_raw)
            self.ax_cod.set_ylim(-zoom_cod_raw, zoom_cod_raw)

        self.fig.canvas.draw_idle()

    def _save(self, _ev):
        fname = f"conformal_{self.func_name.replace(' ', '_').replace('(','').replace(')','')}.png"
        self.fig.savefig(fname, dpi=180, bbox_inches="tight")
        print(f"[✓] Salvato: {fname}")

    def _reset(self, _ev):
        self.sl_lines.reset()
        self.sl_zoom.reset()
        self.sl_zoom_cod.reset()


# ── Export statico (batch) ──────────────────────────────────────────────────

def export_all(output_dir=".", dpi=150):
    """
    Genera e salva un'immagine per ogni funzione disponibile
    con griglia rettangolare e zoom automatico.
    Utile per report o presentazioni.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    for name, fn in FUNCTIONS.items():
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f"f(z) = {name}", fontsize=13, fontweight="bold")

        h_lines, v_lines = make_rect_grid(R=2.0, n=12)

        draw_grid(axes[0], h_lines, v_lines, fn=None, title="Dominio z")
        axes[0].set_xlim(-2.1, 2.1); axes[0].set_ylim(-2.1, 2.1)

        draw_grid(axes[1], h_lines, v_lines, fn=fn, title=f"w = {name}")
        x0, x1, y0, y1 = auto_zoom(h_lines, v_lines, fn)
        axes[1].set_xlim(x0, x1); axes[1].set_ylim(y0, y1)

        safe = name.replace("/", "div").replace(" ", "_").replace("(","").replace(")","")
        path = os.path.join(output_dir, f"conformal_{safe}.png")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  Salvato: {path}")

    print("Export completato.")


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if "--export" in sys.argv:
        # Modalità batch: python conformal_mapping.py --export [cartella]
        out = sys.argv[sys.argv.index("--export") + 1] if "--export" in sys.argv[:-1] else "export"
        print(f"Esportazione batch in '{out}/'...")
        export_all(output_dir=out)
    else:
        # Modalità interattiva
        ConformalApp()
