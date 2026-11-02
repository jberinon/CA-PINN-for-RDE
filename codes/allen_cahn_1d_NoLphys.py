"""
Physics-Informed Neural Network (PINN) for parameter identification
of the 1D Allen-Cahn equation:

    u_t = D * u_xx + r * u * (1 - u^2)

with energy-based auxiliary constraints to separately identify D and r.

Why energy constraints?
-----------------------
The Allen-Cahn equation has a Lyapunov energy functional:
    E[u] = integral( 0.5*D*u_x^2 + 0.25*r*(1-u^2)^2 ) dx

For the equilibrium tanh profile u* = tanh(x / delta) with delta = sqrt(2D/r):
    W  := mean(u_x^2)   = C1 sqrt(r / (2D)) with C1 = 2/3         (interface sharpness)
    E  := mean(0.5*D*u_x^2 + 0.25*r*(1-u^2)^2)   =  C2 sqrt(D*r / 2) with C2 = 2sqrt(2)/3 

This gives:  C1 E / (C2 W)  = D   and   E * W /(C1 C2) ~ r
leading to the auxiliary constraints:
    L_D = ( D - C1 E/(C2 W) )^2
    L_r = ( r - E*W /(C1 C2) )^2

These constraints are computed at the current network prediction u_theta
and add independent signal that distinguishes D from r, breaking the
degeneracy that would arise from the PDE residual alone.

Kink solution 
---------------------------
    u_exact(x,t) = tanh((x-ct)/sqrt(2D/r))

satisfying Dirichlet BCs u=-1 at x=0, and u=1 at x=L.

Outputs
-------
* allen_cahn_1d_results.pdf
* allen_cahn_1d_history.csv
* allen_cahn_1d_summary.csv   (appended each run)

Usage
-----
python allen_cahn_1d_NoLphys.py --w_phys 0
"""

import argparse
import csv
import datetime
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Exact solution
# ──────────────────────────────────────────────────────────────────────────────

def exact_u(x, t, D, r):
    """Traveling kink solution of Allen-Cahn (numpy)."""
    c  = 3.0 * math.sqrt(D * r / 2.0)   # wave speed for Allen-Cahn kink
    xi = (x - c * t) / math.sqrt(2.0 * D / r)
    return np.tanh(xi)


def kink_center(t, D, r):
    """Return the x-position of the kink (where u=0) at time t."""
    c = 3.0 * math.sqrt(D * r / 2.0)
    return c * t


def kink_width(D, r):
    """Characteristic half-width delta = sqrt(2D/r) of the kink."""
    return math.sqrt(2.0 * D / r)



# ──────────────────────────────────────────────────────────────────────────────
# 2.  Oversampling helpers
# ──────────────────────────────────────────────────────────────────────────────

def sample_collocation(n_col, n_kink, x_min, x_max, t_min, t_max,
                       D, r, rng):
    """
    Build the initial collocation set:
      * n_col   uniform random points in [x_min,x_max] x [t_min,t_max]
      * n_kink  points concentrated near the moving kink front

    For the kink oversampling we:
      1. Draw t uniformly.
      2. Place x as  x_center(t) + N(0, sigma_x)  with sigma_x = 3*delta,
         clipped to the domain.  This puts ~99% of extra points inside
         [-3*delta, +3*delta] around the front where |u_x| is large.

    Returns xc, tc  as float32 arrays of shape (n_col+n_kink, 1).
    """
    delta = kink_width(D, r)

    # Uniform base set
    x_uni = rng.uniform(x_min, x_max, (n_col, 1)).astype(np.float32)
    t_uni = rng.uniform(t_min, t_max, (n_col, 1)).astype(np.float32)

    if n_kink > 0:
        t_kink  = rng.uniform(t_min, t_max, (n_kink, 1)).astype(np.float32)
        x_ctr   = kink_center(t_kink, D, r)
        x_kink  = (x_ctr + rng.normal(0.0, 3.0 * delta,
                                       (n_kink, 1))).astype(np.float32)
        x_kink  = np.clip(x_kink, x_min, x_max)
        xc = np.vstack([x_uni, x_kink])
        tc = np.vstack([t_uni, t_kink])
    else: 
        xc, tc = x_uni, t_uni

    return xc, tc


def resample_collocation(model, n_col, n_kink, x_min, x_max, t_min, t_max,
                         rng, device, D,r, n_candidates=20000,
                         sigma_factor=2.0):
    """ 
    Adaptive resampling with Gaussian spreading around gradient peaks.

    Strategy:
      1. Draw n_candidates candidate points uniformly.
      2. Evaluate |u_x| at each candidate (no-grad). 
      3. Pick the top-n_kink candidates by |u_x| as *seed* locations.
      4. For each seed, draw a new point from:
             x_new ~ N(x_seed, (sigma_factor * epsilon)^2)   clipped to [x_min, x_max]
             t_new ~ Uniform(t_min, t_max)                   (front can be anywhere in time)
      5. Combine with a fresh uniform base set of size n_col.

    The Gaussian std sigma_factor * epsilon lets you control how tightly
    points hug the interface:
      - sigma_factor ~ 1-2  : points stay within the transition layer
      - sigma_factor ~ 3-5  : wider spread, safer when epsilon is tiny
    """
    model.eval()

    epsilon = kink_width(D, r) 

    # ── 1. Candidate pool ──────────────────────────────────────────────────
    x_cand = rng.uniform(x_min, x_max, (n_candidates, 1)).astype(np.float32)
    t_cand = rng.uniform(t_min, t_max, (n_candidates, 1)).astype(np.float32)

    xT = torch.tensor(x_cand, device=device, requires_grad=True)
    tT = torch.tensor(t_cand, device=device)

    with torch.enable_grad():
        u = model(xT, tT)
        u_x = torch.autograd.grad(u, xT,
                                  grad_outputs=torch.ones_like(u),
                                  create_graph=False)[0] 
    grad_mag = u_x.detach().abs().cpu().numpy().ravel()

    # ── 2. Seeds: top-n_kink locations by |u_x| ───────────────────────────
    top_idx = np.argpartition(grad_mag, -n_kink)[-n_kink:]
    x_seeds = x_cand[top_idx]          # shape (n_kink, 1)

    # ── 3. Gaussian perturbation around each seed ─────────────────────────
    sigma = sigma_factor * epsilon
    x_noise = rng.normal(0.0, sigma, x_seeds.shape).astype(np.float32)
    x_kink  = np.clip(x_seeds + x_noise, x_min, x_max)

    # t is sampled fresh: the interface can be at any time slice
    t_kink = rng.uniform(t_min, t_max, (n_kink, 1)).astype(np.float32)

    # ── 4. Base uniform set ───────────────────────────────────────────────
    x_uni = rng.uniform(x_min, x_max, (n_col, 1)).astype(np.float32)
    t_uni = rng.uniform(t_min, t_max, (n_col, 1)).astype(np.float32)

    xc = np.vstack([x_uni, x_kink])
    tc = np.vstack([t_uni, t_kink])

    model.train()
    return xc, tc



# ──────────────────────────────────────────────────────────────────────────────
# 3.  Neural network
# ──────────────────────────────────────────────────────────────────────────────

class PINN(nn.Module):
    """
    Fully-connected tanh network  (x, t) -> u.
    Output uses tanh so u in (-1, 1) — consistent with Allen-Cahn range.
    Trainable D = exp(log_D) > 0,  r = exp(log_r) > 0.
    """

    def __init__(self, n_layers, n_neurons, D_init, r_init):
        super().__init__()
        layers = [nn.Linear(2, n_neurons), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers += [nn.Linear(n_neurons, 1), nn.Tanh()]   # output in (-1,1)
        self.net   = nn.Sequential(*layers)
        self.log_D = nn.Parameter(torch.tensor(math.log(D_init)))
        self.log_r = nn.Parameter(torch.tensor(math.log(r_init)))

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    @property
    def D(self): return torch.exp(self.log_D)

    @property
    def r(self): return torch.exp(self.log_r)

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))


# ──────────────────────────────────────────────────────────────────────────────
# 4.  PDE residual  u_t - D*u_xx - r*u*(1-u^2)  = 0
# ──────────────────────────────────────────────────────────────────────────────

def pde_and_gradients(model, x_col, t_col):
    """
    Returns (residual, u, u_x) at collocation points.
    u_x is reused for the energy/width constraints.
    """
    x_col = x_col.requires_grad_(True)
    t_col = t_col.requires_grad_(True)

    u    = model(x_col, t_col)
    ones = torch.ones_like(u)

    u_t  = torch.autograd.grad(u,   t_col, grad_outputs=ones, create_graph=True)[0]
    u_x  = torch.autograd.grad(u,   x_col, grad_outputs=ones, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_col,
                               grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0]
    return u_t, u, u_x, u_xx


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Energy and width functionals
# ──────────────────────────────────────────────────────────────────────────────

def energy_width(model, u, u_x):
    """
    E = mean( 0.5*D*u_x^2  +  0.25*r*(1 - u^2)^2 )
    W = mean( u_x^2 )

    At equilibrium:  E/W  -> D  and  E*W  -> r  (up to O(1) constants).
    """
    #D, r = model.D, model.r
    #E = torch.mean(0.5 * D * u_x ** 2 + 0.25 * r * (1.0 - u ** 2) ** 2)
    #W = torch.mean(u_x ** 2)

    # constant for the kink
    E = 2./3.*np.sqrt(2*args.D_true * args.r_true)
    W = 2./3.*np.sqrt(2*args.r_true / args.D_true)

    return E, W


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Full loss
# ──────────────────────────────────────────────────────────────────────────────

def compute_loss(model, tensors, weights):
    """
    Returns (total, loss_pde, loss_bc, loss_data, loss_phys_D, loss_phys_r).
    """ 
    mse = nn.MSELoss()
    w_pde, w_bc, w_data, w_phys = weights

    # ── PDE residual ─────────────────────────────────────────────────────────
    u_t, u_col, u_x, u_xx = pde_and_gradients(
        model, tensors["x_col"], tensors["t_col"])
    pde_res  = u_t - model.D * u_xx - model.r * u_col * (1.0 - u_col ** 2) 
    loss_pde = mse(pde_res, torch.zeros_like(pde_res))

    # ── Energy / width constraints ────────────────────────────────────────────
    E, W = energy_width(model, u_col, u_x)
    # guard against W ~ 0 early in training
    W_safe     = W + 1e-8
    #C1 = 2./3. ;  C2 = 2.*np.sqrt(2)/3. 
    #loss_phys_D = (model.D - C1*E / (C2*W_safe)) ** 2 
    #loss_phys_r = (model.r - E * W_safe/(C1*C2)) ** 2

    loss_phys_D = (model.D - E / W_safe) ** 2 
    loss_phys_r = (model.r - (9/8)* E * W_safe) ** 2  
 
    # ── Boundary / initial conditions ─────────────────────────────────────────
    loss_bc = (
        mse(model(tensors["x_ic"],  tensors["t_ic"]),  tensors["u_ic"])  +
        mse(model(tensors["x_lb"],  tensors["t_lb"]),  tensors["u_lb"])  +
        mse(model(tensors["x_rb"],  tensors["t_rb"]),  tensors["u_rb"])
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    loss_data = mse(
        model(tensors["x_dat"], tensors["t_dat"]),
        tensors["u_dat"]
    )
 
    loss = (w_pde  * loss_pde
          + w_bc   * loss_bc
          + w_data * loss_data
          #+ w_phys * (loss_phys_D + loss_phys_r)
          )

    return loss, loss_pde, loss_bc, loss_data, loss_phys_D, loss_phys_r


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Training
# ──────────────────────────────────────────────────────────────────────────────

def train(args, device):
    D_true, r_true = args.D_true, args.r_true

    print(f"\nTrue parameters:  D = {D_true:.4f},  r = {r_true:.4f}")
    print(f"Kink width delta = {kink_width(D_true, r_true):.4f}")
    print(f"Oversampling: n_col={args.n_col}  n_kink={args.n_kink}  "
          f"(total collocation = {args.n_col + args.n_kink})  "
          f"resample_every={args.resample_every}")

    x_min, x_max = args.x_min, args.x_max
    t_min, t_max = args.t_min, args.t_max
    rng = np.random.default_rng(args.seed)

    def rnd(n, lo, hi):
        return rng.uniform(lo, hi, (n, 1)).astype(np.float32)

    # ── Initial collocation set (uniform base + kink-focused oversampling) ────
    xc, tc = sample_collocation(
        args.n_col, args.n_kink,
        x_min, x_max, t_min, t_max,
        D_true, r_true, rng,
    )

    # IC  t = t_min
    x_ic = rnd(args.n_bc, x_min, x_max)
    t_ic = np.full((args.n_bc, 1), t_min, dtype=np.float32)
    u_ic = exact_u(x_ic, t_ic, D_true, r_true).astype(np.float32)

    # Left BC  x = x_min
    x_lb = np.full((args.n_bc, 1), x_min, dtype=np.float32)
    t_lb = rnd(args.n_bc, t_min, t_max)
    u_lb = exact_u(x_lb, t_lb, D_true, r_true).astype(np.float32)

    # Right BC  x = x_max
    x_rb = np.full((args.n_bc, 1), x_max, dtype=np.float32)
    t_rb = rnd(args.n_bc, t_min, t_max)
    u_rb = exact_u(x_rb, t_rb, D_true, r_true).astype(np.float32)

    # Data
    x_dat = rnd(args.n_data, x_min, x_max)
    t_dat = rnd(args.n_data, t_min, t_max)
    u_dat = exact_u(x_dat, t_dat, D_true, r_true).astype(np.float32)
    if args.noise > 0.0:
        u_dat += (args.noise * rng.standard_normal(u_dat.shape)).astype(np.float32)

    def T(a): return torch.tensor(a, device=device)

    def make_tensors(xc_np, tc_np):
        """Build the collocation portion of the tensors dict."""
        return dict(
            x_col=T(xc_np), t_col=T(tc_np),
            x_ic=T(x_ic),   t_ic=T(t_ic),  u_ic=T(u_ic),
            x_lb=T(x_lb),   t_lb=T(t_lb),  u_lb=T(u_lb),
            x_rb=T(x_rb),   t_rb=T(t_rb),  u_rb=T(u_rb),
            x_dat=T(x_dat), t_dat=T(t_dat), u_dat=T(u_dat),
        )

    tensors = make_tensors(xc, tc)
    # Keep a copy of the initial collocation for the figure
    xc_init, tc_init = xc.copy(), tc.copy()
    weights = (args.w_pde, args.w_bc, args.w_data, args.w_phys)

    model = PINN(args.layers, args.neurons, args.D_init, args.r_init).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Network:  {args.layers} layers x {args.neurons} neurons  "
          f"({n_params:,} params)")
    print(f"Init D={args.D_init:.4f}, r={args.r_init:.4f}  |  "
          f"n_col={args.n_col}+{args.n_kink}(kink), n_bc={args.n_bc}, "
          f"n_data={args.n_data}")
    print(f"Weights  w_pde={args.w_pde}, w_bc={args.w_bc}, "
          f"w_data={args.w_data}, w_phys={args.w_phys}")
    

    keys = ["loss", "loss_pde", "loss_bc", "loss_data",
            "loss_phys_D", "loss_phys_r", "D", "r"]
    history = {k: [] for k in keys}
    log_every = max(1, args.adam_epochs // 10)

    def record(vals):
        loss, lp, lb, ld, lpD, lpr = vals
        with torch.no_grad():
            history["loss"].append(loss.item())
            history["loss_pde"].append(lp.item())
            history["loss_bc"].append(lb.item())
            history["loss_data"].append(ld.item())
            history["loss_phys_D"].append(lpD.item())
            history["loss_phys_r"].append(lpr.item())
            history["D"].append(model.D.item())
            history["r"].append(model.r.item())

    def log_step(label, step, epochs, vals):
        loss, lp, lb, ld, lpD, lpr = vals
        #print(f"  [{label} {step:6d}/{args.adam_epochs}]  "
        print(f"  [{label} {step:6d}/{epochs}]  "
              f"loss={loss.item():.3e}  pde={lp.item():.3e}  "
              f"bc={lb.item():.3e}  data={ld.item():.3e}  "
              f"phys_D={lpD.item():.3e}  phys_r={lpr.item():.3e}  "
              f"D={model.D.item():.5f}  r={model.r.item():.5f}  "
              f"({time.time()-t0:.1f}s)")


    t0 = time.time()

    # ---- Phase 1: Adam ------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.adam_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.adam_epochs, eta_min=1e-5)

    for epoch in range(1, args.adam_epochs + 1):  
        # ── Adaptive resampling ───────────────────────────────────────────
        if (args.resample_every > 0
                and args.n_kink > 0
                and epoch > 1
                and epoch % args.resample_every == 0):
            xc, tc = resample_collocation(
                model, args.n_col, args.n_kink,
                x_min, x_max, t_min, t_max, rng, device,D_true,r_true)
            tensors = make_tensors(xc, tc)
            print(f"    [resample @ epoch {epoch}]")

        model.train()
        optimizer.zero_grad()
        vals = compute_loss(model, tensors, weights)
        vals[0].backward()
        optimizer.step()
        scheduler.step()
        record(vals)
        if epoch % log_every == 0 or epoch == 1:
            log_step("Adam", epoch, args.adam_epochs, vals)  

    
     # ---- Phase 2: L-BFGS fine-tuning of the network weights -----------
    if args.lbfgs_epochs > 0:
    
        def create_lbfgs_optimizer():
            return torch.optim.LBFGS(
                model.parameters(),
                lr             = args.lbfgs_lr,
                max_iter       = args.lbfgs_max_iter,
                history_size   = args.lbfgs_history_size,
                line_search_fn = "strong_wolfe",
            )

        optimizer = create_lbfgs_optimizer()
        _cache = {}

        def closure():
            optimizer.zero_grad()
            vals = compute_loss(model, tensors, weights)
            vals[0].backward()
            _cache["vals"] = vals
            return vals[0]

        for step in range(1, args.lbfgs_epochs + 1):
            # ── Adaptive resampling ───────────────────────────────────────────
            if (args.resample_every > 0
                    and args.n_kink > 0
                    and step > 1
                    and step % args.resample_every == 0):
                xc, tc = resample_collocation(
                    model, args.n_col, args.n_kink, 
                    x_min, x_max, t_min, t_max, rng, device, D_true, r_true)
                tensors = make_tensors(xc, tc)
                optimizer = create_lbfgs_optimizer()
                print(f"    [resample & reset optimizer @ step {step}]")

            model.train()
            optimizer.step(closure)
            record(_cache["vals"]) 
            if step == 1 or step == args.lbfgs_epochs:
                log_step("L-BFGS", step, args.lbfgs_epochs, _cache["vals"])
            
    
    train_time = time.time() - t0
    D_final, r_final = model.D.item(), model.r.item()
    print(f"\nIdentified:  D = {D_final:.6f}  (true {D_true:.6f})"
          f"  rel_err = {abs(D_final-D_true)/D_true*100:.3f}%")
    print(f"             r = {r_final:.6f}  (true {r_true:.6f})"
          f"  rel_err = {abs(r_final-r_true)/r_true*100:.3f}%")
    print(f"Wall time: {train_time:.1f}s")
    # xc/tc are the FINAL collocation set (after any resampling steps)
    return model, history, train_time, xc_init, tc_init, xc, tc


# ──────────────────────────────────────────────────────────────────────────────
# 8.  CSV export
# ──────────────────────────────────────────────────────────────────────────────

def save_csv(args, history, train_time, D_final, r_final,
             summary_path="1D_allen_cahn_summary.csv",
             history_path="1D_allen_cahn_history.csv"):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    D_true, r_true = args.D_true, args.r_true

    summary_fields = [
        "timestamp", 
        "x_min", "x_max", "t_min", "t_max",
        "layers", "neurons", "n_trainable_params",
        "n_col", "n_kink", "n_bc", "n_data", "noise",
        "resample_every",
        "w_pde", "w_bc", "w_data", "w_phys",
        "adam_epochs", "adam_lr", "lbfgs_epochs", "lbfgs_lr",
        "lbfgs_max_iter", "lbfgs_history_size", "seed",
        "D_true", "r_true", "D_init", "r_init",
        "D_final", "r_final", "D_rel_err_pct", "r_rel_err_pct",
        "final_loss", "final_loss_pde", "final_loss_bc", "final_loss_data",
        "final_loss_phys_D", "final_loss_phys_r",
        "train_time_s",
    ]

    n_params = sum(p.numel() for p in
                   PINN(args.layers, args.neurons, args.D_init, args.r_init).parameters())
    row = {
        "timestamp"          : timestamp,
        "x_min"              : args.x_min,
        "x_max"              : args.x_max,
        "t_min"              : args.t_min, 
        "t_max"              : args.t_max,
        "layers"             : args.layers,
        "neurons"            : args.neurons,
        "n_trainable_params" : n_params,
        "n_col"              : args.n_col,
        "n_kink"             : args.n_kink,
        "n_bc"               : args.n_bc,
        "n_data"             : args.n_data,
        "noise"              : args.noise,
        "resample_every"     : args.resample_every,
        "w_pde"              : args.w_pde,
        "w_bc"               : args.w_bc,
        "w_data"             : args.w_data,
        "w_phys"             : args.w_phys,
        "adam_epochs"        : args.adam_epochs,
        "adam_lr"            : args.adam_lr,
        "lbfgs_epochs"       : args.lbfgs_epochs,
        "lbfgs_lr"           : args.lbfgs_lr,
        "lbfgs_max_iter"     : args.lbfgs_max_iter,
        "lbfgs_history_size" : args.lbfgs_history_size,
        "seed"               : args.seed,
        "D_true"             : D_true,
        "r_true"             : r_true,
        "D_init"             : args.D_init,
        "r_init"             : args.r_init,
        "D_final"            : round(D_final, 8),
        "r_final"            : round(r_final, 8),
        "D_rel_err_pct"      : round(abs(D_final - D_true) / D_true * 100, 4),
        "r_rel_err_pct"      : round(abs(r_final - r_true) / r_true * 100, 4),
        "final_loss"         : round(history["loss"][-1],       8),
        "final_loss_pde"     : round(history["loss_pde"][-1],   8),
        "final_loss_bc"      : round(history["loss_bc"][-1],    8),
        "final_loss_data"    : round(history["loss_data"][-1],  8),
        "final_loss_phys_D"  : round(history["loss_phys_D"][-1], 8),
        "final_loss_phys_r"  : round(history["loss_phys_r"][-1], 8),
        "train_time_s"       : round(train_time, 2),
    }

    write_header = not os.path.isfile(summary_path)
    with open(summary_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"Summary CSV  -> {summary_path}  ({len(summary_fields)} columns, appended)")

    history_fields = ["step", "loss", "loss_pde", "loss_bc", "loss_data",
                      "loss_phys_D", "loss_phys_r", "D", "r"]
    with open(history_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=history_fields)
        w.writeheader()
        for i in range(len(history["loss"])):
            w.writerow({
                "step"         : i + 1,
                "loss"         : round(history["loss"][i],        8),
                "loss_pde"     : round(history["loss_pde"][i],    8),
                "loss_bc"      : round(history["loss_bc"][i],     8),
                "loss_data"    : round(history["loss_data"][i],   8),
                "loss_phys_D"  : round(history["loss_phys_D"][i], 8),
                "loss_phys_r"  : round(history["loss_phys_r"][ i], 8),
                "D"            : round(history["D"][i],           8),
                "r"            : round(history["r"][i],           8),
            })
    print(f"History CSV  -> {history_path}  ({len(history['loss'])} rows)")


# ──────────────────────────────────────────────────────────────────────────────
# 9.  Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def plot_results(model, history, args, device,
                 xc_init, tc_init, xc_final, tc_final, filename="allen_cahn_1d_results.pdf"):
    D_true, r_true = args.D_true, args.r_true
    D_final, r_final = model.D.item(), model.r.item()
    model.eval()

    Nx, Nt = 300, 200
    xv = np.linspace(args.x_min, args.x_max, Nx, dtype=np.float32)
    tv = np.linspace(args.t_min, args.t_max, Nt, dtype=np.float32)
    XX, TT = np.meshgrid(xv, tv)

    x_flat = torch.tensor(XX.ravel()[:, None], device=device)
    t_flat = torch.tensor(TT.ravel()[:, None], device=device)
    with torch.no_grad():
        u_pred = model(x_flat, t_flat).cpu().numpy().reshape(Nt, Nx)

    u_exact = exact_u(XX, TT, D_true, r_true)
    err     = np.abs(u_pred - u_exact)
    steps   = np.arange(1, len(history["loss"]) + 1)

    fig = plt.figure(figsize=(18, 16))
    gs = GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35, top=0.93, bottom=0.05)

    # ── Row 0: space-time heatmaps ────────────────────────────────────────────
    for col, (data, title, cmap) in enumerate([
        (u_exact, "Exact  u(x,t)",               "RdBu_r"),
        (u_pred,  "PINN   u(x,t)",               "RdBu_r"),
        (err,     f"|PINN - Exact|  max={err.max():.2e}", "hot_r"),
    ]):
        ax  = fig.add_subplot(gs[0, col])
        kw  = {"vmin": -1, "vmax": 1} if col < 2 else {}
        im  = ax.pcolormesh(XX, TT, data, cmap=cmap, shading="gouraud", **kw)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=13); 
        ax.set_xlabel("x", fontsize=12); ax.set_ylabel("t", fontsize=12)

    # ── Row 1 left: collocation point scatter (initial vs final) ─────────────
    ax_pts = fig.add_subplot(gs[1, 0])
    # Thin out for readability
    stride = max(1, len(xc_init) // 2000)
    ax_pts.scatter(xc_init[::stride], tc_init[::stride],
                   s=1.5, alpha=0.4, color="steelblue", label="initial")
    if args.resample_every > 0 and args.n_kink > 0:
        ax_pts.scatter(xc_final[::stride], tc_final[::stride],
                       s=1.5, alpha=0.4, color="darkorange", label="final (resampled)")
    # Overlay the true kink trajectory
    t_traj = np.linspace(args.t_min, args.t_max, 200)
    x_traj = [kink_center(t, D_true, r_true) for t in t_traj]
    ax_pts.plot(x_traj, t_traj, "r-", lw=1.5, label="kink trajectory")
    ax_pts.set_xlim(args.x_min, args.x_max)
    ax_pts.set_ylim(args.t_min, args.t_max)
    ax_pts.set_xlabel("x", fontsize=12); ax_pts.set_ylabel("t", fontsize=12)
    ax_pts.set_title("Collocation pts (initial vs resampled)", fontsize=13)
    ax_pts.legend(fontsize=9, markerscale=5)  

    # ── Row 1 middle: |u_x| heatmap (exact) ──────────────────────────────────
    delta_true = kink_width(D_true, r_true)
    c_true     = 3.0 * math.sqrt(D_true * r_true / 2.0)
    xi_grid    = (XX - c_true * TT) / delta_true
    ux_exact   = np.abs(-1.0 / (delta_true * np.cosh(xi_grid) ** 2))
    ax_ux = fig.add_subplot(gs[1, 1])
    im2   = ax_ux.pcolormesh(XX, TT, ux_exact, cmap="YlOrRd", shading="gouraud")
    fig.colorbar(im2, ax=ax_ux, fraction=0.046, pad=0.04)
    ax_ux.set_title("|u_x| (exact)  — shows kink region", fontsize=13)
    ax_ux.set_xlabel("x", fontsize=12); ax_ux.set_ylabel("t", fontsize=12)

    # ── Row 1 right: profiles at 4 times ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 2])
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, 4))
    for i, ti in enumerate(np.linspace(args.t_min, args.t_max, 4)):
        idx = int((ti - args.t_min) / (args.t_max - args.t_min) * (Nt - 1))
        ax3.plot(xv, u_exact[idx], color=colors[i], lw=2,
                 label=f"exact t={ti:.2f}")
        ax3.plot(xv, u_pred[idx],  color=colors[i], lw=1.5, ls="--",
                 label=f"PINN  t={ti:.2f}")
    ax3.set_xlabel("x", fontsize=12); ax3.set_ylabel("u", fontsize=12)
    ax3.set_title("Profiles at selected times", fontsize=13)
    ax3.legend(fontsize=9, ncol=1, loc='upper left')

    # ── Row 2: loss history + D + r convergence ───────────────────────────────
    ax4 = fig.add_subplot(gs[2, 0])
    for key, lbl, ls in [
        ("loss",        "Total",    "-"),
        ("loss_pde",    "PDE",      "--"),
        ("loss_bc",     "BC",       "-."),
        ("loss_data",   "Data",     ":"),
        ("loss_phys_D", "Phys D",   (0,(3,1,1,1))),
        ("loss_phys_r", "Phys r",   (0,(5,1))),
    ]:
        ax4.semilogy(steps, history[key], label=lbl, ls=ls)
    ax4.set_xlabel("Epoch", fontsize=12); ax4.set_ylabel("Loss", fontsize=12)
    ax4.set_title(f"Loss history  [Adam + L-BFGS]", fontsize=13)
    ax4.legend(fontsize=9, ncol=2, loc='lower left')

    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(steps, history["D"], lw=2, color="steelblue")
    ax5.axhline(D_true, color="red", lw=1.5, ls="--", label=f"True D={D_true}")
    ax5.set_xlabel("Epoch", fontsize=12); ax5.set_ylabel("D", fontsize=12)
    ax5.set_title(f"D convergence  (final={D_final:.5f})", fontsize=13)
    ax5.legend(fontsize=10)

    ax6 = fig.add_subplot(gs[2, 2])
    ax6.plot(steps, history["r"], lw=2, color="darkorange")
    ax6.axhline(r_true, color="red", lw=1.5, ls="--", label=f"True r={r_true}")
    ax6.set_xlabel("Epoch", fontsize=12); ax6.set_ylabel("r", fontsize=12)
    ax6.set_title(f"r convergence  (final={r_final:.5f})", fontsize=13)
    ax6.legend(fontsize=10)

    # ── Row 3: relative errors ────────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[3, 0])
    rel_D = np.abs(np.array(history["D"]) - D_true) / D_true * 100
    rel_r = np.abs(np.array(history["r"]) - r_true) / r_true * 100
    ax7.semilogy(steps, rel_D, lw=2, color="steelblue",  label="D (%)")
    ax7.semilogy(steps, rel_r, lw=2, color="darkorange", label="r (%)")
    ax7.set_xlabel("Epoch", fontsize=12); ax7.set_ylabel("Rel. error (%)", fontsize=12)
    ax7.set_title("Parameter relative error", fontsize=13)
    ax7.legend(fontsize=10, loc='lower left')

    # ── Row 3 middle: histogram of collocation x-density ─────────────────────
    ax8 = fig.add_subplot(gs[3, 1])
    ax8.hist(xc_init.ravel(), bins=60, alpha=0.6,
             color="steelblue", label=f"initial", density=True)
    if args.resample_every > 0 and args.n_kink > 0:
        ax8.hist(xc_final.ravel(), bins=60, alpha=0.6,
                 color="darkorange", label=f"final", density=True)
    ax8.set_xlabel("x", fontsize=12); ax8.set_ylabel("density", fontsize=12)
    ax8.set_title("x-distribution of collocation pts", fontsize=13)
    ax8.legend(fontsize=10, loc='lower right', markerscale=4)

    # ── Row 3 right: histogram of collocation t-density ──────────────────────
    ax9 = fig.add_subplot(gs[3, 2])
    ax9.hist(tc_init.ravel(), bins=60, alpha=0.6,
             color="steelblue", label="initial", density=True)
    if args.resample_every > 0 and args.n_kink > 0:
        ax9.hist(tc_final.ravel(), bins=60, alpha=0.6,
                 color="darkorange", label="final", density=True)
    ax9.set_xlabel("t", fontsize=12); ax9.set_ylabel("density", fontsize=12)
    ax9.set_title("t-distribution of collocation pts", fontsize=13)
    ax9.legend(fontsize=10, loc='lower right', markerscale=4)

    fig.suptitle(
        f"1D Allen-Cahn PINN [Adam + L-BFGS] | "
        f"D_true={D_true}, r_true={r_true} | "
        f"D_id={D_final:.4f}, r_id={r_final:.4f} | "
        f"{args.layers}Lx{args.neurons}N | \n"
        f"n_col={args.n_col}+{args.n_kink}kink | "
        f"resample_every={args.resample_every} | "
        f"w=({args.w_pde},{args.w_bc},{args.w_data},{args.w_phys}) ",
        fontsize=14, y=1.005  
    )

    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────────
# 10.  CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="1D Allen-Cahn PINN — Kink + energy/width constraints + oversampling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("Domain")
    g.add_argument("--x_min",  type=float, default=-10.0)
    g.add_argument("--x_max",  type=float, default=10.0)
    g.add_argument("--t_min",  type=float, default=0.0)
    g.add_argument("--t_max",  type=float, default=1.0)


    g = p.add_argument_group("True physics parameters")
    g.add_argument("--D_true", type=float, default=1.0)
    g.add_argument("--r_true", type=float, default=1.0)

    g = p.add_argument_group("Initial guess")
    g.add_argument("--D_init", type=float, default=2.0)
    g.add_argument("--r_init", type=float, default=2.0)

    g = p.add_argument_group("Network architecture")
    g.add_argument("--layers",  type=int, default=5)
    g.add_argument("--neurons", type=int, default=64)

    g = p.add_argument_group("Data")
    g.add_argument("--n_col",  type=int,   default=2000,
                   help="Number of uniform collocation points")
    g.add_argument("--n_kink", type=int,   default=500,
                   help="Extra collocation pts concentrated near the kink "
                        "(where |u_x| is large).  Set to 0 to disable oversampling.")
    g.add_argument("--resample_every", type=int, default=50,
                   help="Re-evaluate |u_x| on the current network and resample "
                        "the n_kink oversampled pts every N steps/epochs.  "
                        "Set to 0 to disable adaptive resampling.")
    g.add_argument("--n_bc",   type=int,   default=300)
    g.add_argument("--n_data", type=int,   default=500)
    g.add_argument("--noise",  type=float, default=0.0)

    g = p.add_argument_group("Loss weights")
    g.add_argument("--w_pde",  type=float, default=1) 
    g.add_argument("--w_bc",   type=float, default=10)
    g.add_argument("--w_data", type=float, default=10)
    g.add_argument("--w_phys", type=float, default=0,
                   help="Weight for energy/width parameter constraints L_D + L_r") 

    g = p.add_argument_group("Optimizer")
    g.add_argument("--adam_epochs", type=int,   default=5000)   
    g.add_argument("--adam_lr",     type=float, default=1e-3) 
    g.add_argument("--lbfgs_epochs", type=int, default=200)  
    g.add_argument("--lbfgs_lr",    type=float, default=1.0)    
    g.add_argument("--lbfgs_max_iter",     type=int, default=20)
    g.add_argument("--lbfgs_history_size", type=int, default=50)
  
    g = p.add_argument_group("Misc")
    g.add_argument("--seed",        type=int, default=42)
    g.add_argument("--device",      type=str, default="cpu",
                   choices=["cpu", "cuda", "mps"])
    g.add_argument("--summary_csv", type=str,
                   default="allen_cahn_1d_summary.csv")
    g.add_argument("--history_csv", type=str,
                   default="allen_cahn_1d_history.csv")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# 11.  Entry point
# ────────────────────────────────────────────────────────────────────────────── 

if __name__ == "__main__": 
    args = parse_args()

    #n_col_values = [500, 1000, 2000, 5000, 10000]
    n_col_values = [2000]

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU."); args.device = "cpu"
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU.");  args.device = "cpu"
    device = torch.device(args.device)
    print(f"Device: {device}")


    for n in n_col_values:
        print(f"\n--- RUNNING EXPERIMENT ---")
        
        args.n_col = n
         
        torch.manual_seed(args.seed)

        model, history, train_time, xc_init, tc_init, xc_final, tc_final = train(args, device)
        
        D_final = model.D.item()
        r_final = model.r.item()
        
        
        summary_path = args.summary_csv
        history_path = f"allen_cahn_1d_history.csv"

        save_csv(args, history, train_time, D_final, r_final,
                 summary_path=summary_path,
                 history_path=history_path)

        plot_filename = f"allen_cahn_1d_results.pdf"
        
        plot_results(model, history, args, device,
                 xc_init, tc_init, xc_final, tc_final, filename=plot_filename)
        
    print(f"\nAll experiments complete. Check {args.summary_csv} for your results!")
