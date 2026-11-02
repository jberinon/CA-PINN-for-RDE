"""
Physics-Informed Neural Network (PINN) for parameter identification
of the 1D Fisher-KPP equation:

    u_t - D * u_xx = r * u * (1 - u)

Uses the exact traveling wave solution to generate synthetic data,
then identifies D and r from the data + PDE residual.

Exact solution (minimum-speed traveling wave):
    alpha = sqrt(r / (6*D))
    c     = 5 * sqrt(D*r/6)
    u(x,t)= 1 / (1 + exp(alpha*(x - c*t)))^2

Outputs
-------
* fisher_kpp_results.pdf     – diagnostic figure
* fisher_kpp_summary.csv     – one-row summary (config + final results),
                               appended so multiple runs accumulate
* fisher_kpp_history.csv     – per-step loss / parameter history

Usage
-----
python fisher_kpp_1d_pinnLphys.py --noise 0.2

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
    """Traveling-wave exact solution of Fisher-KPP (numpy)."""
    alpha = math.sqrt(r / (6.0 * D))
    c     = 5.0 * math.sqrt(D * r / 6.0)
    xi    = alpha * (x - c * t)
    return 1.0 / (1.0 + np.exp(xi)) ** 2


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Neural network
# ──────────────────────────────────────────────────────────────────────────────

class PINN(nn.Module):
    """
    Fully-connected tanh network  (x, t) -> u  with sigmoid output (u in (0,1)).
    Trainable physics parameters D = exp(log_D) > 0  and  r = exp(log_r) > 0.
    """

    def __init__(self, n_layers: int, n_neurons: int,
                 D_init: float, r_init: float):
        super().__init__()

        layers = [nn.Linear(2, n_neurons), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers += [nn.Linear(n_neurons, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

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
# 3.  PDE residual
# ──────────────────────────────────────────────────────────────────────────────

def pde_residual(model, x_col, t_col):
    """Pointwise residual  u_t - D*u_xx - r*u*(1-u)."""
    x_col = x_col.requires_grad_(True)
    t_col = t_col.requires_grad_(True)
    u     = model(x_col, t_col)

    ones  = torch.ones_like(u)
    u_t   = torch.autograd.grad(u,   t_col, grad_outputs=ones, create_graph=True)[0]
    u_x   = torch.autograd.grad(u,   x_col, grad_outputs=ones, create_graph=True)[0]
    u_xx  = torch.autograd.grad(u_x, x_col,
                                grad_outputs=torch.ones_like(u_x),
                                create_graph=True)[0]
    return u_t - model.D * u_xx - model.r * u * (1.0 - u)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Loss helper (shared by both optimizers)
# ──────────────────────────────────────────────────────────────────────────────

def compute_loss(model, tensors, weights):
    """Returns (total_loss, loss_pde, loss_bc, loss_data)."""
    mse = nn.MSELoss()
    w_pde, w_bc, w_data = weights

    res       = pde_residual(model, tensors["x_col"], tensors["t_col"])
    loss_pde  = mse(res, torch.zeros_like(res))

    loss_bc   = (mse(model(tensors["x_ic"], tensors["t_ic"]), tensors["u_ic"]) +
                 mse(model(tensors["x_lb"], tensors["t_lb"]), tensors["u_lb"]) +
                 mse(model(tensors["x_rb"], tensors["t_rb"]), tensors["u_rb"]))

    loss_data = mse(model(tensors["x_dat"], tensors["t_dat"]), tensors["u_dat"])


    a = math.sqrt(args.r_true / (6.0 * args.D_true))
    c     = 5.0 * math.sqrt(args.D_true * args.r_true / 6.0)
    a_safe     = a + 1e-8
    loss_phys_D = (model.D - c / (5*a_safe)) ** 2
    loss_phys_r = (model.r - 6 * c*a_safe/5) ** 2

    loss = w_pde * loss_pde + w_bc * loss_bc + w_data * loss_data + (loss_phys_D + loss_phys_r)
    return loss, loss_pde, loss_bc, loss_data


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Training
# ──────────────────────────────────────────────────────────────────────────────

def train(args, device):

    D_true, r_true = args.D_true, args.r_true
    print(f"\nTrue parameters:  D = {D_true:.4f},  r = {r_true:.4f}")

    # ── Sampling ──────────────────────────────────────────────────────────────
    x_min, x_max = args.x_min, args.x_max
    t_min, t_max = args.t_min, args.t_max
    rng = np.random.default_rng(args.seed)

    def sample_xt(n, x_lo=x_min, x_hi=x_max, t_lo=t_min, t_hi=t_max):
        return (rng.uniform(x_lo, x_hi, (n, 1)).astype(np.float32),
                rng.uniform(t_lo, t_hi, (n, 1)).astype(np.float32))

    x_col, t_col = sample_xt(args.n_col)

    x_ic = rng.uniform(x_min, x_max, (args.n_bc, 1)).astype(np.float32)
    t_ic = np.full((args.n_bc, 1), t_min, dtype=np.float32)
    u_ic = exact_u(x_ic, t_ic, D_true, r_true).astype(np.float32)

    x_lb = np.full((args.n_bc, 1), x_min, dtype=np.float32)
    t_lb = rng.uniform(t_min, t_max, (args.n_bc, 1)).astype(np.float32)
    u_lb = exact_u(x_lb, t_lb, D_true, r_true).astype(np.float32)

    x_rb = np.full((args.n_bc, 1), x_max, dtype=np.float32)
    t_rb = rng.uniform(t_min, t_max, (args.n_bc, 1)).astype(np.float32)
    u_rb = exact_u(x_rb, t_rb, D_true, r_true).astype(np.float32)

    x_dat, t_dat = sample_xt(args.n_data)
    u_dat = exact_u(x_dat, t_dat, D_true, r_true).astype(np.float32)
    if args.noise > 0.0:
        u_dat += (args.noise * rng.standard_normal(u_dat.shape)).astype(np.float32)

    # ── Tensors ───────────────────────────────────────────────────────────────
    def T(a): return torch.tensor(a, device=device)
    tensors = dict(
        x_col=T(x_col), t_col=T(t_col),
        x_ic=T(x_ic),   t_ic=T(t_ic),   u_ic=T(u_ic),
        x_lb=T(x_lb),   t_lb=T(t_lb),   u_lb=T(u_lb),
        x_rb=T(x_rb),   t_rb=T(t_rb),   u_rb=T(u_rb),
        x_dat=T(x_dat), t_dat=T(t_dat), u_dat=T(u_dat),
    )
    weights = (args.w_pde, args.w_bc, args.w_data)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = PINN(args.layers, args.neurons, args.D_init, args.r_init).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Network:  {args.layers} hidden layers x {args.neurons} neurons  "
          f"({n_params:,} trainable params)")
    print(f"Init D={args.D_init:.4f}, r={args.r_init:.4f}  |  "
          f"n_col={args.n_col}, n_bc={args.n_bc}, n_data={args.n_data}")
    print(f"Weights  w_pde={args.w_pde}, w_bc={args.w_bc}, w_data={args.w_data}")

    history = {k: [] for k in ["loss", "loss_pde", "loss_bc", "loss_data", "D", "r"]}
    log_every = max(1, args.adam_epochs // 20)

    t0 = time.time()

    step_counter = 0

    # ---- Phase 1: Adam ------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.adam_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.adam_epochs, eta_min=1e-5)

    for epoch in range(1, args.adam_epochs + 1):
            model.train()
            optimizer.zero_grad()
            loss, lp, lb, ld = compute_loss(model, tensors, weights)
            loss.backward()
            optimizer.step()
            scheduler.step()

            step_counter += 1

            with torch.no_grad():
                history["loss"].append(loss.item())
                history["loss_pde"].append(lp.item())
                history["loss_bc"].append(lb.item())
                history["loss_data"].append(ld.item())
                history["D"].append(model.D.item())
                history["r"].append(model.r.item())

            if epoch % log_every == 0 or epoch == 1:
                print(
                    f"[Adam {epoch:5d}/{args.adam_epochs}] loss={loss.item():.3e} "
                    f"(pde={lp.item():.3e}, bc={lb.item():.3e}, data={ld.item():.3e}) | "
                    f"D={model.D.item():.4f} | r={model.r.item():.4f} | "
                    f"({time.time()-t0:.1f}s)"
                )
            

    # ---- Phase 2: L-BFGS fine-tuning of the network weights -----------
    if args.lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=args.lbfgs_lr,
            max_iter=args.lbfgs_max_iter,
            history_size=args.lbfgs_history_size,
            line_search_fn="strong_wolfe",
        )

        state = {}

        def closure():
            nonlocal step_counter
            lbfgs.zero_grad()
            loss, lp, lb, ld = compute_loss(model, tensors, weights)
            loss.backward()

            step_counter += 1

            # Save state for the outer loop logging
            state["loss"] = loss.item()
            state["loss_pde"] = lp.item()
            state["loss_bc"] = lb.item()
            state["loss_data"] = ld.item()
            state["D"] = model.D.item()
            state["r"] = model.r.item()

            return loss

        for lbfgs_step in range(1, args.lbfgs_steps + 1):
            model.train()
            lbfgs.step(closure)

            with torch.no_grad():
                history["loss"].append(state["loss"])
                history["loss_pde"].append(state["loss_pde"])
                history["loss_bc"].append(state["loss_bc"])
                history["loss_data"].append(state["loss_data"])
                history["D"].append(state["D"])
                history["r"].append(state["r"])

            if lbfgs_step % log_every == 0 or lbfgs_step == 1:
                print(
                    f"[L-BFGS {lbfgs_step:5d}/{args.lbfgs_steps}] loss={state['loss']:.3e} | "
                    f"(pde={state['loss_pde']:.3e}, bc={state['loss_bc']:.3e}, data={state['loss_data']:.3e}) | "
                    f"D={state['D']:.4f} | r={state['r']:.4f}"
                )

        print(
            f"[LBFGS Done] Final Loss={history['loss'][-1]:.3e} | "
            f"D={model.D.item():.4f} | r={model.r.item():.4f}"
        )

    train_time = time.time() - t0
    D_final = model.D.item()
    r_final = model.r.item()
    print(f"\nIdentified:  D = {D_final:.6f}  (true {D_true:.6f})"
          f"  rel_err = {abs(D_final-D_true)/D_true*100:.3f}%")
    print(f"             r = {r_final:.6f}  (true {r_true:.6f})"
          f"  rel_err = {abs(r_final-r_true)/r_true*100:.3f}%")
    print(f"Wall time: {train_time:.1f}s")

    return model, history, train_time



# ──────────────────────────────────────────────────────────────────────────────
# 6.  CSV export
# ──────────────────────────────────────────────────────────────────────────────

def save_csv(args, history, train_time, D_final, r_final,
             summary_path="fisher_kpp_summary.csv",
             history_path="fisher_kpp_history.csv"):
    """
    Appends one row to the summary CSV (header written only on first run),
    and writes the full per-step history CSV for this run.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    D_true, r_true = args.D_true, args.r_true

    # ── Summary CSV ───────────────────────────────────────────────────────────
    summary_fields = [
        # run metadata
        "timestamp", 
        # domain
        "x_min", "x_max", "t_min", "t_max",
        # architecture
        "layers", "neurons", "n_trainable_params",
        # data
        "n_col", "n_bc", "n_data", "noise",
        # loss weights
        "w_pde", "w_bc", "w_data",
        # optimiser hyper-params
        "adam_epochs", "adam_lr", 
        "lbfgs_steps", "lbfgs_lr", "lbfgs_max_iter", "lbfgs_history_size",
        # reproducibility
        "seed",
        # true parameters
        "D_true", "r_true",
        # initial guess
        "D_init", "r_init",
        # identified results
        "D_final", "r_final",
        "D_rel_err_pct", "r_rel_err_pct",
        # final losses
        "final_loss", "final_loss_pde", "final_loss_bc", "final_loss_data",
        # timing
        "train_time_s",
    ]

    n_params = sum(p.numel()
                   for p in PINN(args.layers, args.neurons,
                                 args.D_init, args.r_init).parameters())

    row = {
        "timestamp"         : timestamp,
        "x_min"             : args.x_min,
        "x_max"             : args.x_max,
        "t_min"             : args.t_min,
        "t_max"             : args.t_max,
        "layers"            : args.layers,
        "neurons"           : args.neurons,
        "n_trainable_params": n_params,
        "n_col"             : args.n_col,
        "n_bc"              : args.n_bc,
        "n_data"            : args.n_data,
        "noise"             : args.noise,
        "w_pde"             : args.w_pde,
        "w_bc"              : args.w_bc,
        "w_data"            : args.w_data,
        "adam_epochs"       : args.adam_epochs,
        "adam_lr"           : args.adam_lr,
        "lbfgs_steps"       : args.lbfgs_steps,
        "lbfgs_lr"          : args.lbfgs_lr,
        "lbfgs_max_iter"    : args.lbfgs_max_iter,
        "lbfgs_history_size": args.lbfgs_history_size,
        "seed"              : args.seed,
        "D_true"            : D_true,
        "r_true"            : r_true,
        "D_init"            : args.D_init,
        "r_init"            : args.r_init,
        "D_final"           : round(D_final, 8),
        "r_final"           : round(r_final, 8),
        "D_rel_err_pct"     : round(abs(D_final - D_true) / D_true * 100, 4),
        "r_rel_err_pct"     : round(abs(r_final - r_true) / r_true * 100, 4),
        "final_loss"        : round(history["loss"][-1],      8),
        "final_loss_pde"    : round(history["loss_pde"][-1],  8),
        "final_loss_bc"     : round(history["loss_bc"][-1],   8),
        "final_loss_data"   : round(history["loss_data"][-1], 8),
        "train_time_s"      : round(train_time, 2),
    }

    write_header = not os.path.isfile(summary_path)
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"Summary CSV  -> {summary_path}  (appended, {len(summary_fields)} columns)")

    # ── History CSV ───────────────────────────────────────────────────────────
    # history_fields = ["step", "loss", "loss_pde", "loss_bc", "loss_data", "D", "r"]
    # with open(history_path, "w", newline="") as f:
    #     writer = csv.DictWriter(f, fieldnames=history_fields)
    #     writer.writeheader()
    #     for i in range(len(history["loss"])):
    #         writer.writerow({
    #             "step"     : i + 1,
    #             "loss"     : round(history["loss"][i],      8),
    #             "loss_pde" : round(history["loss_pde"][i],  8),
    #             "loss_bc"  : round(history["loss_bc"][i],   8),
    #             "loss_data": round(history["loss_data"][i], 8),
    #             "D"        : round(history["D"][i],         8),
    #             "r"        : round(history["r"][i],         8),
    #         })
    # print(f"History CSV  -> {history_path}  ({len(history['loss'])} rows)")


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def plot_results(model, history, args, device):
    D_true, r_true = args.D_true, args.r_true
    D_final = model.D.item()
    r_final = model.r.item()

    model.eval()
    Nx, Nt = 200, 100
    xv  = np.linspace(args.x_min, args.x_max, Nx, dtype=np.float32)
    tv  = np.linspace(args.t_min, args.t_max, Nt, dtype=np.float32)
    XX, TT = np.meshgrid(xv, tv)

    x_flat = torch.tensor(XX.ravel()[:, None], device=device)
    t_flat = torch.tensor(TT.ravel()[:, None], device=device)
    with torch.no_grad():
        u_pred = model(x_flat, t_flat).cpu().numpy().reshape(Nt, Nx)

    u_exact = exact_u(XX, TT, D_true, r_true)
    err     = np.abs(u_pred - u_exact)
    steps   = np.arange(1, len(history["loss"]) + 1)

    fig = plt.figure(figsize=(18, 12))
    gs  = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Row 0 – solution fields
    for col, (data, title) in enumerate([
        (u_exact, "Exact  u(x,t)"),
        (u_pred,  "PINN  u(x,t)"),
        (err,     f"|PINN - Exact|  (max={err.max():.2e})"),
    ]):
        ax   = fig.add_subplot(gs[0, col])
        cmap = "viridis" if col < 2 else "hot_r"
        kw   = {"vmin": u_exact.min(), "vmax": u_exact.max()} if col == 1 else {}
        im   = ax.pcolormesh(XX, TT, data, cmap=cmap, shading="auto", **kw)
        fig.colorbar(im, ax=ax)
        ax.set_title(title); ax.set_xlabel("x"); ax.set_ylabel("t")

    # Row 1 left – profiles at 5 times
    ax3 = fig.add_subplot(gs[1, :2])
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, 5))
    for i, ti in enumerate(np.linspace(args.t_min, args.t_max, 5)):
        idx = int((ti - args.t_min) / (args.t_max - args.t_min) * (Nt - 1))
        ax3.plot(xv, u_exact[idx], color=colors[i], lw=2,
                 label=f"exact t={ti:.2f}")
        ax3.plot(xv, u_pred[idx],  color=colors[i], lw=1.5, ls="--",
                 label=f"PINN  t={ti:.2f}")
    ax3.set_xlabel("x"); ax3.set_ylabel("u")
    ax3.set_title("Profiles at selected times")
    ax3.legend(fontsize=7, ncol=2)

    # Row 1 right – loss history
    ax4 = fig.add_subplot(gs[1, 2])
    for key, label, ls in [("loss","Total","-"),("loss_pde","PDE","--"),
                            ("loss_bc","BC","-."),("loss_data","Data",":")]:
        ax4.semilogy(steps, history[key], label=label, ls=ls)
    ax4.set_xlabel("Step"); ax4.set_ylabel("MSE loss")
    ax4.set_title(f"Loss history")
    ax4.legend(fontsize=8)

    # Row 2 – D, r convergence + relative errors
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(steps, history["D"], lw=2, color="steelblue")
    ax5.axhline(D_true, color="red", lw=1.5, ls="--", label=f"True D={D_true}")
    ax5.set_xlabel("Step"); ax5.set_ylabel("D")
    ax5.set_title(f"D convergence  (final={D_final:.5f})")
    ax5.legend(fontsize=8)

    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(steps, history["r"], lw=2, color="darkorange")
    ax6.axhline(r_true, color="red", lw=1.5, ls="--", label=f"True r={r_true}")
    ax6.set_xlabel("Step"); ax6.set_ylabel("r")
    ax6.set_title(f"r convergence  (final={r_final:.5f})")
    ax6.legend(fontsize=8)

    ax7 = fig.add_subplot(gs[2, 2])
    rel_D = np.abs(np.array(history["D"]) - D_true) / D_true * 100
    rel_r = np.abs(np.array(history["r"]) - r_true) / r_true * 100
    ax7.semilogy(steps, rel_D, lw=2, color="steelblue",  label="D (%)")
    ax7.semilogy(steps, rel_r, lw=2, color="darkorange", label="r (%)")
    ax7.set_xlabel("Step"); ax7.set_ylabel("Relative error (%)")
    ax7.set_title("Parameter relative error")
    ax7.legend(fontsize=8)

    fig.suptitle(
        f"Fisher-KPP PINN |  "
        f"D_true={D_true}, r_true={r_true}  |  "
        f"D_id={D_final:.4f}, r_id={r_final:.4f}  |  "
        f"{args.layers}L x {args.neurons}N  n_col={args.n_col}  "
        f"w=({args.w_pde},{args.w_bc},{args.w_data})",
        fontsize=10, y=1.01,
    )

    out = "fisher_kpp_results.pdf"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Figure saved")
    plt.close(fig)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 8.  CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="PINN parameter identification for Fisher-KPP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("Domain")
    g.add_argument("--x_min",  type=float, default=-20.0)
    g.add_argument("--x_max",  type=float, default= 20.0)
    g.add_argument("--t_min",  type=float, default=  0.0)
    g.add_argument("--t_max",  type=float, default=  5.0)

    g = p.add_argument_group("True physics parameters")
    g.add_argument("--D_true", type=float, default=1.0,  help="True diffusion coeff")
    g.add_argument("--r_true", type=float, default=1.0,  help="True growth rate")

    g = p.add_argument_group("Initial guess for identified parameters")
    g.add_argument("--D_init", type=float, default=2.)
    g.add_argument("--r_init", type=float, default=2.)

    g = p.add_argument_group("Network architecture")
    g.add_argument("--layers",  type=int, default=5,  help="Hidden layers")
    g.add_argument("--neurons", type=int, default=64, help="Neurons per layer")

    g = p.add_argument_group("Data")
    g.add_argument("--n_col",  type=int,   default=2000, help="PDE collocation pts")
    g.add_argument("--n_bc",   type=int,   default=400,  help="BC/IC pts per boundary")
    g.add_argument("--n_data", type=int,   default=800,  help="Observation pts")
    g.add_argument("--noise",  type=float, default=0.0,  help="Gaussian noise std")

    g = p.add_argument_group("Loss weights")
    g.add_argument("--w_pde",  type=float, default=1.0)
    g.add_argument("--w_bc",   type=float, default=10.0)
    g.add_argument("--w_data", type=float, default=10.0)

    g = p.add_argument_group("Optimizer")
    g.add_argument("--adam_epochs", type=int, default=5000,
                   help="Adam iterations")             
    g.add_argument("--adam_lr",     type=float, default=1e-3)
    g.add_argument("--lbfgs_steps", type=int, default=200)
    g.add_argument("--lbfgs_lr",    type=float, default=1.0)
    g.add_argument("--lbfgs_max_iter",     type=int, default=20,
                   help="[L-BFGS] max function evals per .step() call")
    g.add_argument("--lbfgs_history_size", type=int, default=50,
                   help="[L-BFGS] number of past gradients stored")

    g = p.add_argument_group("Misc")
    g.add_argument("--seed",        type=int, default=42)
    g.add_argument("--device",      type=str, default="cpu",
                   choices=["cpu", "cuda", "mps"])
    g.add_argument("--summary_csv", type=str, default="fisher_kpp_summary.csv",
                   help="Appended summary file (one row per run)")
    g.add_argument("--history_csv", type=str, default="fisher_kpp_history.csv",
                   help="Per-step history file (overwritten each run)")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# 9.  Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        args.device = "cpu"
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)
    print(f"Device: {device}")

    torch.manual_seed(args.seed)

    model, history, train_time = train(args, device)
    D_final = model.D.item()
    r_final = model.r.item()

    plot_results(model, history, args, device)
    save_csv(args, history, train_time, D_final, r_final,
             summary_path=args.summary_csv,
             history_path=args.history_csv)
