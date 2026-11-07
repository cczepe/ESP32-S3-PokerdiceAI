"""Train the classifier from scratch, resume, and compare several variants.

- All crops are in GPU memory; the degradation happens on the GPU, new at random in every epoch:
  shift/scale/rotate (boxes never sit exactly), brightness, contrast, gamma, slight white balance,
  slight saturation (NO hue shift - colour tells e.g. queen from jack apart), blur, noise, downscaling.
- AdamW + OneCycle, mixed precision, after every epoch a checkpoint (.ckpt, atomic) and the best model (.pth).
- --compare: table of several models (parameters, estimated .espdl size, accuracy per die and per throw,
  synthetic and real, most frequent confusions).

Examples:
  python cls_train.py --data .../classifier/crops --output .../classifier/models --name b025 --width 0.25
  python cls_train.py --data ... --output ... --name s025 --width 0.25 --fast      (fast architecture)
  python cls_train.py --data ... --output ... --name b100 --width 1.0 --lr 0.001
  python cls_train.py --data ... --output ... --name b025 --resume
  python cls_train.py --data ... --output ... --name timing --width 0.25 --epochs 1      (measure the time of 1 epoch)
  python cls_train.py --data ... --output ... --compare b025 b050 b100
  python cls_train.py --data ... --output ... --name s025r --width 0.25 --fast --blur 2 --color-cast 2   (more robust for blurred/colour-cast camera images)
  python cls_train.py --help
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cls_model  # noqa: E402

GRAY = 114 / 255.0


# ------------------------------------------------------------------ data
def load_data(folder, device):
    with open(os.path.join(folder, "info.json"), encoding="utf-8") as f:
        info = json.load(f)
    xs, ys, bs, vs = [], [], [], []
    for f in sorted(glob.glob(os.path.join(folder, "crops_*.npz"))):
        if ".tmp" in f:
            continue
        d = np.load(f)
        xs.append(d["x"]), ys.append(d["y"]), bs.append(d["image"]), vs.append(d["val"])
    if not xs:
        sys.exit(f"No crops in {folder} (run cls_crops.py first).")
    x, y, b, v = np.concatenate(xs), np.concatenate(ys), np.concatenate(bs), np.concatenate(vs)
    real = None
    if os.path.isfile(os.path.join(folder, "real.npz")):
        e = np.load(os.path.join(folder, "real.npz"))
        real = (torch.from_numpy(e["x"]).to(device), torch.from_numpy(e["y"].astype(np.int64)).to(device), e["image"])
    t = lambda a: torch.from_numpy(a).to(device)
    return info, t(x[~v]), t(y[~v].astype(np.int64)), t(x[v]), t(y[v].astype(np.int64)), b[v], real


# ------------------------------------------------------------------ degradation on the GPU
def prepare(xb, N, factor, train, strength=1.0, blur=1.0, color_cast=1.0):
    """uint8 NHWC (stored size G) -> float NCHW N×N in [0,1].
    blur/color_cast (default 1.0 = normal): strengthen ONLY the blur or the white balance/contrast -
    for real camera board images (dice close to the camera can be much blurrier, colour casts happen)."""
    x = xb.permute(0, 3, 1, 2).float() / 255.0
    B = x.shape[0]
    dev = x.device
    s = torch.full((B,), 1.0 / factor, device=dev)
    rot = torch.zeros(B, device=dev)
    tx = torch.zeros(B, device=dev)
    ty = torch.zeros(B, device=dev)
    if train:
        s = s * torch.empty(B, device=dev).uniform_(1 - 0.1 * strength, 1 + 0.1 * strength)
        rot = torch.empty(B, device=dev).uniform_(-15, 15) * math.pi / 180 * strength
        tx = torch.empty(B, device=dev).uniform_(-0.06, 0.06) * strength
        ty = torch.empty(B, device=dev).uniform_(-0.06, 0.06) * strength
    theta = torch.stack([torch.stack([s * torch.cos(rot), -s * torch.sin(rot), tx], 1),
                         torch.stack([s * torch.sin(rot), s * torch.cos(rot), ty], 1)], 1)
    grid = F.affine_grid(theta, (B, 3, N, N), align_corners=False)
    x = F.grid_sample(x - GRAY, grid, mode="bilinear", padding_mode="zeros", align_corners=False) + GRAY
    if not train:
        return x.clamp(0, 1)
    k = strength

    def u(lo, hi, shape=(B, 1, 1, 1)):
        return torch.empty(shape, device=dev).uniform_(lo, hi)
    x = x * u(1 - 0.3 * k, 1 + 0.3 * k)                                        # brightness
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - m) * u(1 - 0.25 * k * (1 if color_cast <= 1 else 1.6), 1 + 0.25 * k) + m   # contrast (flatter with colour cast > 1)
    x = x.clamp(1e-4, 1) ** u(0.8, 1.25)                                       # gamma
    x = x * u(1 - 0.08 * k * color_cast, 1 + 0.08 * k * color_cast, (B, 3, 1, 1))  # white balance (colour cast)
    g = x.mean(dim=1, keepdim=True)
    x = g + (x - g) * u(1 - 0.25 * k, 1 + 0.2 * k)                             # slight saturation
    sel = torch.rand(B, device=dev) < min(0.9, 0.4 * k * blur)                 # blur
    if sel.any():
        sigma = float(np.random.uniform(0.4, 1.3 * blur))
        rr = 2 if blur <= 1 else int(math.ceil(2.5 * 1.3 * blur))   # kernel radius (5×5 at 1.0)
        r = torch.arange(-rr, rr + 1, device=dev, dtype=torch.float32)
        kern = torch.exp(-r ** 2 / (2 * sigma ** 2))
        kern = (kern / kern.sum())
        k2 = (kern[:, None] * kern[None, :]).expand(3, 1, 2 * rr + 1, 2 * rr + 1)
        x[sel] = F.conv2d(F.pad(x[sel], (rr, rr, rr, rr), mode="replicate"), k2, groups=3)
    sel = torch.rand(B, device=dev) < 0.3 * k                                   # downscale (few pixels)
    if sel.any():
        f = float(np.random.uniform(0.5, 0.9))
        small = F.interpolate(x[sel], scale_factor=f, mode="bilinear", align_corners=False)
        x[sel] = F.interpolate(small, size=(N, N), mode="bilinear", align_corners=False)
    x = x + torch.randn_like(x) * u(0, 0.05 * k)                               # noise
    return x.clamp(0, 1)


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def predict(model, x, N, factor, batch=1024):
    model.eval()
    out = []
    for i in range(0, len(x), batch):
        with torch.autocast(device_type=x.device.type, enabled=x.device.type == "cuda"):
            out.append(model(prepare(x[i:i + batch], N, factor, False)).float().argmax(1))
    return torch.cat(out) if out else torch.zeros(0, dtype=torch.long, device=x.device)


def key_figures(pred, y, image, classes):
    p, t = pred.cpu().numpy(), y.cpu().numpy()
    if len(t) == 0:
        return None
    right = p == t
    throws = {}
    for r, b in zip(right, image):
        throws.setdefault(int(b), []).append(r)
    conf = {}
    for a, b in zip(t[~right], p[~right]):
        k = f"{classes[a]}→{classes[b]}"
        conf[k] = conf.get(k, 0) + 1
    return {"per_die": round(float(right.mean()), 4),
            "per_throw": round(float(np.mean([all(v) for v in throws.values()])), 4),
            "count": int(len(t)), "throws": len(throws),
            "confusions": dict(sorted(conf.items(), key=lambda kv: -kv[1])[:6])}


# ------------------------------------------------------------------ training
def save_atomic(obj, path):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def train(args, device):
    info, xt, yt, xv, yv, bv, real = load_data(args.data, device)
    kl, N, factor = info["classes"], info["size"], info["factor"]
    os.makedirs(args.output, exist_ok=True)
    ckpt_p = os.path.join(args.output, f"{args.name}.ckpt")
    pth_p = os.path.join(args.output, f"{args.name}.pth")
    model = cls_model.Classifier(len(kl), args.width, args.split, fast=args.fast).to(device)
    steps = math.ceil(len(xt) / args.batch)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps, pct_start=0.15)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    start, best = 0, -1.0
    if args.resume:
        if not os.path.isfile(ckpt_p):
            print(f"Nothing to resume ({ckpt_p} is missing).")
            return
        c = torch.load(ckpt_p, map_location=device, weights_only=False)
        model.load_state_dict(c["model"]), opt.load_state_dict(c["opt"]), sched.load_state_dict(c["sched"])
        scaler.load_state_dict(c["scaler"])
        start, best = c["epoch"] + 1, c["best"]
        args.epochs = c["args"]["epochs"]
        print(f"resuming at epoch {start + 1}/{args.epochs} (best validation accuracy so far {best:.4f})")
    elif os.path.isfile(ckpt_p):
        sys.exit(f"“{args.name}” exists already. To continue it use --resume, otherwise another --name.")
    print(f"Variant {args.name}: width {args.width}{' split' if args.split else ''}{' fast' if args.fast else ''}, "
          f"{cls_model.parameters(model):,} parameters, {cls_model.compute(model, N) / 1e6:.1f} million MAC, "
          f"{len(xt)} training / {len(xv)} validation, input {N}×{N}, {len(kl)} classes")
    loss_f = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    for ep in range(start, args.epochs):
        t0 = time.time()
        model.train()
        order = torch.randperm(len(xt), device=device)
        total = 0.0
        for i in range(steps):
            idx = order[i * args.batch:(i + 1) * args.batch]
            xb = prepare(xt[idx], N, factor, True, args.degrade, args.blur, args.color_cast)
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                loss = loss_f(model(xb), yt[idx])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item()
        pv = predict(model, xv, N, factor)
        acc = float((pv == yv).float().mean()) if len(yv) else 0.0
        new_best = acc > best
        if new_best:
            best = acc
            save_atomic({"architecture": model.architecture, "model": model.state_dict(), "classes": kl, "size": N,
                         "margin": info["margin"], "factor": factor, "epoch": ep + 1, "val_accuracy": acc}, pth_p)
        save_atomic({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                     "scaler": scaler.state_dict(), "epoch": ep, "best": best, "args": vars(args)}, ckpt_p)
        print(f"  epoch {ep + 1:3d}/{args.epochs}: loss {total / steps:.4f} · validation {acc:.4f}"
              f"{' ★' if new_best else ''} · {time.time() - t0:.1f} s", flush=True)
    with open(os.path.join(args.output, f"{args.name}.done.json"), "w", encoding="utf-8") as f:
        json.dump({"name": args.name, "width": args.width, "fast": args.fast, "epochs": args.epochs,
                   "blur": args.blur, "color_cast": args.color_cast, "val_accuracy": best}, f, indent=1)
    print(f"✔ done: best model {pth_p} (validation accuracy {best:.4f})")


def compare(args, device):
    info, xt, yt, xv, yv, bv, real = load_data(args.data, device)
    kl, N, factor = info["classes"], info["size"], info["factor"]
    rows = []
    for name in args.compare:
        p = name if name.endswith(".pth") else os.path.join(args.output, f"{name}.pth")
        model, d = cls_model.load(p, device)
        par = cls_model.parameters(model)
        e = {"name": os.path.basename(p)[:-4] if not name.endswith(".pth")
             else os.path.basename(os.path.dirname(p)) + "/" + os.path.basename(p)[:-4],
             "width": d["architecture"]["width"], "split": d["architecture"]["split"],
             "fast": d["architecture"].get("fast", False),
             "parameters": par, "espdl_kb_estimated": round(par * 1.2 / 1024 + 8, 1),
             "million_mac": round(cls_model.compute(model, N) / 1e6, 1),
             "synthetic": key_figures(predict(model, xv, N, factor), yv, bv, kl)}
        if real is not None:
            e["real"] = key_figures(predict(model, real[0], N, factor), real[1], real[2], kl)
        rows.append(e)
    md = ["| Variant | Parameters | .espdl ≈ | Compute | synth. per die | synth. per throw | real per die | real per throw | most frequent confusions |",
          "|---|---|---|---|---|---|---|---|---|"]
    for e in rows:
        k, r = e["synthetic"], e.get("real")
        conf = r["confusions"] if r else k["confusions"]
        big = " ⚠ too large (partition 1 MB)" if e["espdl_kb_estimated"] > 1000 else ""
        md.append(f"| {e['name']} | {e['parameters']:,} | {e['espdl_kb_estimated']} KB{big} | {e['million_mac']} million MAC | "
                  f"{k['per_die']:.1%} | {k['per_throw']:.1%} | "
                  + (f"{r['per_die']:.1%} | {r['per_throw']:.1%}" if r else "– | –") + " | "
                  + ", ".join(f"{a} {n}×" for a, n in list(conf.items())[:4]) + " |")
    text = "\n".join(md) + ("\n\nCompute = multiplications per die; the compute time on the board grows roughly in "
                             "the same proportion.")
    print(text)
    with open(os.path.join(args.output, "compare.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    with open(os.path.join(args.output, "compare.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1, ensure_ascii=False)
    if real is None:
        print("⚠ No real test images yet - values only on synthetic images (deceptive!).")


def main():
    ap = argparse.ArgumentParser(description="Train the classifier from scratch or compare variants.")
    ap.add_argument("--data", required=True, help="folder from cls_crops.py")
    ap.add_argument("--output", required=True, help="folder for models (on Drive)")
    ap.add_argument("--name", default="b025")
    ap.add_argument("--width", type=float, default=0.25)
    ap.add_argument("--split", action="store_true", help="split (depthwise) convolutions - smaller, loses more with int8")
    ap.add_argument("--fast", action="store_true", help="first stage with stride 2 - about half the compute")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.003, help="learning rate (large variants: ~0.001, otherwise the training tips over)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--degrade", type=float, default=1.0)
    ap.add_argument("--blur", type=float, default=1.0,
                    help="strengthen only the blur (1.0 = normal; 2.0 = twice as strong/frequent, for blurred camera images)")
    ap.add_argument("--color-cast", type=float, default=1.0,
                    help="strengthen only the white balance/contrast (1.0 = normal; 2.0 = ±16 %% per colour channel, flatter contrast)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--compare", nargs="+", help="compare names or .pth files")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    torch.manual_seed(0)
    np.random.seed(0)
    compare(args, device) if args.compare else train(args, device)


if __name__ == "__main__":
    main()
