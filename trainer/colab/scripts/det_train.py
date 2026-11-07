"""Train the detector ESPDet-Pico - from scratch or (--start-from) continuing from your own previous model -, resume, test.

Proven settings: 300 epochs, patience 100, batch 64, imgsz 224, cache, fliplr=0, mosaic with close_mosaic=30.
Continuing (--start-from <old run>/weights/best.pt): a new run that starts with the weights of the previous model
(fine-tuning, e.g. with additional renders). Own, shorter settings: AdamW, learning rate 0.0005,
1 warm-up epoch, mosaic off in the last 10 epochs. This is not resuming the old run (that would be --resume).
Checkpoints (last.pt / best.pt) go directly into the output folder on Drive -> resuming after an abort is possible.
The degradation to camera board quality (blur, noise, JPEG, slight colour cast) happens at random for each image.

Examples:
  python det_train.py --data /content/det_data/data.yaml --output .../detector/runs --name run1
  python det_train.py --data ... --output ... --name timing --epochs 1          (measure the time of 1 epoch)
  python det_train.py --data ... --output ... --name run1 --resume
  python det_train.py --data ... --output ... --name run1 --test-only
  python det_train.py --data ... --output ... --name run2 --start-from .../runs/run1/weights/best.pt --epochs 80 --patience 25
  python det_train.py --help
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import espdet_patches  # noqa: E402


def test(weights, data, imgsz, device):
    from ultralytics import YOLO
    m = YOLO(weights)
    res = {}
    for split, name in (("val", "synthetic"), ("test", "real")):
        try:
            r = m.val(data=data, split=split, imgsz=imgsz, batch=32, device=device, verbose=False, plots=False)
        except Exception:   # no test split (no real images yet)
            if split == "test":
                continue
            raise
        res[name] = {"mAP50": round(float(r.box.map50), 4), "recall": round(float(r.box.mr), 4),
                     "precision": round(float(r.box.mp), 4)}
    return res


def duration(seconds):
    return f"{seconds / 60:.0f} min" if seconds < 3600 else f"{seconds / 3600:.1f} h"


def epoch_report(model):
    """Prints one short line per epoch: number, validation, duration, best epoch and the maximum remaining time."""
    state = {"t": time.time(), "epoch": False}

    def on_train_epoch_end(trainer):
        state["epoch"] = True

    def on_fit_epoch_end(trainer):
        m = trainer.metrics or {}
        if not state["epoch"]:   # also called once more at the end for the final validation of best.pt
            print(f"  final validation of best.pt: mAP50 {float(m.get('metrics/mAP50(B)', 0)):.3f} · "
                  f"recall {float(m.get('metrics/recall(B)', 0)):.3f} · precision {float(m.get('metrics/precision(B)', 0)):.3f}",
                  flush=True)
            return
        state["epoch"] = False
        e, total = trainer.epoch + 1, trainer.epochs
        now = time.time()
        seconds, state["t"] = now - state["t"], now
        stopper = getattr(trainer, "stopper", None)
        best = stopper.best_epoch if stopper is not None else e   # stopper counts from 1
        patience = getattr(stopper, "patience", 0) or 0
        left = min(total - e, max(0, best + patience - e)) if patience else total - e
        print(f"  epoch {e}/{total}: mAP50 {float(m.get('metrics/mAP50(B)', 0)):.3f} · "
              f"recall {float(m.get('metrics/recall(B)', 0)):.3f} · precision {float(m.get('metrics/precision(B)', 0)):.3f} · "
              f"{seconds:.0f} s · best epoch {best} · at most {left} more (≤ {duration(left * seconds)})", flush=True)

    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    return model


def main():
    ap = argparse.ArgumentParser(description="Train ESPDet-Pico.")
    ap.add_argument("--data", required=True, help="data.yaml (from det_dataset.py)")
    ap.add_argument("--output", required=True, help="folder for runs (on Drive)")
    ap.add_argument("--name", default="run1")
    ap.add_argument("--esp-detection", default=os.environ.get("ESP_DETECTION", "/content/esp-detection"))
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=100, help="stop if there is no improvement for this many epochs")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--degrade", type=float, default=1.0, help="strength of the degradation to camera board quality (0 = off)")
    ap.add_argument("--device", default=None, help="0 = first GPU, cpu = processor (default: GPU if present)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--start-from", help="start weights (best.pt of an earlier run) -> continue training instead of from scratch")
    ap.add_argument("--lr", type=float, default=0.0005, help="learning rate when continuing (default 0.0005)")
    ap.add_argument("--test-only", action="store_true")
    ap.add_argument("--result", default="result.json", help="file name for the test result in the run folder")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.device is None:
        import torch
        args.device = "0" if torch.cuda.is_available() else "cpu"

    yaml = espdet_patches.patches(args.esp_detection)
    from ultralytics import YOLO
    run = os.path.join(args.output, args.name)
    last = os.path.join(run, "weights", "last.pt")
    best = os.path.join(run, "weights", "best.pt")

    if not args.test_only:
        t0 = time.time()
        if args.resume:
            if not os.path.isfile(last):
                print(f"Nothing to resume ({last} is missing).")
                return
            epoch_report(YOLO(last)).train(resume=True)
        else:
            if os.path.isfile(last):
                sys.exit(f"Run “{args.name}” exists already. To continue it use --resume, otherwise choose another --name.")
            common = dict(
                data=args.data, imgsz=args.imgsz, epochs=args.epochs, patience=args.patience, batch=args.batch,
                single_cls=True, cache=True, fliplr=0.0, mosaic=1.0, mixup=0.0, copy_paste=0.1,
                project=args.output, name=args.name, exist_ok=True, device=args.device, workers=args.workers,
                plots=False, val=True,
                augmentations=espdet_patches.degradation(args.degrade) if args.degrade > 0 else None)
            if args.start_from:
                if not os.path.isfile(args.start_from):
                    sys.exit(f"Start weights missing: {args.start_from}")
                print(f"continuing from {args.start_from} (at most {args.epochs} epochs, learning rate {args.lr})", flush=True)
                epoch_report(YOLO(args.start_from)).train(optimizer="AdamW", lr0=args.lr, warmup_epochs=1, warmup_bias_lr=args.lr,
                                            close_mosaic=min(10, max(0, args.epochs - 1)), **common)
            else:
                epoch_report(YOLO(yaml)).train(optimizer="auto", close_mosaic=min(30, max(0, args.epochs - 1)), **common)
        duration = time.time() - t0
        print(f"⏱ training: {duration / 60:.1f} min")

    if os.path.isfile(best):
        res = test(best, args.data, args.imgsz, args.device)
        res["weights"] = os.path.basename(best)
        res["imgsz"] = args.imgsz
        with open(os.path.join(run, args.result), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
        for k in ("synthetic", "real"):
            if k in res:
                e = res[k]
                print(f"✔ test {k:10s}: mAP50 {e['mAP50']:.3f} · recall {e['recall']:.3f} · precision {e['precision']:.3f}")


if __name__ == "__main__":
    main()
