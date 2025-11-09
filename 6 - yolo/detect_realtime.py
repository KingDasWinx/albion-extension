import time
import argparse
from pathlib import Path
import sys
import threading

import numpy as np
import cv2

# Lazy imports inside functions for optional dependencies

"""Realtime classification of fishing bite using YOLOv8 classification model.
Steps:
1. Load trained classification model (best.pt)
2. Continuously capture a screen region around the bobber (user supplies coordinates) OR read frames from a video file for offline testing
3. Run classification; if class 'bite' exceeds probability threshold for N consecutive frames, trigger key press.
4. Debounce to avoid repeated triggers.

You can find your model after training in runs/fish_cls/weights/best.pt (depending on --name).
"""

timelzm = 0.4

def parse_args():
    p = argparse.ArgumentParser(description="Realtime fishing bite detection")
    p.add_argument("--weights", type=str, default="runs/fish_cls/weights/best.pt", help="Path to trained classification weights")
    p.add_argument("--region", type=int, nargs=4, metavar=("X","Y","W","H"), help="Screen capture region. If omitted, full screen.")
    p.add_argument("--video", type=str, help="Path to a video file (e.g., .mp4) for offline testing instead of live screen capture")
    p.add_argument("--threshold", type=float, default=0.70, help="Probability threshold for 'bite' class")
    p.add_argument("--consecutive", type=int, default=1, help="Consecutive frames required above threshold")
    p.add_argument("--cooldown", type=float, default=2.0, help="Seconds cooldown after action triggers")
    # No click/keyboard action; we start the minigame directly
    p.add_argument("--interval", type=float, default=0.15, help="Seconds between captures")
    p.add_argument("--imgsz", type=int, default=224, help="Classification image size")
    p.add_argument("--device", type=str, default="auto", help="Device for inference: auto/cpu/0")
    p.add_argument("--show", action="store_true", help="Show debug window with probabilities")
    p.add_argument("--verbose", action="store_true", help="Print per-frame logs (probability, consecutive)")
    p.add_argument("--log-every", type=int, default=1, help="Log every N frames when --verbose is set")
    p.add_argument("--decay", action="store_true", help="Use decay for consecutive counter (decrease by 1 instead of reset)")
    p.add_argument("--quit-key", type=str, default="q", help="Key to press in window to quit")
    p.add_argument("--no-action", action="store_true", help="Do not start minigame; only log detections (testing)")
    return p.parse_args()


def load_model(weights_path: str, device: str):
    try:
        from ultralytics import YOLO
        import torch
    except Exception as e:
        print("[error] Missing ultralytics or torch. Install requirements first.")
        print(e)
        sys.exit(1)

    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"
    print(f"[info] Using device {device}")
    model = YOLO(weights_path)
    return model, device


def press_key(key: str):
    """Deprecated: no longer used."""


# No click/keyboard automation during trigger; we run the minigame directly.

def _hold_left_click(seconds: float = 0.7, move_to_center: bool = False):
    """Robust hold of left button for 'seconds'. Optionally move cursor to minigame region center first.
    Fallback order: pydirectinput -> pyautogui -> ctypes SendInput.
    """
    # Try to get region center from config
    center = None
    if move_to_center:
        try:
            from script import config as _cfg
            if getattr(_cfg, 'MINIGAME_REGION', None):
                x, y, w, h = _cfg.MINIGAME_REGION
                center = (int(x + w/2), int(y + h/2))
        except Exception:
            center = None

    def _move(xy):
        if not xy:
            return
        x, y = xy
        try:
            import pydirectinput
            pydirectinput.moveTo(x, y)
        except Exception:
            try:
                import pyautogui
                pyautogui.moveTo(x, y)
            except Exception:
                pass

    def _send_ctypes(down: bool):
        # Windows only low-level fallback
        try:
            import ctypes
            import ctypes.wintypes as wt
            MOUSEEVENTF_LEFTDOWN = 0x0002
            MOUSEEVENTF_LEFTUP = 0x0004
            flags = MOUSEEVENTF_LEFTDOWN if down else MOUSEEVENTF_LEFTUP
            ctypes.windll.user32.mouse_event(flags, 0, 0, 0, 0)
        except Exception as e:
            if down:
                print(f"[warn] Down fallback failed: {e}")
            else:
                print(f"[warn] Up fallback failed: {e}")

    if center:
        _move(center)

    started = False
    try:
        import pydirectinput
        pydirectinput.mouseDown(button='left')
        started = True
        time.sleep(max(0.0, seconds))
        pydirectinput.mouseUp(button='left')
        return
    except Exception:
        pass
    if not started:
        try:
            import pyautogui
            pyautogui.mouseDown(button='left')
            started = True
            time.sleep(max(0.0, seconds))
            pyautogui.mouseUp(button='left')
            return
        except Exception:
            pass
    # ctypes fallback
    if not started:
        _send_ctypes(True)
        time.sleep(max(0.0, seconds))
    _send_ctypes(False)


def _run_minigame_direct():
    """Import and run the minigame directly via function, blocking until done.
    Configuration is read from script/config.py to avoid CLI arg bloat.
    """
    try:
        # Ensure local import from script package
        from script.hotkey_runner import run_minigame
        try:
            from script import config as _cfg
        except Exception:
            _cfg = None
    except Exception as e:
        print(f"[error] Could not import run_minigame from script.hotkey_runner: {e}")
        return -1
    # Defaults
    script = 'v2'
    region = None
    threshold = 0.5
    fps = 60
    debug = False
    focus_window = None
    if _cfg is not None:
        script = getattr(_cfg, 'MINIGAME_SCRIPT', script)
        region = getattr(_cfg, 'MINIGAME_REGION', region)
        threshold = getattr(_cfg, 'MINIGAME_THRESHOLD', threshold)
        fps = getattr(_cfg, 'MINIGAME_FPS', fps)
        debug = getattr(_cfg, 'MINIGAME_DEBUG', debug)
        focus_window = getattr(_cfg, 'MINIGAME_FOCUS_WINDOW', focus_window)
    return run_minigame(script=script, region=region, threshold=threshold, fps=fps, debug=debug, focus_window=focus_window)


def _start_pause_hotkey_listener(toggle_callback):
    """Start a background thread that toggles pause on Ctrl+F using Windows hotkey API.
    Returns (stop_event, thread)."""
    stop_event = threading.Event()

    def loop():
        try:
            import ctypes
            import ctypes.wintypes as wt
            user32 = ctypes.windll.user32
            MOD_CONTROL = 0x0002
            VK_F = 0x46
            WM_HOTKEY = 0x0312
            PM_REMOVE = 0x0001

            if not user32.RegisterHotKey(None, 1, MOD_CONTROL, VK_F):
                print("[warn] Could not register Ctrl+F hotkey (maybe already registered)")

            msg = wt.MSG()
            while not stop_event.is_set():
                # Process all queued messages non-blocking
                while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
                    if msg.message == WM_HOTKEY and msg.wParam == 1:
                        toggle_callback()
                time.sleep(0.01)
        except Exception as e:
            print(f"[warn] Hotkey listener error: {e}")
        finally:
            try:
                # Best-effort unregister
                import ctypes
                user32 = ctypes.windll.user32
                user32.UnregisterHotKey(None, 1)
            except Exception:
                pass

    t = threading.Thread(target=loop, name="pause-hotkey", daemon=True)
    t.start()
    return stop_event, t


def _estimate_brightness(bgr: np.ndarray) -> float:
    """Return mean brightness in [0,255] using Y channel (YCrCb)."""
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0]
    return float(np.mean(y))


def _apply_low_light_normalization(bgr: np.ndarray, clip: float = 2.0, grid: int = 8, gamma: float = 0.7) -> np.ndarray:
    """Improve visibility on dark frames: CLAHE on Y + gamma brighten (gamma<1 -> brighten)."""
    # CLAHE on Y channel
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=max(0.5, float(clip)), tileGridSize=(max(2, int(grid)), max(2, int(grid))))
    y_eq = clahe.apply(y)
    ycrcb_eq = cv2.merge((y_eq, cr, cb))
    out = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)
    # Gamma LUT
    gamma = max(0.1, float(gamma))
    inv = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv * 255 for i in range(256)]).astype("uint8")
    out = cv2.LUT(out, table)
    return out


def grab_region(region):
    """Capture a region of the screen using mss. region=(x,y,w,h)."""
    try:
        import mss
    except Exception as e:
        print("[error] mss not installed. Add it to requirements.txt and pip install.")
        raise e

    with mss.mss() as sct:
        monitor = sct.monitors[1]
        if region is None:
            bbox = {"left": monitor["left"], "top": monitor["top"], "width": monitor["width"], "height": monitor["height"]}
        else:
            x, y, w, h = region
            bbox = {"left": x, "top": y, "width": w, "height": h}
        img = np.array(sct.grab(bbox))
        # mss returns BGRA
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img


def preprocess(img: np.ndarray, imgsz: int) -> np.ndarray:
    # Resize keeping aspect to square by padding (letterbox) or just resize
    resized = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
    return resized


def crop_region(img: np.ndarray, region):
    """Crop a BGR frame by (x,y,w,h) if provided and within bounds."""
    if region is None:
        return img
    x, y, w, h = region
    h_img, w_img = img.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w_img, x + w)
    y1 = min(h_img, y + h)
    if x0 >= x1 or y0 >= y1:
        return img  # invalid crop; fallback to full frame
    return img[y0:y1, x0:x1]


def classify(model, device: str, img: np.ndarray):
    # Ultralytics YOLO classification expects path or numpy BGR with size imgsz
    # We already resized.
    results = model.predict(img, verbose=False, device=device)
    if not results:
        return None
    r = results[0]
    # r.probs.top1, r.names, r.probs.data
    probs = r.probs.data.tolist()
    names = r.names
    return probs, names


def main():
    args = parse_args()

    weights = Path(args.weights).resolve()
    if not weights.exists():
        print(f"[error] Weights not found: {weights}")
        sys.exit(1)
    model, device = load_model(str(weights), args.device)
    # Determine class index for 'bite'
    bite_index = None
    # Run a dummy forward to map names
    dummy = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
    res = model.predict(dummy, verbose=False, device=device)
    if res and res[0].names:
        for k, v in res[0].names.items():
            if v.lower() == "bite":
                bite_index = k
                break
    if bite_index is None:
        print("[error] Could not find class 'bite' in model names. Check training classes.")
        print("Model classes:", res[0].names if res else "(none)")
        sys.exit(1)

    consecutive = 0
    # Initialize last_trigger so first valid detection isn't delayed by cooldown
    last_trigger = time.time() - args.cooldown
    # no separate waiting state; direct call blocks until minigame finishes
    # Counter for fallback action when probability stays near zero
    zero_prob_frames = 0
    last_small_prob = None

    if args.show:
        cv2.namedWindow("bite-detector", cv2.WINDOW_NORMAL)

    print("[info] Starting loop. Press Ctrl+C to stop.")
    print("[info] Press Ctrl+F to pause/unpause detection.")

    paused = {"value": False}

    def _toggle_pause():
        paused["value"] = not paused["value"]
        state = "PAUSED" if paused["value"] else "RUNNING"
        print(f"[pause] Toggled -> {state}")

    hotkey_stop, hotkey_thread = _start_pause_hotkey_listener(_toggle_pause)

    # Video mode setup (optional)
    cap = None
    frame_interval = args.interval
    if args.video:
        video_path = Path(args.video).resolve()
        if not video_path.exists():
            print(f"[error] Video not found: {video_path}")
            sys.exit(1)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[error] Could not open video: {video_path}")
            sys.exit(1)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps and fps > 0:
            frame_interval = 1.0 / fps
        print(f"[info] Using video input: {video_path} (fps={fps or 'unknown'})")

    try:
        while True:
            start = time.time()
            if cap is not None:
                ret, frame = cap.read()
                if not ret:
                    print("[info] End of video.")
                    break
                # Optional region crop in video coordinates
                frame = crop_region(frame, args.region)
            else:
                frame = grab_region(args.region)

            if paused["value"]:
                # Still update window if show enabled (display last frame probability message)
                if args.show:
                    dummy_txt = "PAUSED"
                    dummy_img = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
                    disp = cv2.putText(dummy_img, dummy_txt, (10, args.imgsz // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow("bite-detector", disp)
                    if cv2.waitKey(1) & 0xFF == ord(args.quit_key):
                        break
                time.sleep(0.05)
                continue

            # Lighting normalization and adaptive thresholding (config-driven, no new CLI flags)
            try:
                from script import config as _cfg
            except Exception:
                _cfg = None

            brightness = _estimate_brightness(frame)
            is_dark = False
            if _cfg is not None:
                min_b = float(getattr(_cfg, 'LIGHT_BRIGHTNESS_THRESH', 60.0))
                if brightness < min_b:
                    is_dark = True
                if getattr(_cfg, 'ENABLE_LIGHT_NORMALIZATION', True) and is_dark:
                    clip = float(getattr(_cfg, 'LIGHT_CLAHE_CLIP', 2.0))
                    grid = int(getattr(_cfg, 'LIGHT_CLAHE_GRID', 8))
                    gamma = float(getattr(_cfg, 'LIGHT_GAMMA_BRIGHTEN', 0.7))
                    frame = _apply_low_light_normalization(frame, clip=clip, grid=grid, gamma=gamma)

            proc = preprocess(frame, args.imgsz)
            probs, names = classify(model, device, proc)
            if probs is None:
                continue
            bite_prob = probs[bite_index]

            # Effective thresholds when dark
            required_consecutive = args.consecutive
            eff_threshold = args.threshold
            if 'is_dark' in locals() and is_dark:
                bump_thr = 0.1
                bump_consec = 1
                if _cfg is not None:
                    bump_thr = float(getattr(_cfg, 'NIGHT_THRESHOLD_BUMP', bump_thr))
                    bump_consec = int(getattr(_cfg, 'NIGHT_CONSEC_BUMP', bump_consec))
                eff_threshold = 0.85
                # required_consecutive = max(1, args.consecutive + bump_consec)

            if bite_prob >= eff_threshold:
                consecutive += 1
            else:
                consecutive = max(0, consecutive - 1) if args.decay else 0

            if args.verbose:
                # basic rate limiting
                if not hasattr(main, "_frame_idx"):
                    main._frame_idx = 0
                main._frame_idx += 1
                if main._frame_idx % max(1, args.log_every) == 0:
                    extra = ""
                    if 'brightness' in locals():
                        extra = f" bright={brightness:.1f}{' DARK' if ('is_dark' in locals() and is_dark) else ''} thr={eff_threshold:.2f} req_consec={required_consecutive}"
                    print(f"[frame] p={bite_prob:.3f} consec={consecutive}/{required_consecutive}{extra}")

            # Fallback: only when p stays effectively constant and ~0 for many frames (avoid resetting when it varies)
            if bite_prob <= 0.001 and consecutive == 0:
                # consider "constant" if variation is tiny
                if last_small_prob is None:
                    zero_prob_frames = 1
                    last_small_prob = bite_prob
                else:
                    if abs(bite_prob - last_small_prob) <= 0.0001:
                        zero_prob_frames += 1
                    else:
                        # variation detected -> restart counter
                        zero_prob_frames = 1
                        last_small_prob = bite_prob
            else:
                zero_prob_frames = 0
                last_small_prob = None

            if zero_prob_frames >= 50:
                print("[fallback] ~50 frames with constant p≈0 and consec=0 -> short hold click (0.4s)")
                _hold_left_click(timelzm, move_to_center=False)
                zero_prob_frames = 0
                last_small_prob = None
                consecutive = 0
                time.sleep(2.0)  # brief pause after fallback

            now = time.time()
            if consecutive >= required_consecutive and (now - last_trigger) >= args.cooldown:
                if args.no_action:
                    print(f"[action] Bite detected (p={bite_prob:.2f}). no-action mode -> not pressing any key")
                else:
                    print(f"[action] Bite detected (p={bite_prob:.2f}). Starting minigame")
                    # Start minigame directly and wait until it finishes
                    rc = _run_minigame_direct()
                    print(f"[minigame] finished with code {rc}")
                    # Post-minigame: instant controlled hold (no pre-wait) then settle pause
                    print("[info] Post-minigame: hold left click 0.7s at region center")
                    time.sleep(3.0)  # brief pause before action
                    _hold_left_click(timelzm, move_to_center=False)
                    time.sleep(1.0)
                last_trigger = time.time()
                consecutive = 0  # reset after action

            if args.show:
                txt = f"bite_prob={bite_prob:.3f} consec={consecutive} cooldown={max(0, args.cooldown - (now - last_trigger)):.1f}"
                disp = cv2.putText(proc.copy(), txt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.imshow("bite-detector", disp)
                if cv2.waitKey(1) & 0xFF == ord(args.quit_key):
                    break

            # Sleep to maintain interval (video uses fps if available)
            elapsed = time.time() - start
            target_interval = frame_interval if cap is not None else args.interval
            to_sleep = max(0.0, target_interval - elapsed)
            time.sleep(to_sleep)
    except KeyboardInterrupt:
        print("[info] Stopped by user")
    finally:
        try:
            hotkey_stop.set()
        except Exception:
            pass
        if cap is not None:
            cap.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
