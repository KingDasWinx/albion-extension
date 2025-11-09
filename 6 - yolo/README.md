# Fishing Bite Detector (YOLOv8 Classification)

Detect when the fishing bobber goes down ("bite") in your game by classifying frames of your screen using a lightweight YOLOv8 classification model.

## 1. Dataset Structure
Place your images in class folders:
```
6 - yolo/
  train/
    bite/  # images of the moment the bobber is down (fish biting)
    idle/  # images of idle/normal state
```
You already have augmented images in those folders.

## 2. Environment Setup (Windows PowerShell, RTX 3050)
It's recommended to use a Python virtual environment.

```powershell
# From the project root (6 - yolo)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### CUDA-enabled Torch for RTX 3050
If the default `pip install torch` gave you CPU-only, install the CUDA build matching your installed CUDA runtime (or use the prebuilt wheels). For most users (CUDA 12):
```powershell
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```
If that fails, check: https://pytorch.org/get-started/locally/

Verify:
```powershell
python - <<'PY'
import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())
PY
```

## 3. Train the Classification Model
This script will split your existing `train/` folder into `dataset_cls/train` and `dataset_cls/val`.

```powershell
python .\train_yolo_cls.py --source train --out dataset_cls --model yolov8n-cls.pt --epochs 25 --batch 64 --imgsz 224 --name fish_cls --device auto
```
Notes:
- `--device auto` picks GPU if available.
- Adjust `--epochs` for accuracy vs time; start with 20–30.
- Model variants: `yolov8s-cls.pt`, `yolov8m-cls.pt` (bigger = slower, maybe more accurate).

Outputs saved under `runs/classify/fish_cls/`.
Best weights: `runs/classify/fish_cls/weights/best.pt`.

## 4. Realtime Detection
After training, run the realtime script. Provide a focused region for faster inference (x y w h).

Example capturing a 300x300 box at (1000, 400):
```powershell
python .\detect_realtime.py --weights runs/classify/fish_cls/weights/best.pt --region 1000 400 300 300 --threshold 0.8 --consecutive 3 --cooldown 2 --action space --show
```
Explanation:
- `--threshold 0.8`: probability required for class `bite`.
- `--consecutive 3`: need 3 frames in a row above threshold.
- `--cooldown 2`: wait 2s before another key press.
- `--action space`: key to press when bite detected.
- `--show`: show debug window.

Quit the window with the `q` key or Ctrl+C in terminal.

### Testar com vídeo (offline)
Você pode testar o classificador com um arquivo de vídeo em vez da tela ao vivo:
```powershell
python .\detect_realtime.py --weights runs/classify/fish_cls/weights/best.pt --video .\video.mp4 --threshold 0.8 --consecutive 3 --show --no-action
```
Notas:
- `--video` usa a taxa de quadros do arquivo para o intervalo; se o FPS não estiver disponível, usa `--interval`.
- `--region` (se fornecida) recorta em coordenadas do vídeo.
- `--no-action` evita pressionar teclas durante os testes.

## 5. Choosing Region Coordinates
Use a screen capture tool (Win+Shift+S) or simple printouts to estimate the region around the bobber. Smaller region = faster classification.
If omitted, the whole screen is used (slower).

## 6. Improving Accuracy
- Increase image diversity (lighting, slight motion blur, different backgrounds).
- Balance class counts if heavily skewed.
- Try `yolov8s-cls.pt` if nano (`n`) underfits.
- Increase `--imgsz` (e.g., 256) maybe improves detail.

## 7. Logging & Monitoring
Training metrics in `runs/classify/.../results.csv` and TensorBoard logs can be enabled via:
```powershell
pip install tensorboard
python -m tensorboard --logdir runs/classify
```

## 8. Troubleshooting
| Issue | Fix |
|-------|-----|
| Torch CUDA False | Install correct CUDA wheel (see section 2) |
| ultralytics not found | `pip install ultralytics` |
| Slow inference | Use region, smaller model, lower imgsz |
| Missed bites | Lower threshold or consecutive; gather more training samples |
| False positives | Raise threshold, increase consecutive, add more idle samples |

## 9. Next Steps / Ideas
- Move from classification to object detection if you want auto-locate bobber (annotate bounding boxes and train `yolov8n.pt`).
- Add auditory alert instead of key press.
- Auto-calibrate region by clicking on bobber once (add a small Tkinter/UI helper).

## 10. License / Game Fair Use
Ensure this automation complies with the game's Terms of Service. Use responsibly.

---
Happy fishing automation! 🎣
