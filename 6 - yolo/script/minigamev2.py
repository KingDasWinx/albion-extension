import argparse
import time
import os
import sys

import numpy as np
import cv2
import mss
import pyautogui


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "MinigameV2: replica a lógica do Fisherman para o minigame usando template matching do bobber."
        )
    )
    parser.add_argument(
        "--region",
        nargs=4,
        type=int,
        metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"),
        required=True,
        help="Região da tela para captura: LEFT TOP WIDTH HEIGHT (ex.: 837 533 243 36)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Limiar de correlação do template (0-1). Padrão: 0.5",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Taxa de atualização desejada. Controla o tempo de espera entre frames.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Exibe janelas de debug com ROI e overlay do bobber.",
    )
    parser.add_argument(
        "--focus-window",
        type=str,
        default=None,
        help="Título parcial da janela do jogo para focar (ex.: Albion)",
    )
    return parser.parse_args()


def _to_mss_bbox(left: int, top: int, width: int, height: int):
    return {"left": left, "top": top, "width": width, "height": height}


def _prepare_frame(img_bgra: np.ndarray) -> np.ndarray:
    # Replica as conversões do Fisherman.py
    base = np.flip(img_bgra[:, :, :3], 2)
    base = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
    return base


def _prepare_template(template_bgr: np.ndarray) -> np.ndarray:
    bobber = np.array(template_bgr, dtype=np.uint8)
    bobber = np.flip(bobber[:, :, :3], 2)
    bobber = cv2.cvtColor(bobber, cv2.COLOR_RGB2BGR)
    return bobber


def detect_bobber(sct: mss.mss, bbox: dict, template: np.ndarray, threshold: float):
    start_time = time.time()
    frame_bgra = np.array(sct.grab(bbox))
    frame = _prepare_frame(frame_bgra)

    # Template matching (TM_CCOEFF_NORMED) como no script original
    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    valid = max_val > threshold
    elapsed = time.time() - start_time

    return valid, max_loc, frame.shape[1], frame, max_val, elapsed


def main():
    args = parse_args()
    left, top, width, height = args.region
    bbox = _to_mss_bbox(left, top, width, height)
    threshold = args.threshold
    target_dt = 1.0 / max(1, args.fps)

    # Carrega template do bobber do diretório atual
    base_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(base_dir, "bobber.png")
    if not os.path.isfile(template_path):
        print(f"[erro] bobber.png não encontrado em: {template_path}")
        sys.exit(1)
    # Lê o template como BGR
    template_bgr = cv2.imread(template_path)
    template = _prepare_template(template_bgr)

    # Focar janela do jogo se solicitado
    if args.focus_window:
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32

            titles = []

            EnumWindows = user32.EnumWindows
            EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
            GetWindowTextW = user32.GetWindowTextW
            GetWindowTextLengthW = user32.GetWindowTextLengthW
            IsWindowVisible = user32.IsWindowVisible
            ShowWindow = user32.ShowWindow
            SetForegroundWindow = user32.SetForegroundWindow

            target = None

            def enum_proc(hWnd, lParam):
                if IsWindowVisible(hWnd):
                    length = GetWindowTextLengthW(hWnd)
                    if length > 0:
                        buff = ctypes.create_unicode_buffer(length + 1)
                        GetWindowTextW(hWnd, buff, length + 1)
                        title = buff.value
                        if args.focus_window.lower() in title.lower():
                            nonlocal target
                            target = hWnd
                            return False  # stop enum
                return True

            EnumWindows(EnumWindowsProc(enum_proc), 0)
            if target:
                # Restaurar e focar
                SW_RESTORE = 9
                ShowWindow(target, SW_RESTORE)
                SetForegroundWindow(target)
                time.sleep(0.1)
            else:
                print(f"[aviso] Janela com título contendo '{args.focus_window}' não encontrada.")
        except Exception as e:
            print(f"[aviso] Falha ao focar janela: {e}")

    # Sinal inicial de clique como no Fisherman (mouseDown/Up) antes de procurar o bobber
    pyautogui.mouseDown()
    pyautogui.mouseUp()
    time.sleep(0.5)

    with mss.mss() as sct:
        # Busca inicial: aguarda o bobber aparecer
        valid, loc, size_w, frame, score, elapsed = detect_bobber(sct, bbox, template, threshold)
        if not valid:
            # Se não foi encontrado, apenas encerra (igual lógica de fallback do Fisherman que volta a CASTING)
            print(f"Bobber não encontrado (score={score:.3f}, t={elapsed:.3f}s). Encerrando.")
            return

        print(f"Bobber encontrado! score={score:.3f}, t={elapsed:.3f}s")

        # Loop de minigame: mantém o bobber à esquerda do centro com mouseDown, direita com mouseUp
        while True:
            start_loop = time.time()
            valid, loc, size_w, frame, score, elapsed = detect_bobber(sct, bbox, template, threshold)
            if not valid:
                pyautogui.mouseUp()
                print(f"Bobber perdido (score={score:.3f}, t={elapsed:.3f}s). Saindo do minigame.")
                break

            # Decisão exatamente como no original
            if loc[0] < (size_w / 2):
                pyautogui.mouseDown()
            else:
                pyautogui.mouseUp()

            if args.debug:
                # Overlay simples: centro e ponto do bobber
                overlay = frame.copy()
                center_x = int(size_w / 2)
                cv2.line(overlay, (center_x, int(height / 2)), (center_x, int(height / 2)), (0, 255, 255), 1)
                cv2.circle(overlay, (loc[0], int(height / 2)), 4, (0, 0, 255), -1)
                cv2.putText(
                    overlay,
                    f"score={score:.3f}",
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow("ROI", overlay)
                # Exibe result simplificado como máscara da posição do bobber
                # Nota: não temos o 'Mask' original por usar template matching; mostramos o frame cinza
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imshow("Mask", gray)

                if cv2.waitKey(1) & 0xFF == 27:
                    # ESC para sair
                    pyautogui.mouseUp()
                    break

            # Controla FPS
            dt = time.time() - start_loop
            if dt < target_dt:
                time.sleep(target_dt - dt)

    if args.debug:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()