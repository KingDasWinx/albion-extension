import argparse
import subprocess
import sys
from pathlib import Path
import ctypes
from ctypes import wintypes
import threading
import time


MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
WM_HOTKEY = 0x0312


def vk_from_key(key: str) -> int:
    k = key.lower()
    if len(k) == 1 and 'a' <= k <= 'z':
        return ord(k.upper())
    fkeys = {f'f{i}': 0x70 + (i - 1) for i in range(1, 13)}
    if k in fkeys:
        return fkeys[k]
    digits = {str(i): 0x30 + i for i in range(0, 10)}
    if k in digits:
        return digits[k]
    # Default: D
    return ord('D')


def parse_hotkey(spec: str) -> tuple[int, int]:
    parts = [p.strip() for p in spec.lower().replace('+', '-').split('-') if p.strip()]
    mods = 0
    key = None
    for p in parts:
        if p in ('ctrl', 'control'):
            mods |= MOD_CONTROL
        elif p == 'alt':
            mods |= MOD_ALT
        elif p == 'shift':
            mods |= MOD_SHIFT
        elif p in ('win', 'super'):
            mods |= MOD_WIN
        else:
            key = p
    vk = vk_from_key(key or 'd')
    return mods, vk


def main():
    parser = argparse.ArgumentParser(description='Runner com hotkey para iniciar o minigame (Ctrl+D por padrão).')
    parser.add_argument('--hotkey', type=str, default='ctrl+d', help='Hotkey para iniciar o minigame (ex.: ctrl+d).')
    parser.add_argument('--exit-hotkey', type=str, default='ctrl+shift+d', help='Hotkey para encerrar o runner.')
    parser.add_argument('--status-file', type=str, default='minigame.lock', help='Arquivo de status criado enquanto o minigame estiver rodando')
    # Parâmetros pass-through para o minigame
    parser.add_argument('--monitor', type=int, default=1)
    parser.add_argument('--region', type=int, nargs=4, metavar=('X', 'Y', 'W', 'H'))
    parser.add_argument('--set-roi', action='store_true')
    parser.add_argument('--tune-hsv', action='store_true')
    parser.add_argument('--invert-mask', action='store_true', help='Inverte a máscara HSV.')
    parser.add_argument('--detect-smallest', action='store_true', help='Detecta o menor contorno (boia).')
    parser.add_argument('--min-area', type=int, default=80, help='Área mínima do contorno.')
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--hide-cursor', action='store_true')
    parser.add_argument('--green-hsv', type=str, default='35,50,50:85,255,255')
    parser.add_argument('--tolerance', type=int, default=6)
    parser.add_argument('--pulse-ms', type=int, default=40)
    parser.add_argument('--kp-ms', type=float, default=2.0)
    parser.add_argument('--kd-ms', type=float, default=0.0)
    parser.add_argument('--max-seconds', type=int, default=60)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--focus-window', type=str, default=None, help='Título parcial da janela do jogo para focar (ex.: Albion).')
    # Opções do minigamev2
    parser.add_argument('--threshold', type=float, default=0.5, help='Limiar de correlação do template (minigamev2).')
    # Escolha do script
    parser.add_argument('--script', type=str, default='auto', help="Qual script rodar: 'auto' (padrão), 'v2', 'v1' ou caminho.")
    args = parser.parse_args()

    # Seleciona script: v2 por padrão se existir (ajustado para caminho local)
    base_dir = Path(__file__).parent
    v2_path = base_dir / 'minigamev2.py'
    v1_path = base_dir / 'minigame.py'
    if args.script.lower() == 'v2':
        script_path = v2_path
    elif args.script.lower() == 'v1':
        script_path = v1_path
    elif args.script.lower() == 'auto':
        script_path = v2_path if v2_path.exists() else v1_path
    else:
        # Caminho customizado
        script_path = Path(args.script)

    child_args = [sys.executable, str(script_path)]

    # Montar args para o minigame selecionado
    if script_path.name == 'minigamev2.py':
        # Args do v2
        if args.region is not None:
            child_args += ['--region'] + [str(v) for v in args.region]
        child_args += ['--threshold', str(args.threshold), '--fps', str(args.fps)]
        if args.debug:
            child_args.append('--debug')
        if args.focus_window:
            child_args += ['--focus-window', args.focus_window]
    else:
        # Args do v1 (minigame.py)
        child_args += [
            '--monitor', str(args.monitor),
            '--fps', str(args.fps),
            '--tolerance', str(args.tolerance),
            '--pulse-ms', str(args.pulse_ms),
            '--kp-ms', str(args.kp_ms),
            '--kd-ms', str(args.kd_ms),
            '--max-seconds', str(args.max_seconds),
            '--min-area', str(args.min_area),
            '--green-hsv', args.green_hsv,
        ]
        if args.region is not None:
            child_args += ['--region'] + [str(v) for v in args.region]
        if args.set_roi:
            child_args.append('--set-roi')
        if args.tune_hsv:
            child_args.append('--tune-hsv')
        if args.invert_mask:
            child_args.append('--invert-mask')
        if args.detect_smallest:
            child_args.append('--detect-smallest')
        if args.hide_cursor:
            child_args.append('--hide-cursor')
        if args.debug:
            child_args.append('--debug')
        if args.focus_window:
            child_args += ['--focus-window', args.focus_window]
    if args.region is not None:
        child_args += ['--region'] + [str(v) for v in args.region]
    if args.set_roi:
        child_args.append('--set-roi')
    if args.tune_hsv:
        child_args.append('--tune-hsv')
    if args.invert_mask:
        child_args.append('--invert-mask')
    if args.detect_smallest:
        child_args.append('--detect-smallest')
    if args.hide_cursor:
        child_args.append('--hide-cursor')
    if args.debug:
        child_args.append('--debug')
    if args.focus_window:
        child_args += ['--focus-window', args.focus_window]

    # Registrar hotkeys
    mods_start, vk_start = parse_hotkey(args.hotkey)
    mods_exit, vk_exit = parse_hotkey(args.exit_hotkey)
    user32 = ctypes.windll.user32
    if not user32.RegisterHotKey(None, 1, mods_start, vk_start):
        print(f'Falha ao registrar hotkey de início: {args.hotkey}')
        return
    if not user32.RegisterHotKey(None, 2, mods_exit, vk_exit):
        print(f'Falha ao registrar hotkey de sair: {args.exit_hotkey}')
        user32.UnregisterHotKey(None, 1)
        return

    print(f'Runner ativo. Pressione {args.hotkey.upper()} para iniciar o minigame.')
    print(f'Pressione {args.exit_hotkey.upper()} para encerrar este runner.')

    proc = None
    status_path = Path(__file__).parent / args.status_file

    def ensure_removed():
        try:
            if status_path.exists():
                status_path.unlink(missing_ok=True)
        except Exception:
            pass

    def watcher_thread(p: subprocess.Popen):
        # Aguarda o término do minigame e remove o lock
        try:
            p.wait()
        except Exception:
            pass
        ensure_removed()

    try:
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), 0, 0, 0) != 0:
            if msg.message == WM_HOTKEY:
                hot_id = msg.wParam
                if hot_id == 1:
                    if proc is None or proc.poll() is not None:
                        try:
                            proc = subprocess.Popen(child_args)
                            print('Minigame iniciado.')
                            # Cria lock file
                            try:
                                status_path.write_text('running', encoding='utf-8')
                            except Exception:
                                pass
                            # Inicia watcher para limpar lock ao terminar
                            threading.Thread(target=watcher_thread, args=(proc,), daemon=True).start()
                        except Exception as e:
                            print(f'Falha ao iniciar minigame: {e}')
                    else:
                        print('Minigame já está em execução.')
                elif hot_id == 2:
                    # Encerrar runner e minigame (se estiver rodando)
                    if proc is not None and proc.poll() is None:
                        try:
                            proc.terminate()
                            print('Minigame encerrado.')
                        except Exception:
                            pass
                    # Remove lock
                    ensure_removed()
                    break
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
    finally:
        user32.UnregisterHotKey(None, 1)
        user32.UnregisterHotKey(None, 2)
        ensure_removed()


def _build_child_args(
    script: str = 'v2',
    region: tuple | None = None,
    threshold: float = 0.5,
    fps: int = 30,
    debug: bool = False,
    focus_window: str | None = None,
    monitor: int = 1,
    set_roi: bool = False,
    tune_hsv: bool = False,
    invert_mask: bool = False,
    detect_smallest: bool = False,
    hide_cursor: bool = False,
    tolerance: int = 6,
    pulse_ms: int = 40,
    kp_ms: float = 2.0,
    kd_ms: float = 0.0,
    max_seconds: int = 60,
    green_hsv: str = '35,50,50:85,255,255',
):
    base_dir = Path(__file__).parent
    v2_path = base_dir / 'minigamev2.py'
    v1_path = base_dir / 'minigame.py'

    script_l = (script or 'auto').lower()
    if script_l == 'v2':
        script_path = v2_path
    elif script_l == 'v1':
        script_path = v1_path
    else:
        script_path = v2_path if v2_path.exists() else v1_path

    args = [sys.executable, str(script_path)]
    if script_path.name == 'minigamev2.py':
        if region is not None:
            args += ['--region'] + [str(v) for v in region]
        args += ['--threshold', str(threshold), '--fps', str(fps)]
        if debug:
            args.append('--debug')
        if focus_window:
            args += ['--focus-window', focus_window]
    else:
        args += [
            '--monitor', str(monitor),
            '--fps', str(fps),
            '--tolerance', str(tolerance),
            '--pulse-ms', str(pulse_ms),
            '--kp-ms', str(kp_ms),
            '--kd-ms', str(kd_ms),
            '--max-seconds', str(max_seconds),
            '--min-area', str(80),
            '--green-hsv', green_hsv,
        ]
        if region is not None:
            args += ['--region'] + [str(v) for v in region]
        if set_roi:
            args.append('--set-roi')
        if tune_hsv:
            args.append('--tune-hsv')
        if invert_mask:
            args.append('--invert-mask')
        if detect_smallest:
            args.append('--detect-smallest')
        if hide_cursor:
            args.append('--hide-cursor')
        if debug:
            args.append('--debug')
        if focus_window:
            args += ['--focus-window', focus_window]
    return args


def run_minigame(
    script: str = 'v2',
    region: tuple | None = None,
    threshold: float = 0.5,
    fps: int = 30,
    debug: bool = False,
    focus_window: str | None = None,
):
    """Start the minigame process and block until it finishes. Returns process returncode."""
    child_args = _build_child_args(
        script=script,
        region=region,
        threshold=threshold,
        fps=fps,
        debug=debug,
        focus_window=focus_window,
    )
    try:
        p = subprocess.Popen(child_args)
        p.wait()
        return p.returncode
    except Exception as e:
        print(f"Falha ao iniciar/aguardar minigame: {e}")
        return -1


if __name__ == '__main__':
    main()