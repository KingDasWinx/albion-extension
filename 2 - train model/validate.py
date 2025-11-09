import os
import platform
import subprocess
import sys
import time
from typing import Optional

import torch


def run_cmd(cmd: list[str]) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except FileNotFoundError:
        return 127, "", "not found"


def check_nvidia_smi() -> dict:
    info = {"available": False}
    code, out, err = run_cmd([
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,cuda_version",
        "--format=csv,noheader",
    ])
    if code == 0 and out:
        info["available"] = True
        gpus = []
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            # Expected: name, driver, mem_total, cuda_version (may be "N/A")
            if len(parts) >= 4:
                gpus.append({
                    "name": parts[0],
                    "driver": parts[1],
                    "memory_total": parts[2],
                    "cuda_version": parts[3],
                })
        info["gpus"] = gpus
        return info

    # Fallback: simple call to nvidia-smi without query
    code2, out2, err2 = run_cmd(["nvidia-smi"]) 
    info["available"] = code2 == 0 and bool(out2)
    info["raw"] = out2
    return info


def torch_cuda_summary() -> dict:
    summary = {
        "torch_version": torch.__version__,
        "torch_cuda_compiled": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.enabled else None,
    }
    if summary["cuda_available"]:
        devices = []
        for i in range(summary["cuda_device_count"]):
            props = torch.cuda.get_device_properties(i)
            devices.append({
                "index": i,
                "name": props.name,
                "total_memory_bytes": props.total_memory,
                "capability": f"{props.major}.{props.minor}",
            })
        summary["devices"] = devices
    return summary


def quick_gpu_test() -> Optional[str]:
    if not torch.cuda.is_available():
        return "CUDA indisponível no PyTorch (torch.cuda.is_available() == False)."
    try:
        dev = torch.device("cuda:0")
        torch.cuda.set_device(dev)
        x = torch.randn(1024, 1024, device=dev)
        t0 = time.time()
        y = x @ x
        torch.cuda.synchronize()
        dt = time.time() - t0
        return f"Teste simples de matmul na GPU OK: shape={tuple(y.shape)}, tempo={dt:.4f}s."
    except Exception as e:
        return f"Falha ao executar teste na GPU: {e}"


def main():
    print("=== Validação de Ambiente PyTorch/CUDA ===")
    print(f"Python: {platform.python_version()} | Executável: {sys.executable}")
    print(f"Sistema: {platform.system()} {platform.release()} | Arquitetura: {platform.machine()}")

    torch_info = torch_cuda_summary()
    print(f"Torch: {torch_info['torch_version']} | Compilado com CUDA: {torch_info['torch_cuda_compiled']}")
    print(f"CUDA disponível: {torch_info['cuda_available']} | GPUs detectadas: {torch_info['cuda_device_count']}")
    if torch_info["cudnn_enabled"]:
        print(f"cuDNN habilitado: True | Versão: {torch_info['cudnn_version']}")
    else:
        print("cuDNN habilitado: False")

    if torch_info.get("devices"):
        for d in torch_info["devices"]:
            mem_gb = d["total_memory_bytes"] / (1024**3)
            print(f"GPU[{d['index']}] {d['name']} | Memória: {mem_gb:.2f} GB | Compute Capability: {d['capability']}")

    smi = check_nvidia_smi()
    print(f"nvidia-smi disponível: {smi['available']}")
    if smi.get("gpus"):
        for i, g in enumerate(smi["gpus"]):
            print(f"nvidia-smi GPU[{i}]: {g['name']} | Driver: {g['driver']} | Memória: {g['memory_total']} | CUDA: {g['cuda_version']}")
    elif smi.get("raw"):
        print("Saída nvidia-smi (resumo):")
        print("\n".join(smi["raw"].splitlines()[:6]))

    print("\n=== Teste rápido de GPU ===")
    test_msg = quick_gpu_test()
    print(test_msg)

    print("\n=== Recomendações ===")
    if not torch_info["cuda_available"]:
        if torch_info["torch_cuda_compiled"] is None:
            print("- O pacote PyTorch instalado é CPU-only. Reinstale com build CUDA.")
            print("  Ex.: pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision")
            print("  Se seu driver for antigo, use cu121 (ou cu118):")
            print("  Ex.: pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision")
        else:
            print("- PyTorch foi compilado com CUDA, mas não conseguiu inicializar a GPU.")
            print("  Ações sugeridas:")
            print("  1) Atualize o driver NVIDIA (Studio/Game Ready) para versão compatível.")
            print("  2) Garanta que a GPU NVIDIA está ativa e o Windows está usando-a.")
            print("  3) Verifique conflitos com múltiplos Python/venvs. Rode este script no mesmo .venv do treino.")
            print("  4) Se continuar falhando, reinstale torch/torchvision do índice CUDA que corresponde ao seu driver.")
    else:
        print("- CUDA disponível, você pode treinar com --device cuda.")
        print("- Mantenha --amp ativo para acelerar com float16.")


if __name__ == "__main__":
    main()