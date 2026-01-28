import os
import argparse
import time
import io
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from concurrent.futures import ThreadPoolExecutor


def list_images(input_dir, exts=(".png", ".jpg", ".jpeg", ".bmp", ".webp")):
    paths = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in exts:
                paths.append(p)
    return sorted(paths)


def rotate_and_crop(img, angle):
    base_w, base_h = img.size
    rotated = img.rotate(angle, resample=Image.BICUBIC, expand=True)
    w, h = rotated.size
    left = max(0, (w - base_w) // 2)
    top = max(0, (h - base_h) // 2)
    right = min(w, left + base_w)
    bottom = min(h, top + base_h)
    crop = rotated.crop((left, top, right, bottom))
    if crop.size != (base_w, base_h):
        crop = crop.resize((base_w, base_h), resample=Image.BICUBIC)
    return crop


def to_white_background(img):
    if "A" in img.getbands():
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, img).convert("RGB")
    return img


def random_resize_with_pad_or_crop(img, rng):
    w0, h0 = img.size
    scale = rng.uniform(0.85, 1.25)
    new_w = max(1, int(round(w0 * scale)))
    new_h = max(1, int(round(h0 * scale)))
    resized = img.resize((new_w, new_h), resample=Image.BICUBIC)
    if new_w >= w0 and new_h >= h0:
        max_x = new_w - w0
        max_y = new_h - h0
        left = rng.randint(0, max(0, max_x))
        top = rng.randint(0, max(0, max_y))
        cropped = resized.crop((left, top, left + w0, top + h0))
        return cropped
    else:
        if "A" in img.getbands():
            background = Image.new("RGBA", (w0, h0), (0, 0, 0, 0))
        else:
            bg_color = tuple(rng.randint(0, 20) for _ in range(3))
            background = Image.new("RGB", (w0, h0), bg_color)
        max_x = w0 - new_w
        max_y = h0 - new_h
        left = rng.randint(0, max(0, max_x))
        top = rng.randint(0, max(0, max_y))
        background.paste(resized, (left, top))
        return background


def color_jitter(img, rng):
    b = rng.uniform(0.6, 1.4)
    c = rng.uniform(0.6, 1.4)
    s = rng.uniform(0.6, 1.4)
    if "A" in img.getbands():
        a = img.getchannel("A")
        rgb = img.convert("RGB")
        rgb = ImageEnhance.Brightness(rgb).enhance(b)
        rgb = ImageEnhance.Contrast(rgb).enhance(c)
        rgb = ImageEnhance.Color(rgb).enhance(s)
        r, g, bch = rgb.split()
        return Image.merge("RGBA", (r, g, bch, a))
    else:
        img = ImageEnhance.Brightness(img).enhance(b)
        img = ImageEnhance.Contrast(img).enhance(c)
        img = ImageEnhance.Color(img).enhance(s)
        return img


def adjust_hsv(img, rng):
    if "A" in img.getbands():
        a = img.getchannel("A")
        rgb = img.convert("RGB")
        hsv = rgb.convert("HSV")
        arr = np.array(hsv).astype(np.int16)
        h_shift = rng.randint(-40, 40)
        s_scale = rng.uniform(0.7, 1.3)
        v_scale = rng.uniform(0.7, 1.3)
        arr[..., 0] = (arr[..., 0] + h_shift) % 256
        arr[..., 1] = np.clip(np.round(arr[..., 1] * s_scale), 0, 255)
        arr[..., 2] = np.clip(np.round(arr[..., 2] * v_scale), 0, 255)
        out_rgb = Image.fromarray(arr.astype(np.uint8), mode="HSV").convert("RGB")
        r, g, b = out_rgb.split()
        return Image.merge("RGBA", (r, g, b, a))
    else:
        hsv = img.convert("HSV")
        arr = np.array(hsv).astype(np.int16)
        h_shift = rng.randint(-40, 40)
        s_scale = rng.uniform(0.7, 1.3)
        v_scale = rng.uniform(0.7, 1.3)
        arr[..., 0] = (arr[..., 0] + h_shift) % 256
        arr[..., 1] = np.clip(np.round(arr[..., 1] * s_scale), 0, 255)
        arr[..., 2] = np.clip(np.round(arr[..., 2] * v_scale), 0, 255)
        out = Image.fromarray(arr.astype(np.uint8), mode="HSV").convert("RGB")
        return out


def add_gaussian_noise(img, rs):
    arr = np.array(img).astype(np.float32)
    sigma = rs.uniform(5.0, 25.0)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        rgb = arr[..., :3]
        a = arr[..., 3:4]
        noise = rs.normal(0.0, sigma, rgb.shape).astype(np.float32)
        rgb = np.clip(rgb + noise, 0, 255).astype(np.uint8)
        out = np.concatenate([rgb, a.astype(np.uint8)], axis=-1)
        return Image.fromarray(out, mode="RGBA")
    else:
        noise = rs.normal(0.0, sigma, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)


def blur_or_sharpen(img, rng):
    if "A" in img.getbands():
        a = img.getchannel("A")
        rgb = img.convert("RGB")
        if rng.random() < 0.5:
            radius = rng.uniform(0.5, 2.0)
            rgb = rgb.filter(ImageFilter.GaussianBlur(radius=radius))
        else:
            radius = rng.uniform(1.0, 2.0)
            percent = rng.randint(80, 180)
            threshold = rng.randint(0, 5)
            rgb = rgb.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))
        r, g, b = rgb.split()
        return Image.merge("RGBA", (r, g, b, a))
    else:
        if rng.random() < 0.5:
            radius = rng.uniform(0.5, 2.0)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        else:
            radius = rng.uniform(1.0, 2.0)
            percent = rng.randint(80, 180)
            threshold = rng.randint(0, 5)
            return img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))


def adjust_gamma(img, rng):
    gamma = rng.uniform(0.7, 1.5)
    lut = [min(255, max(0, int(round(255.0 * ((i / 255.0) ** gamma))))) for i in range(256)]
    if "A" in img.getbands():
        a = img.getchannel("A")
        rgb = img.convert("RGB").point(lut * 3)
        r, g, b = rgb.split()
        return Image.merge("RGBA", (r, g, b, a))
    else:
        return img.point(lut * 3)


def jpeg_artifact(img, rng):
    q = rng.randint(25, 95)
    buf = io.BytesIO()
    if "A" in img.getbands():
        a = img.getchannel("A")
        rgb = img.convert("RGB")
        rgb.save(buf, format="JPEG", quality=q, subsampling=2)
        buf.seek(0)
        out_rgb = Image.open(buf).convert("RGB")
        buf.close()
        r, g, b = out_rgb.split()
        return Image.merge("RGBA", (r, g, b, a))
    else:
        img.save(buf, format="JPEG", quality=q, subsampling=2)
        buf.seek(0)
        out = Image.open(buf).convert("RGB")
        buf.close()
        return out


def maybe_grayscale(img, rng):
    if rng.random() < 0.2:
        if "A" in img.getbands():
            a = img.getchannel("A")
            rgb = img.convert("RGB").convert("L").convert("RGB")
            r, g, b = rgb.split()
            return Image.merge("RGBA", (r, g, b, a))
        else:
            return img.convert("L").convert("RGB")
    return img


def random_flip(img, rng):
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.2:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    return img


def random_affine_shear(img, rng):
    w, h = img.size
    shear_x = math.tan(math.radians(rng.uniform(-8, 8)))
    shear_y = math.tan(math.radians(rng.uniform(-8, 8)))
    return img.transform((w, h), Image.AFFINE, (1, shear_x, 0, shear_y, 1, 0), resample=Image.BICUBIC)


def augment_once(img, rng, rs):
    img = img.convert("RGBA") if "A" in img.getbands() else img.convert("RGB")
    # Todas as operações são closures aceitando apenas (img), com rng/rs capturados
    ops = [
        lambda im: rotate_and_crop(im, rng.uniform(-30, 30)),
        lambda im: random_flip(im, rng),
        lambda im: random_resize_with_pad_or_crop(im, rng),
        lambda im: color_jitter(im, rng),
        lambda im: adjust_hsv(im, rng),
        lambda im: add_gaussian_noise(im, rs),
        lambda im: blur_or_sharpen(im, rng),
        lambda im: adjust_gamma(im, rng),
        lambda im: jpeg_artifact(im, rng),
        lambda im: random_affine_shear(im, rng),
        lambda im: maybe_grayscale(im, rng),
    ]
    k = rng.randint(4, 7)
    chosen = rng.sample(ops, k)
    out = img
    for op in chosen:
        out = op(out)
    return out


def process_images(input_dir, output_dir, per_image, seed, out_format, type_class=None, default=False, max_content=None, workers=None):
    paths = list_images(input_dir)
    if not paths:
        print(f"Nenhuma imagem encontrada em: {input_dir}")
        return
    os.makedirs(output_dir, exist_ok=True)
    start = time.time()
    total_out = 0
    groups = {}
    for p in paths:
        if default:
            key = p.parent.relative_to(Path(input_dir))
        elif type_class:
            key = Path(type_class)
        else:
            key = Path(p.stem)
        groups.setdefault(key, []).append(p)

    if not workers or workers <= 0:
        workers = max(1, os.cpu_count() or 1)

    jobs = []
    idx = 0

    for key, imgs in groups.items():
        out_dir = Path(output_dir) / key
        out_dir.mkdir(parents=True, exist_ok=True)
        if max_content is not None:
            total_needed = max(0, int(max_content))
            n = len(imgs)
            if n == 0 or total_needed == 0:
                continue
            base_repeat = total_needed // n
            remainder = total_needed % n
            counts = [base_repeat + (1 if i < remainder else 0) for i in range(n)]
            for img_path, count in zip(imgs, counts):
                base = Path(img_path).stem
                for i in range(count):
                    jobs.append((img_path, out_dir, base, i, out_format, seed + idx))
                    idx += 1
        else:
            for img_path in imgs:
                base = Path(img_path).stem
                for i in range(per_image):
                    jobs.append((img_path, out_dir, base, i, out_format, seed + idx))
                    idx += 1

    def _run_job(img_path, out_dir, base, i, out_format, job_seed):
        try:
            with Image.open(img_path) as img:
                rng = random.Random(job_seed)
                rs = np.random.RandomState(job_seed)
                aug = augment_once(img, rng, rs)
                aug = to_white_background(aug)
                name = f"{base}__aug_{i:05d}.{out_format}"
                aug.save(out_dir / name)
                return True
        except Exception as e:
            print(f"Falha ao processar {img_path}: {e}")
            return False

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(lambda j: _run_job(*j), jobs))
        total_out = sum(1 for r in results if r)
    dur = time.time() - start
    print(f"Concluído: {len(paths)} imagens de entrada, {total_out} geradas em {dur:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Gerador de variações de imagens para treino de IA")
    parser.add_argument("--input", default=str(Path(__file__).parent / "input"), help="Pasta de entrada")
    parser.add_argument("--output", default=str(Path(__file__).parent / "output"), help="Pasta de saída")
    parser.add_argument("--type", choices=["bite", "idle"], help="Classe destino no output (organiza em subpastas bite/idle)")
    parser.add_argument("--per-image", type=int, default=200, help="Quantidade de variações por imagem")
    parser.add_argument("--seed", type=int, default=42, help="Semente para reproducibilidade")
    parser.add_argument("--format", choices=["png", "jpg", "jpeg"], default="png", help="Formato de saída")
    parser.add_argument("--default", action="store_true", help="Espelha a estrutura de pastas de entrada no output")
    parser.add_argument("--max-content", type=int, help="Quantidade máxima de imagens por pasta no output")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Quantidade de workers simultâneos")
    args = parser.parse_args()

    process_images(args.input, args.output, args.per_image, args.seed, args.format, args.type, args.default, args.max_content, args.workers)


if __name__ == "__main__":
    main()