import argparse
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from tqdm import tqdm


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(img_size: int):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, val_tf


def create_model(name: str, num_classes: int, pretrained: bool = True, freeze_backbone: bool = True):
    name = name.lower()
    if name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = m.classifier[3].in_features
        m.classifier[3] = nn.Linear(in_features, num_classes)
    elif name == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None)
        in_features = m.classifier[3].in_features
        m.classifier[3] = nn.Linear(in_features, num_classes)
    elif name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = m.fc.in_features
        m.fc = nn.Linear(in_features, num_classes)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Modelo não suportado: {name}")

    if freeze_backbone:
        # Congela tudo, depois descongela apenas o head
        for p in m.parameters():
            p.requires_grad = False
        # Maneira segura: marcar grad apenas nas camadas finais
        head_modules = []
        if hasattr(m, 'classifier'):
            head_modules.append(m.classifier)
        if hasattr(m, 'fc'):
            head_modules.append(m.fc)
        for mod in head_modules:
            for p in mod.parameters():
                p.requires_grad = True

    return m


def split_or_load(data_dir: Path, val_split: float, train_tf, val_tf, seed: int):
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    if train_dir.exists() and any(train_dir.iterdir()) and val_dir.exists() and any(val_dir.iterdir()):
        train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
        val_ds = datasets.ImageFolder(val_dir, transform=val_tf)
        return train_ds, val_ds
    else:
        # Modo único diretório seguindo ImageFolder: cada subpasta é uma classe
        base_ds = datasets.ImageFolder(data_dir, transform=train_tf)
        n = len(base_ds)
        val_n = max(1, int(round(n * val_split)))
        train_n = n - val_n
        # Usar mesma distribuição de classes aproximada no split
        generator = torch.Generator().manual_seed(seed)
        train_ds, val_ds = random_split(base_ds, [train_n, val_n], generator=generator)
        # Corrige transform do val_ds
        val_ds.dataset.transform = val_tf
        return train_ds, val_ds


def evaluate(model, loader, device, amp=False):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            if amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)
            total_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += images.size(0)
    return total_loss / max(1, total), correct / max(1, total)


def train_one_epoch(model, loader, optimizer, device, scaler=None, amp=False, progress=True):
    model.train()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0
    iterator = tqdm(loader, desc="Treino", unit="batch", leave=False) if progress else loader
    for images, targets in iterator:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and amp:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(targets).sum().item()
        total += images.size(0)
        if progress and total > 0:
            iterator.set_postfix({"loss": f"{(running_loss/total):.4f}", "acc": f"{(correct/total):.4f}"}, refresh=False)
    return running_loss / max(1, total), correct / max(1, total)


def main():
    parser = argparse.ArgumentParser(description="Treino de modelo de visão (classificação) para detectar evento da bóia")
    default_data = Path(__file__).parent.parent / "1 - script imagem" / "output"
    parser.add_argument('--data-dir', type=str, default=str(default_data), help='Pasta dos dados (ImageFolder).')
    parser.add_argument('--model', type=str, default='mobilenet_v3_small',
                        choices=['mobilenet_v3_small', 'mobilenet_v3_large', 'resnet18', 'efficientnet_b0'],
                        help='Arquitetura base para fine-tuning.')
    parser.add_argument('--img-size', type=int, default=256, help='Tamanho de entrada das imagens.')
    parser.add_argument('--epochs', type=int, default=15, help='Número de épocas.')
    parser.add_argument('--batch-size', type=int, default=64, help='Tamanho de batch.')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate.')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay (L2).')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['sgd', 'adamw'], help='Otimizador.')
    parser.add_argument('--val-split', type=float, default=0.1, help='Proporção de validação caso não haja train/val.')
    parser.add_argument('--num-workers', type=int, default=4, help='Workers do DataLoader.')
    parser.add_argument('--seed', type=int, default=42, help='Semente para reproducibilidade.')
    parser.add_argument('--freeze-backbone', action='store_true', help='Congela o backbone e treina só o head.')
    parser.add_argument('--no-pretrained', action='store_true', help='Não usar pesos pré-treinados.')
    parser.add_argument('--amp', action='store_true', help='Usa mixed precision (AMP).')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Dispositivo de treino.')
    parser.add_argument('--device-id', type=int, default=0, help='Índice da GPU (quando cuda).')
    parser.add_argument('--no-progress', action='store_true', help='Desativa barra de progresso por batch.')
    parser.add_argument('--save-dir', type=str, default=str(Path(__file__).parent / 'runs'), help='Diretório para salvar modelos.')
    parser.add_argument('--name', type=str, default='bobber_event_cls', help='Nome do experimento (subpasta em runs/).')

    args = parser.parse_args()

    # Seletor de device
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.device_id}')
        torch.cuda.set_device(args.device_id)
        print(f"Usando GPU: {torch.cuda.get_device_name(args.device_id)} (id={args.device_id})")
    else:
        device = torch.device('cpu')
        print("CUDA indisponível; usando CPU.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir não encontrado: {data_dir}")

    train_tf, val_tf = build_transforms(args.img_size)
    train_ds, val_ds = split_or_load(data_dir, args.val_split, train_tf, val_tf, args.seed)

    # Determina número de classes a partir do dataset
    # Para random_split, classes ficam no dataset interno
    classes = getattr(train_ds, 'dataset', train_ds).classes
    num_classes = len(classes)
    print(f"Encontradas {num_classes} classes: {classes}")

    model = create_model(
        args.model,
        num_classes=num_classes,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
    )
    model.to(device)

    # Otimizador
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    # DataLoaders
    pin_mem = device.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_mem)

    # AMP scaler
    use_amp = args.amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Diretório de saída
    run_dir = Path(args.save_dir) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, scaler=scaler, amp=use_amp, progress=not args.no_progress
        )
        val_loss, val_acc = evaluate(model, val_loader, device, amp=use_amp)
        dt = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} acc={train_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f} | {dt:.1f}s")

        # Salva melhor
        if val_acc >= best_acc:
            best_acc = val_acc
            ckpt_path = run_dir / 'best.pt'
            torch.save({'model': model.state_dict(), 'classes': classes, 'args': vars(args)}, ckpt_path)
            print(f"Salvo melhor modelo em: {ckpt_path} (acc={best_acc:.4f})")

    total_dt = time.time() - start
    print(f"Treino concluído em {total_dt/60:.1f} min. Melhor val_acc={best_acc:.4f}")


if __name__ == '__main__':
    main()