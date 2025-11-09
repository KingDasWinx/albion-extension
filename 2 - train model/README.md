# Módulo 2 — Treino de Modelo

Este módulo treina um modelo de visão computacional para identificar o evento da bóia afundando (quando o peixe morde) a partir de imagens. Por padrão, usa classificação binária com MobileNetV3 e suporta outras arquiteturas leves.

## Modelo ideal para o caso
- Para identificar um estado visual claro em um frame (bóia afundada vs. normal), a abordagem mais simples e eficiente é **classificação de imagens** com um backbone leve (ex.: MobileNetV3 Small/Large, EfficientNet-B0, ResNet18 com transferência de aprendizado).
- MobileNetV3 é projetado para eficiência em tempo real em dispositivos móveis/edge, mantendo boa precisão, o que torna-o ideal para aplicações responsivas e com baixa latência.
  - Referências:
    - MobileNetV3 é balanceado para precisão/eficiência em edge e tempo real: https://arxiv.org/html/2505.03303v1
    - Deploy em dispositivos de borda com alta precisão e baixo custo usando MobileNetV3: https://www.mdpi.com/2076-3417/13/13/7804
    - Post oficial do Google: MobileNetV3 é até 2× mais rápido que MobileNetV2 em CPU móvel, mantendo a acurácia: https://research.google/blog/introducing-the-next-generation-of-on-device-vision-models-mobilenetv3-and-mobilenetedgetpu/
- Se for necessário localizar a bóia em múltiplos cenários variados e, além de classificar o estado, também delimitar a região, modelos de **detecção** pequenos (ex.: YOLOv8n/YOLOv5n/YOLOv4-tiny) são recomendados pela velocidade e bom compromisso acurácia-latência em edge/GPU.
  - Referências:
    - Revisão e benchmarking de detecção em edge, incluindo YOLOv8n como opção rápida: https://arxiv.org/html/2409.16808v1
    - Revisão das séries YOLO (v5/v8/v10) e variantes nano/pequenas para tempo real: https://arxiv.org/html/2407.02988v1

Para seu hardware (RTX 3050 / RTX 5060), MobileNetV3-small/large ficará muito rápido e deve atingir alta precisão. Caso precise de localização da bóia, YOLOv8n é uma boa alternativa leve.

## Estrutura esperada dos dados
O script espera dados no formato **ImageFolder**:
- `data_dir/CLASS_A/...` e `data_dir/CLASS_B/...`
- Se você já tem imagens geradas em `1 - script imagem/output/`, recomendo organizar em duas classes, por exemplo:
  - `1 - script imagem/output/bite/` (bóia afundando)
  - `1 - script imagem/output/idle/` (bóia normal)

Se não existir `train/` e `val/` dentro de `data_dir`, o script cria um **split automático** com `--val-split`.

## Instalação (Windows)
1. Instale Python 3.9+.
2. Crie/ative seu ambiente virtual (opcional).
3. Instale dependências do módulo:
   - `pip install -r "2 - train model/requirements.txt"`
4. Para usar GPU com CUDA, instale PyTorch com a build CUDA do site oficial (ajuste a versão conforme sua máquina):
   - https://pytorch.org/get-started/locally/
   - Exemplo (CUDA 12.1, pode variar): `pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision`

## Uso
Exemplos de execução no Windows (PowerShell):
- Treinar com MobileNetV3-small diretamente em `output/` com split automático:
  - `python "2 - train model/train.py" --data-dir "1 - script imagem/output" --model mobilenet_v3_small --epochs 20 --batch-size 64 --img-size 256 --amp`
- Especificar outra arquitetura (ex.: EfficientNet-B0) e congelar backbone:
  - `python "2 - train model/train.py" --data-dir "1 - script imagem/output" --model efficientnet_b0 --freeze-backbone --epochs 30`
- Caso você já tenha `train/` e `val/` dentro de `data_dir`, o script usa estas pastas automaticamente.

### Principais flags
- `--data-dir` caminho dos dados (default aponta para `1 - script imagem/output`)
- `--model` `mobilenet_v3_small|mobilenet_v3_large|resnet18|efficientnet_b0`
- `--epochs` número de épocas
- `--batch-size` tamanho do batch
- `--lr` learning rate; `--weight-decay` regularização L2
- `--img-size` tamanho de entrada (default 256)
- `--optimizer` `sgd|adamw`
- `--freeze-backbone` treina só o head (útil com poucos dados)
- `--amp` ativa mixed precision para acelerar em GPU
- `--val-split` fração de validação quando não há `train/val`
- `--device cuda|cpu` e `--device-id` para escolher GPU
- `--save-dir` local de checkpoints; `--name` subpasta do experimento

## Dicas de dataset
- Garanta que há **duas classes** (ex.: `bite` e `idle`). Treinar só a classe positiva não permite ao modelo aprender a discriminação.
- Amostre frames representativos do evento e do estado normal; aplique augmentações (já preparadas no módulo 1) para robustez a rotação, cor, blur, ruído, etc.
- Se o jogo é muito dinâmico, considere treinar primeiro classificação; se necessário, evoluir para detecção com YOLOv8n.

## Saída
- O melhor modelo é salvo em `2 - train model/runs/<name>/best.pt` com pesos e metadados.

## Próximos passos (opcionais)
- Exportar para ONNX/TensorRT para inferência mais rápida.
- Adicionar suavização temporal na inferência (ex.: consenso em N frames) para reduzir falsos positivos.