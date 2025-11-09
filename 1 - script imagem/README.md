# Script de Augmentação de Imagens

Este script gera milhares de variações de coloração, rotação, corte, ruído e outros efeitos a partir de imagens colocadas em `input/`. As variações são salvas em `output/` para uso em treinamento de IA de visão computacional.

## Estrutura
- `1 - script imagem/input/` — coloque aqui as imagens originais (png, jpg, jpeg, bmp, webp)
- `1 - script imagem/output/` — o script salva aqui as variações, organizadas por subpastas por imagem
- `augment.py` — script principal
- `requirements.txt` — dependências mínimas

## Instalação
1. Python 3.9+ instalado
2. Instale dependências:
   - `pip install -r "1 - script imagem/requirements.txt"`

## Uso
Exemplos de execução no Windows (PowerShell):
- `python "1 - script imagem/augment.py" --per-image 500`
  - Lê de `1 - script imagem/input/` e escreve em `1 - script imagem/output/`
  
- Organizar diretamente por classe para o Módulo 2 (ImageFolder):
  - `python "1 - script imagem/augment.py" --type bite --per-image 200`
  - `python "1 - script imagem/augment.py" --type idle --per-image 200`
  - Saída ficará em `output/bite/` e `output/idle/` respectivamente.
- Parâmetros:
  - `--input` caminho da pasta de entrada (padrão: `input` ao lado do script)
  - `--output` caminho da pasta de saída (padrão: `output` ao lado do script)
  - `--type` `bite` | `idle` (organiza saída diretamente em subpastas de classe)
  - `--per-image` quantidade de variações por imagem (padrão: 200)
  - `--seed` semente de aleatoriedade para reprodutibilidade (padrão: 42)
  - `--format` `png` | `jpg` | `jpeg` (padrão: `png`)

## Observações
- As variações combinam operações aleatórias (rotação, flips, shear, cortes/redimensionamentos, jitter de cor, ajuste de HSV, ruído gaussiano, blur/unsharp, gamma, artefato JPEG e conversão ocasional para grayscale).
- Saída é organizada em subpastas por nome base da imagem original.
- Ajuste `--per-image` conforme necessidade para gerar milhares de amostras.