# Swin Transformer - Hierarchical Vision Transformer using Shifted Windows

# Introduction

# Method

## Overall Architecture
[그림3]

ViT와 같이 이미지를 patch로 분할하여 swin transformer block에 입력한다. 이때, 각 patch는 token으로 취급

* Stage 0: 4 by 4 patch를 통해 48개 channel을 가진 patch 생성
* Stage 1: 이를 linear embedding을 거쳐 C차원으로 변경, 이 token들이 block 통과
* Stage 2:
  * hierarchical representation을 만들기 위해 patch merging 단계 거침
  * 인접한 (2 x 2) = 4개의 patch들 끼리 결합하여 하나의 큰 patch 만듦 4C
  * 이를 linear layer에 통과시켜 2C로 조정(down sampling)
* Stage 3,4: patch size는 점점 커지고 수도 많아지며 각 token 차원은 두 배씩 늘어남
  > 더 작은 resolution의 representation을 만듦


## Shifted Window based Self-Attention

#### Self-attention in non-overlapped windows
겹치지않는 윈도우의 self-attentiondml
