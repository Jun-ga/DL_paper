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

### Self-attention in non-overlapped windows
겹치지않는 윈도우의 self-attention은 아래의 식과 같음
[식 1,2]
__MSA__ 는 제곱에 비례해 계산량이 증가하지만 __W-MSA__ 는 선형적임

### Shifted window partitioning in successive blocks 
Window가 고정되어 있기때문에 self-attention시 고정된 부분만 수행한다는 문제점을 해결하기 위한 방법
 > global 한 특징을 잡아내는 것이 어려워짐

non-overlapped window의 효율적인 계산을 유지하면서 cross-window connections을 도입하기 위해 __Transformer block에서 두개의 분할 구성을 번갈아 사용하는 shifted window partitioning 접근 방식을 제안__

[그림 2]

* 첫번째 모듈은 왼족 상단 픽셀에 시작하는 일반 window partitioning 전략을 사용한다
* 8x8 feature map size는 4x4인 2x2 window로 균등하게 분할
* 다음 모듈은 규칙적으로 분할된 window에서 pixel만큼 shift하여 이전 layer의 구성에서 shift된 window 구성을 채택한다.

#### consecutive Swin Transformer blocks are computed as
[식 3]
* MLP는 layer 2개와 GELU를 적용

### Efficient batch computation for shifted configuration

[그림 4]
