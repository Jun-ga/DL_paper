# Swin Transformer - Hierarchical Vision Transformer using Shifted Windows

# Introduction

# Method

## Overall Architecture
<p align="center"><img width="1093" alt="스크린샷 2023-03-07 오전 11 16 16" src="https://user-images.githubusercontent.com/56713634/223302188-607ac60d-06e8-418d-9196-4c12b501c3cd.png">
</p>

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
<p align="center"><img width="432" alt="스크린샷 2023-03-06 오후 5 06 28" src="https://user-images.githubusercontent.com/56713634/223302300-fb19ca8e-a907-45b1-a608-966685479c03.png">
</p>
__MSA__ 는 제곱에 비례해 계산량이 증가하지만 __W-MSA__ 는 선형적임

### Shifted window partitioning in successive blocks 
Window가 고정되어 있기때문에 self-attention시 고정된 부분만 수행한다는 문제점을 해결하기 위한 방법
 > global 한 특징을 잡아내는 것이 어려워짐

non-overlapped window의 효율적인 계산을 유지하면서 cross-window connections을 도입하기 위해 __Transformer block에서 두개의 분할 구성을 번갈아 사용하는 shifted window partitioning 접근 방식을 제안__ </br>

<p align="center"><img width="527" alt="스크린샷 2023-03-07 오전 11 18 12" src="https://user-images.githubusercontent.com/56713634/223302401-94f66100-800f-4ef8-84e6-acd8b50c159a.png"></p>

* 첫번째 모듈은 왼족 상단 픽셀에 시작하는 일반 window partitioning 전략을 사용한다
* 8x8 feature map size는 4x4인 2x2 window로 균등하게 분할
* 다음 모듈은 규칙적으로 분할된 window에서 pixel만큼 shift하여 이전 layer의 구성에서 shift된 window 구성을 채택한다.



#### consecutive Swin Transformer blocks are computed as

<p align="center"><img width="422" alt="스크린샷 2023-03-07 오전 11 22 17" src="https://user-images.githubusercontent.com/56713634/223302996-a5147ed4-2934-417d-91af-fcccb264819c.png"></p>

* MLP는 layer 2개와 GELU를 적용
* 각 모듈에서 self attention 전에 LayerNorm 적용
* MSA은 window 내에서 계산

### 문제점
* shifting을 적용함에 있어 효율적으로 window를 배치해야함
* W-MSA의 window 수와 SW-MSA의 window 수가 달라짐
  > 2x2인 window 개수가 3x3으로 늘고 크기가 MxM보다 작은 window들이 생김

### Efficient batch computation for shifted configuration

<p align="center"><img width="536" alt="스크린샷 2023-03-07 오전 11 22 59" src="https://user-images.githubusercontent.com/56713634/223303089-d8784ff7-8a65-4753-ade0-3501a372c80f.png"></p>

* window를 shift 시키는 것을 cyclic shift
* window size의 1/2 만큼 우측 하단으로 shift 하고 A, B, C 구역을 padding
  > padding 시키는 부분은 반대편인 좌 상단에서 온 것이므로 A, B, C를 포함해서 self-attention을 진행하는 것은 의미가 없음
* A, B, C에 mask를 씌운 뒤 self-attention을 수행
* reverse cyclic shift를 진행해 원래 값으로 되돌림

__padding을 사용해서 이 방식을 대신할 수 있지만, computational cost가 증가될 수 있어 이 방법을 사용__

## Relative position bias
Swin 방식은 ViT 방식 처럼 위치 정보를 위한 Positional encoding을 처음에 적용하지 않음

__self-attention 과정에서 relative position bias를 추가함으로써 그 역할을 대체__

<p align="center"><img width="503" alt="스크린샷 2023-03-07 오전 11 23 50" src="https://user-images.githubusercontent.com/56713634/223303219-7d47fcbb-eb5a-4a35-9135-23f9cda9528f.png"></p>

* attention score 구하는 식 뒤에 B(bias)를 더해줌
* M개의 patch가 하나의 window를 구성하므로 각 축에 따라 상대적 위치는 [-M + 1, M - 1] 범위 내에 있음
  > 1번째 패치를 기준으로 M번째 패치의 거리는 M-1 이고 M번째 패치를 기준으로 1번째 패치는 -M+1
  
  > 절대 거리는 같지만 방향이 다르므로 상대 거리 사용
* 작은 크기의 B 행렬을 <img width="200" alt="스크린샷 2023-03-07 오전 11 24 08" src="https://user-images.githubusercontent.com/56713634/223303270-31d6b631-b2e0-4d1f-b0be-93091dfc8f9b.png">에 속하는 B로 파라미터화 가능
* bias 항이 없거나(Table 4) absolute position embedding을 사용했을 때보다 상당한 모델 성능의 향상을 가져옴

## Architecture Variant
<p align="center"><img width="410" alt="스크린샷 2023-03-07 오전 11 24 59" src="https://user-images.githubusercontent.com/56713634/223303386-4bd0ee8e-6ff3-4702-8313-d6a2a12d395b.png"></p>

# Experiments
