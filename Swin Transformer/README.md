# Swin Transformer - Hierarchical Vision Transformer using Shifted Windows

# Introduction

<p align="center"><img width="537" alt="스크린샷 2023-03-07 오전 11 57 05" src="https://user-images.githubusercontent.com/56713634/223308700-4f387ac1-dbcc-43a2-b671-2f7405f7df30.png">
</p>

* 본 논문에서는 Transformer가 NLP와 CNN이 vision에서 하는 것처럼 컴퓨터 비전을 위한 backbone 역할을 할 수 있도록 적용 가능성을 확장하고자 함
* token은 고정 scale이기에 vision task에 적합하지않고 NLP task에 비해 해상도가 많이 높다 라는 문제
* 이를 극복하기 위해 hierarchical feature maps을 구성하고 Swin transformer를 제안한다.
* 기존 ViT와 다르게 hierarchical 구조로 작은 patch에서 시작하여 layer가 깊어짐에 따라 주변 patch들과 병합하는 방식으로 진행
  > FPN, U-Net과 같은 dense predicition이 가능
  
<p align="center"><img width="527" alt="스크린샷 2023-03-07 오전 11 18 12" src="https://user-images.githubusercontent.com/56713634/223302401-94f66100-800f-4ef8-84e6-acd8b50c159a.png"></p>

* self-attention을 진행한 뒤 다음 layer에서 window를 patch 1/2 크기만큼 오른쪽 밑으로 shift 시킨 뒤 self-attention을 진행
* sliding window 방식은 다른 query pixel에 대해 다른 key set을 가지게 되어 낮은 latency를 가지게 되는 반면, shifted window 방식은 모든 query patch는 같은 key set을 공유해서 memory access면에서 latency가 더 적다

# Method

## Overall Architecture
<p align="center"><img width="1093" alt="스크린샷 2023-03-07 오전 11 16 16" src="https://user-images.githubusercontent.com/56713634/223302188-607ac60d-06e8-418d-9196-4c12b501c3cd.png">
</p>

ViT와 같이 이미지를 patch로 분할하여 swin transformer block에 입력한다. 이때, 각 patch는 token으로 취급
  > ViT는 16x16 고정, patch 사이즈가 작아지면 전체 patch 사이즈의 개수가 제곱으로 늘어가고 이에 연산량 증가
* Stage 0: 4 by 4 patch를 통해 48개 channel을 가진 patch 생성
  > 4x4 patch로 상당히 작은 크기임
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

<p align="center"><img width="265" alt="스크린샷 2023-03-07 오후 12 21 04" src="https://user-images.githubusercontent.com/56713634/223312481-ef2ad23e-7adf-451d-a2ad-2828dbccca2a.png">
</p>

## Architecture Variant

<p align="center"><img width="410" alt="스크린샷 2023-03-07 오전 11 24 59" src="https://user-images.githubusercontent.com/56713634/223303386-4bd0ee8e-6ff3-4702-8313-d6a2a12d395b.png"></p>

# Experiments
다음을 사용하여 실험 진행

* ImageNet-1K image classification
* COCO object detection
* ADE20K semantic segmentation

## ImageNet-1K image classification

<p align="center"><img width="268" alt="스크린샷 2023-03-07 오후 12 20 10" src="https://user-images.githubusercontent.com/56713634/223312333-40441a27-df76-4275-9da4-1ccb5983c1c8.png">
</p>

## COCO object detection

<p align="center"><img width="277" alt="스크린샷 2023-03-07 오후 12 20 39" src="https://user-images.githubusercontent.com/56713634/223312389-1d73c95a-6c18-4b0c-a81d-ebac7391897f.png">
</p>

## ADE20K semantic segmentation

<p align="center"><img width="275" alt="스크린샷 2023-03-07 오후 12 20 44" src="https://user-images.githubusercontent.com/56713634/223312418-6d6a493e-fad8-46fd-bb19-88107999a8f7.png">
</p>

# Conclusion
* 본 논문은 hierarchical representation을 생성하고 입력 이미지 크기에 대해 선형 계산 복잡도를 가진 Swin transformer를 제안했다
* 이는 COCO 및 ADE20K에서 높은 성능을 보였고 이전의 best 방법을 훨씬 능가한다.
