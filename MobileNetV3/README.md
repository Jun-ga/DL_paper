# Searching for MobileNetV3

# Introduction
* 컴퓨터 비전을 지원하는 차세대 고정확도 호율적인 신경망 작업 모델을 제공하기 위헤

# Efficient Mobile Building Blocks
* Mobilenet은 보다 효율적인 building blocks을 구축함
* MobileNetV1에서 convlayer를 효율적으로 대체하기 위해 Depthwise Separable Convolution을 제안해서 효율적인 구조 제안
  * Feature 생성 매커니즘에서 공간 필터링을 분리하여 기존 Convolution을 효과적으로 분해함
    > Light weight Depthwise Convolutions
    
    > 1 x 1 Pointwise Convolutions

[그림3]

* MobileNetV2에서는 linear bottleneck과 Inverted Residual 구조를 도입하여 low feature을 활용하여 보다 효율적인 레이어 구조를 제안
*  
