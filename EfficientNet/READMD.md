EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks

# Introduction
* 더 높은 정확도를 달성하기 위해서 convnet을 확장하는 방법이 널리 사용됨
  > ResNet은 18, 200으로도 확장이 가능함
 * 하지만 이는 정확한 이해가 없이 진행되고 있음 (본 저자 의견)
  > 너비, 깊이 해상도를 각각 수동으로 scalig하여 진행
 * 본 연구에선 scaling을 보다 이해하여 compound scaling을 제안한다
  > 이를 MobileNet, ResNet을 통해 검증함

그림 1

# Compound Model Scaling
## Problem Formulation
