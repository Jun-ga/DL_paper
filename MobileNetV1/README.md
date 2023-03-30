# Efficient Convolutional Neural Networks for Mobile Vision Applications
# introdution
* alexnet으로 부터 deeper net

# MobileNet Architecture
## Depthwise Separable Convolution
factorized convolutions의 형태로 __depthwise convolution__ , __1×1 convolution__(pointwise convolutionn)으로 구성됨

depthwise convolution은 입력 채널당 1개의 filter, pointwise convolution은 depthwise convolution의 결과를 1x1 convolution을 통해 합침

depthwise convolution은 layer를 2개로 나눔
* a separate layer for filtering | a separate layer for combining
  > 이 과정을 통해 모델의 크기를 크게 줄여준다.


[사진2 첨부]


  
