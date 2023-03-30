# Efficient Convolutional Neural Networks for Mobile Vision Applications
# introdution
* alexnet으로 부터 deeper net

# MobileNet Architecture
## Depthwise Separable Convolution
factorized convolutions의 형태로 __depthwise convolution__ , __1×1 convolution__(pointwise convolutionn)으로 구성됨

depthwise convolution은 입력 채널당 1개의 filter, pointwise convolution은 depthwise convolution의 결과를 1x1 convolution을 통해 합침
> standard convolution은 이 두과정을 하나로 합친 것
depthwise convolution은 layer를 2개로 나눔
* a separate layer for filtering | a separate layer for combining
  > 이 과정을 통해 모델의 크기를 크게 줄여준다.


[사진2 첨부]

### standard convolution input/output size

* input size: __D_F__ x __D_F__ x __M__
* output size: __D_G__ x __D_G__ x __N__
* kernel size : __D_K__ x __D_K__ x __M__ x __N__
* cost : __(D_K)^2__ x __M__ x __N__ x __(D_G)^2__

> D_F : input feature map size | D_K : Kernel size


### depthwise convolution input/output size : Filtering stage

* input size: __D_F__ x __D_F__ x __M__ 
* output size: __D_G__ x __D_G__ x __M__
* kernel size : __D_K__ x __D_K__ x 1 
* cost : __(D_K)^2__ x __M__ x __(D_F)^2__

### pointwise convolution input/output size : Combination stage

* input size: __D_G__ x __D_G__ x __M__
* output size: __D_G__ x __D_G__ x __N__
* kernel size : __D_G__ x __D_G__ x __M__ x __N__
* cost : __(D_K)^2__ x __M__ x __(D_F)^2__


### depthwise sperable convolution cost
__M__ x __(D_K)^2__


### Network Structure and Training

[테이블 1]

* 첫번째 layer를 제외하고는 depthwise sperable convolution을 사용
* Downsampling은 stride를 통해 진행
* average pooling은 spatial resolution을 1로 줄이고, 이를 fully connected layer와 연결


### standard convolution와 depthwise sperable convolution의 layer 비교
[그림 3]

## Width Multiplier: Thinner Models

## Resolution Multiplier: Reduced Representation


> 출처 : CodeEmporium 유튜브, Depthwise Separable Convolution - A FASTER CONVOLUTION!
