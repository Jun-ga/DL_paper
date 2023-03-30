# Efficient Convolutional Neural Networks for Mobile Vision Applications
# Introdution
더 높은 정확도를 달성하기 위해 더 깊고 더 복잡해지는게 당시 트렌드하지만 이러한 발전이 효율적인 네트워크를 만드는 것은 아님

Mobile 기기와 임베디드 비전 어플리케이션을 위한 효율적인(efficient) 모델인 __MobileNets__ 을 제안
* 모델의 크기와 성능을 적절히 선택할 수 있는 2개의 hyper-parameter를 갖는 효율적인 모델

<p align="center"><img width="679" alt="스크린샷 2023-03-31 오전 12 33 37" src="https://user-images.githubusercontent.com/56713634/228888476-28d5ae93-63d8-4b6f-bcde-63ef55453a01.png"></p>

# MobileNet Architecture
## Depthwise Separable Convolution
factorized convolutions의 형태로 __depthwise convolution__ , __1×1 convolution__(pointwise convolutionn)으로 구성됨

depthwise convolution은 입력 채널당 1개의 filter, pointwise convolution은 depthwise convolution의 결과를 1x1 convolution을 통해 합침
> standard convolution은 이 두과정을 하나로 합친 것

depthwise convolution은 layer를 2개로 나눔

* a separate layer for filtering | a separate layer for combining
  > 이 과정을 통해 모델의 크기를 크게 줄여준다.


<p align="center"><img width="591" alt="스크린샷 2023-03-31 오전 12 35 14" src="https://user-images.githubusercontent.com/56713634/228888926-1d30ef84-de7e-4596-b2f2-16b557bd0de3.png"></p>

### standard convolution input/output size

* input size: __D_F__ x __D_F__ x __M__
* output size: __D_K__ x __D_K__ x __N__
* kernel size : __D_K__ x __D_K__ x __M__ x __N__
* cost : __(D_K)^2__ x __M__ x __N__ x __(D_F)^2__

> D_F : input feature map size | D_K : Kernel size


### depthwise convolution input/output size : Filtering stage

* input size: __D_F__ x __D_F__ x __M__ 
* output size: __D_K__ x __D_K__ x __M__
* kernel size : __D_K__ x __D_K__ x 1 
* cost : __(D_K)^2__ x __M__ x __(D_F)^2__

### pointwise convolution input/output size : Combination stage

* input size: __D_K__ x __D_K__ x __M__
* output size: __D_K__ x __D_K__ x __N__
* kernel size : 1 x 1 x __M__ x __N__
* cost : __N__ x __M__ x __(D_F)^2__


### depthwise sperable convolution cost
__M__ x __(D_K)^2__ x __(D_F)^2__ + __N__ x __M__ x __(D_F)^2__


### standard VS depthwise sperable

<p align="center"><img width="797" alt="스크린샷 2023-03-30 오후 11 34 06" src="https://user-images.githubusercontent.com/56713634/228889156-07c61479-645f-419f-a1a5-242df10f0914.png"></p>

## overview

<p align="center"><img width="460" alt="스크린샷 2023-03-30 오후 10 57 10" src="https://user-images.githubusercontent.com/56713634/228892141-445b4696-fd67-4ce5-993b-5abc6975a9b7.png"></p>

### Network Structure and Training

<p align="center"><img width="568" alt="스크린샷 2023-03-31 오전 12 36 52" src="https://user-images.githubusercontent.com/56713634/228889414-edb8c41f-10e5-4a85-89fc-bc84eef8ddef.png"></p>

* 첫번째 layer를 제외하고는 depthwise sperable convolution을 사용
* Downsampling은 stride를 통해 진행
* average pooling은 spatial resolution을 1로 줄이고, 이를 fully connected layer와 연결


### standard convolution와 depthwise sperable convolution의 layer 비교
<p align="center"><img width="567" alt="스크린샷 2023-03-31 오전 12 37 29" src="https://user-images.githubusercontent.com/56713634/228889581-70159c12-384e-4a32-a286-67ab2b0f1b66.png"></p>

## Width Multiplier: Thinner Models
MobileNet은 이미 충분히 작고 빠르지만 더 작고 빠르게 만들어야하는 경우가 발생 -> __Width multiplier__

* 각 layer에서 네트워크를 균일하고 작게 만드는 역할
* hyper-parameter(α)가 각 layer마다 얼마나 얇게 만드는지 결정
* in/output channel은 M,N에서 αM,αN
* α = 1일 때 baseline MobileNet 
* α < 1 일 때 reduced MobileNets
  > 1, 0.75, 0.5, 0.25와 같이 특정한 수로 세팅

## Resolution Multiplier: Reduced Representation
network의 computational cost를 줄이기 위한 방법 -> __Resolution Multiplier__

<p align="center"><img width="553" alt="스크린샷 2023-03-31 오전 12 38 14" src="https://user-images.githubusercontent.com/56713634/228889788-83fe9c10-a181-4d04-b716-1b56e5fb75eb.png"></p>

* input 이미지와 각 layer의 내부 representation를 Resolution Multiplier로 곱해 줄여줌
* ρ = 1 일 때 baseline MobileNet
* ρ < 1 일 때 reduced computation MobileNets
* 이미지 해상도를 224, 192, 160, 128 정도로 만들게 한다.

# Experiments
## Model Choices & Shrinking Hyperparameters
<p align="center"><img width="372" alt="스크린샷 2023-03-31 오전 12 53 26" src="https://user-images.githubusercontent.com/56713634/228893995-7919e563-9403-4d33-baaf-cb58f87d7e83.png"></p>

## Object Detection
<p align="center"><img width="359" alt="스크린샷 2023-03-31 오전 12 58 07" src="https://user-images.githubusercontent.com/56713634/228895212-980f3445-dcc9-4485-b0a6-8b11385675f4.png"></p>


# Conclusion
* depthwise separable convolutions을 기반으로 한 MobilNets을 제안
* 연산량에 비해 높은 성능(성능이 떨어지지않음), 사용 환경에 따라 적절한 크기의 모델을 선택할 수 있는 옵션을 제공했다.

