# Going deeper with convolutions

## Abstract
* 코드명 Inception이라는 심층 컨볼루션 신경망 아키텍처를 제안
* 정교한 설계로 depth와 width를 늘려도 연산량이 증가하지 않고 유지됨
* Hebbian principle과 multi-scale processing을 기반으로 함

## Introduction
* deep learning, convolutional networks의 발전으로 객체 분류 및 탐지 기능이 크게 향상됨
* 이러한 발전은 하드웨어의 발전뿐만 아니라 네트워크 구조에 대한 새로운 아이디어와 알고리즘, 개선된 신경망 구조의 결과
  > GoogLeNet은 AlexNet에 비해 12배 작은 파라티머를 가지면서 더 정확한 성능을 가짐
* 모바일 및 임베디드 컴퓨팅의 지속적인 관심으로 알고리즘의 효율성, 특히 전력 메모리 사용이 중요해지고 있다.
  > 이에, GoogLeNet은 inference시에 multiply-adds(합성곱)을 1.5 billion이하로 지정하여 설계함 -> 대규모 데이터셋에서 합리적임
* Inception 모듈은 영화 인셉션의 대사인 "We need to go deeper"에서 유래했다.
  > 이때, deep는 두 가지 의미를 가짐
  > 1. “Inception module”의 형태의 새로운 구조를 도입
  > 2. 네트워크의 depth가 증가했다는 의미
 
 ## Related Work
 <p align="center"><img width="639" alt="스크린샷 2022-05-08 오후 8 01 44" src="https://user-images.githubusercontent.com/56713634/167293192-9bfe7c5a-1ca7-4e1b-9b8c-417838348bb7.png"></p>

* LeNet-5부터 CNN은 일반적으로 표준 구조를 가짐
  > convolution layer들을 쌓고 그 뒤에 FC layer를 쌓는 구조 <br>
* 2014년 기준 트렌드는 layer 수를 늘리고, overfitting을 피하기 위해 dropout을 사용
* maxpooling layer가 손실을 초래하지만 CNN은 localization(현지화), object detection(객체 탐지), human pose estimation(인간 자세 추정)
* 신경망의 표현력을 높이기 위해 제안된 접근법인 Network in Network 논문에 많은 영향을 받음
  > 1x1 conv layer가 추가되며, relu가 뛰따름 이때, 1x1 conv layer는 병목형산을 제거하기 위한 차원 축소와 네트워크 크기를 제한하는 용도로 사용된다.

## Motivation and High Level Considerations
심층 신경맘의 성능을 향상시키는 가장 간단한 방법  = 신경망의 크기를 늘리는 것(depth와 width의 증가)

* 신경망의 크기를 늘리는 것은 두가지 결점이 존재
  1. 학습할 파라미터의 수가 증가하면서 overfitting을 초래
  2. computational 자원이 더 많이 필요하게 됨

* 이 두가지 문제를 해결하는 근본적인 방법은 __dense한 Fully Connected 구조에서 Soarseky Connected 구조로 바꾸는 것__
 <p align="center"><img width="670" alt="스크린샷 2022-05-09 오전 2 15 00" src="https://user-images.githubusercontent.com/56713634/167307584-98b73957-1e11-4b61-9b99-927d8bf1f35f.png"></p>

더 작성해야하는 부분임


## Architectural Details
Inception의 핵심 아이디어는 convolutional vision network에서 최적의 local sparse 구조를 어떻게 하면 현재 사용 가능한 dense component로 구성할지에서 기반

* 본 논문에서는 패치 정렬 문제를 피하기 위해 filter size를 1x1,3x3,5x5로 제한했다.
  > 이는 필수가 아닌 편의성을 위한 결정이다. 이 계층들은 출력 filter bank가 하나의 출력 벡터로 연결되어 다음 단계에 대한 입력을 형성한다.
* convolutional 네트워크 기술에서 pooling 연산은 핵심 기능이기때문에 alternative 병렬 pooling 경로를 추가한다.

<p align="center"><img width="697" alt="스크린샷 2022-05-09 오전 2 55 47" src="https://user-images.githubusercontent.com/56713634/167309137-5bee1db3-c538-4d3a-a627-cf73e5ccec35.png"></p>


* Inception module이 쌓이면서 출력값의 특징이 달라지게 되는데 이는 더 높은 수준의 추상화 특징이 더 높은 계층에 의해 포착될 수록 그들의 공간적인 집중도도 줄어든다.
  > 3x3, 5x5 convolutional filter의 수도 증가할 것으로 추정 됨 
* __(a) module의 가장 큰 문제는 연산량 증가__
* pooling layer의 출력을 다른 convolution layer의 출력과 함께 concatenate할때 필터수가 module 입력에 비해 증가하게되고, 더욱 심한 연산량 증가를 야기한다. 

* 이 문제를 해결하기 위해 1x1 convolutional filter를 이용하여 차원을 축소했다
  > 3x3, 5x5 convoluton 앞에 1x1을 두어 차원을 줄여 연산량을 낮춤

* 또한, Inception module은 높은 layer에서만 사용하고 낮은 layer에서는 기본적인 CNN 모델을 사용했다. _효율적인 메모리 사용을 위함_

### 이러한 특징을 가진 Inception module을 사용하면 아래 두가지 효과를 가질 수 있다.
1. 연산량에 구애받지않고 각 단계의 유낫수를 증가시킬 수 있다.
    > 이는 차원 축소의 사용을 통해 가능
2. visual 정보는 다양한 scale에 대해 처리한 다음 그 다음 단계가동시에 서로 다른 scale에서 특징을 추출할 수 있도록 한다.


## GoogLeNet
_LeNet5에 대한 경의의 표현으로 GoogLeNe이라는 팀명을 정했다._
* Inception moodule을 더 깊고 넓게 구성했지만 성능이 조금 떨어졌으며 이에, ensemble을 함께 적용했다.

<p align="center"><img width="693" alt="스크린샷 2022-05-09 오전 3 37 56" src="https://user-images.githubusercontent.com/56713634/167310606-6aa93f0ab-c3ed-4864-87d7-77ddaad88172.png"></p>

* input size 224 x 224
* "#3✕3 reduce"와 "#5✕5 reduce"는 3✕3와 5✕5 convolution 이전에 사용된 축소 계층의 1✕1 필터의 개수
* pool proj는 max-pooling 이후의 1x1 conv layer의 필터의 수
* 든 reduction과 projection 또한 ReLU가 적용

__network가 깊어지면서 gradient를 효율적으로 연전파시키는 것이 중요해짐__
* 이에, auxiliary classifier를 network의 Inceptaion 4a와 Inception 4d 뒤에 추가
* 학습중에는 auxiliary classifier의 출력의 0.3을 곱해서 전체 loss에 더해지고, 테스트 시에는 아예 auxiliary classifier를 사용하지 않음

__Auxiliary classifier의 구성요소__

* Inception 4a와 Inception 4d 뒤에 5x5, stride 3인 a$verage pooling
* 필터의 수가 128인 1x1 conv layer 및 ReLU
* 유닛의 개수가 1024개인 fc layer 및 ReLU
* p=0.7의 dropout
* 유닛의 개수가 1000개인 fc layer

<p align="center"><img width="698" alt="스크린샷 2022-05-09 오전 3 48 30" src="https://user-images.githubusercontent.com/56713634/167310994-d7764066-1b5d-48ef-8c82-348c7a0b5ef9.png"></p>

## Training Methodology

* Optimizer: Asynchronous SGD with 0.9 momentum
* Learning rate : 8 epochs 마다 4%씩 감소
* 본래 사이즈의 8%에서 100%까지 고르게 분포된 사이즈를 가지며 가로 세로의 비율을 3:4, 4:3사이로 유지하여 샘플링함
  > GoogLeNet은 샘플링 방법을 매우 자주 바꿈, 위의 방법이 가장 효과적으로 보인 것

## ILSVRC 2014 Classification Challenge Setup and Results

<p align="center"><img width="694" alt="스크린샷 2022-05-09 오전 3 57 57" src="https://user-images.githubusercontent.com/56713634/167311393-7824c578-7716-47e8-a782-fbc50840710b.png"></p>

## ILSVRC 2014 Detection Challenge Setup and Results

<p align="center"><img width="700" alt="스크린샷 2022-05-09 오전 3 58 39" src="https://user-images.githubusercontent.com/56713634/167311448-822bfa66-51e1-4b22-af6b-fa06a9ecafd3.png"></p>


## Conclusions

Inception의 결과는 추측한 최적의 sparse 구조를 준비된 밀집된 빌딩 블록을 통해 근사하는 것은 컴퓨터 비전을 위한 신경망의 성능을 개선하는 현실적인 방법임에 대한 강력한 증거가 되는 것으로 보인다. 대 장점은 얕으면서 넓은 네트워크에 있어 계산 소요가 얼마 증가되지 않았음에도 성능이 개선이 되었다는 점이다.
