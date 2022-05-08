# Going deeper with convolutions

## Abstract
* 코드명 Inception이라는 심층 컨볼루션 신경망 아키텍처를 제안
* 정교한 설계로 depth와 width를 늘려도 연산량이 증가하지 않고 유지됨
* Hebbian principle과 multi-scale processing을 기반으로 함

## introduction
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

* ㅎ


## Architectural Details
Inception의 핵심 아이디어는 convolutional vision network에서 최적의 local sparse 구조를 어떻게 하면 현재 사용 가능한 dense component로 구성할지에서 기반

* 본 논문에서는 패치 정렬 문제를 피하기 위해 filter size를 1x1,3x3,5x5로 제한했다.
  > 이는 필수가 아닌 편의성을 위한 결정이다. 이 계층들은 출력 filter bank가 하나의 출력 벡터로 연결되어 다음 단계에 대한 입력을 형성한다.
* convolutional 네트워크 기술에서 pooling 연산은 핵심 기능이기때문에 alternative 병렬 pooling 경로를 추가한다.

<p align="center"><img width="697" alt="스크린샷 2022-05-09 오전 2 55 47" src="https://user-images.githubusercontent.com/56713634/167309137-5bee1db3-c538-4d3a-a627-cf73e5ccec35.png"></p>


* 
