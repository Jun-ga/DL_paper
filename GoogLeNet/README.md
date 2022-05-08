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

