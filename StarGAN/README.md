# __StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation__

# Abstract
* 최근 연구는 2개의 domain에서의 imageto-image translation에서 큰 성공을 보여줬다.
* 하지만 2개 이상의 domain에서는 scalability(확장성)와 robustness(견고성)이 제한되었다.
  > 각 이미지 domain 쌍에 대해 독립적으로 만들어졌기 때문
* __StarGAN은 단 하나의 모델을 사용하여 여러가지 domain에 대해 imageto-image translation을 할 수 있다__ 

# Introdution

<p align="center"><img width="676" alt="스크린샷 2022-03-28 오전 11 14 55" src="https://user-images.githubusercontent.com/56713634/160315357-3832fc06-5ecf-4e3e-baae-54960ca7c148.png">

* CelebA와 RaFD dataset을 이용하여 얼굴의 특징과 표정을 변화시킴
* 이러한 라벨링 데이터셋을 이용한 multi-domain image translation은 기존 모델에서 비효율적이고 효과적이지 않다.
  > 이는 k개의 domain사이에서의 모든 매핑들을 학습하기 위해서는 k(k-1)개의 generators가 학습되어야 하기 때문이다. 아래 그림 참조

<p align="center"><img width="633" alt="스크린샷 2022-03-28 오후 12 00 26" src="https://user-images.githubusercontent.com/56713634/160319247-fbe38559-974e-44a8-88be-5924362d7bde.png">

* cross-domain models
  * 그림 (a)와 같이 4개의 다른 domain간의 이미지 변환을 위해 12개의 네트워크가 필요하다.
  * 각 데이터셋이 부분적으로 라벨링되어 있기 때문에 서로 다른 데이터셋에서 domain을 공동으로 학습할 수 없다.
* StarGAN
  * 모든 가능한 domain사이의 매핑을 하나의 generator(G)를 통해 학습한다.
    > 이미지와 domain 정보를 input으로 넣고 유연하게 이미지를 알맞는 도메인으로 바꾸는 것을 학습한다.
  * 모든 domain의 정보들을 제어할 수 있도록 mask vector를 사용한다.

# Star Generative Adversarial Networks
* Overview
<p align="center"><img width="696" alt="스크린샷 2022-03-28 오후 1 00 43" src="https://user-images.githubusercontent.com/56713634/160324352-2f527e17-7cfb-49a5-85b2-038a56f4499e.png">

## Multi-Domain Image-to-Image Translation
StarGAN의 목표는 여러 domain간의 매핑을 학습하는 G를 학습시키는 것이다.
이를 위해서는 input x를 target domain 라벨 c의 조건에서 output image y로 변환시키도록 G를 학습한다.

### Adversarial Loss
* 생성된 이미지를 실제 이미지와 구별할 수 없도록 Adversarial Loss을 채택
* G는 둘다 조건이 지정된 이미지 G(x,c)를 생성한다.
* input image x 및 target domain c,D는 실제 이미지와 가짜 이미지를 구별하도록 시도한다.

<p align="center"><img width="228" alt="스크린샷 2022-03-28 오후 2 10 21" src="https://user-images.githubusercontent.com/56713634/160330275-6def309a-69d9-4b8b-8d3b-f9b8de8d2134.png"></p>

  * D_scr(x) : Discriminator가 실제 이미지인지 가짜 이미지인지 판별 > 확률분포로 0~1사이의 값을 냄
  * G : loss를 최소화
  * D : loss를 최대화

### Domain Classification Loss
* 주어진 input image x와 target image labal c에 대해 x를 타켓 도메인 c로 분류된 output image y로 변환한다.
  > 이를 만족하기 위해서 auxiliary classifier를 추가하고 D와 G를 최적화한다.

#### 진짜 이미지의 domain classification loss : 판별자를 최적화하기 위함
<p align="center"><img width="170" alt="스크린샷 2022-03-28 오후 2 28 49" src="https://user-images.githubusercontent.com/56713634/160332144-00d86073-2bde-4361-aa0b-0ef1b2b6b290.png"> 
  
    D는 real image x를 그것에 대응하는 original domain c'로 분류시키는 것을 학습한다.

#### 가짜 이미지의 domain classification loss : 생성자를 최적화하기 위함
<p align="center"><img width="192" alt="스크린샷 2022-03-28 오후 2 33 56" src="https://user-images.githubusercontent.com/56713634/160332648-cccaea60-2638-4bd0-8a3d-82663b541327.png">
  
    G는 target domain c로 분류될 수 있는 image를 생성하도록 loss를 최소화하려고 한다.
