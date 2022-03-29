# __StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation__

# Abstract
* 최근 연구는 2개의 domain에서의 image-to-image translation에서 큰 성공을 보여줬다.
* 하지만 2개 이상의 domain에서는 scalability(확장성)와 robustness(견고성)이 제한되었다.
  > 각 이미지 domain 쌍에 대해 독립적으로 만들어졌기 때문
* __StarGAN은 단 하나의 모델을 사용하여 여러가지 domain에 대해 image-to-image translation을 할 수 있다__ 

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
<p align="center"><img width="696" alt="스크린샷 2022-03-28 오후 1 00 43" src="https://user-images.githubusercontent.com/56713634/160324352-2f527e17-7cfb-49a5-85b2-038a56f4499e.png"></p>

* (a) 진짜 이미지와 가짜 이미지를 구별, 진짜일때 그것과 상응하는 domain을 분류
* (b) input으로 target domain과 이미지를 받고 가짜이미지 생성
* (c) Original domain label을 가지고 가짜이미지를 다시 원래 이미지로 복원
* (d) 진짜 이미지와 구분할 수 없고 D에 의해 target domain으로 분류 가능한 이미지를 생성하도록 한다.

## Multi-Domain Image-to-Image Translation
StarGAN의 목표는 여러 domain간의 매핑을 학습하는 G를 학습시키는 것이다.
이를 위해서는 input x를 target domain label c의 조건에서 output image y로 변환시키도록 G를 학습한다.

### Adversarial Loss
* 생성된 이미지를 실제 이미지와 구별할 수 없도록 Adversarial Loss을 채택
* G는 라벨 조건이 지정된 이미지 G(x,c)를 생성한다.
* D는 실제 이미지와 가짜 이미지를 구별하도록 시도한다.

<p align="center"><img width="228" alt="스크린샷 2022-03-28 오후 2 10 21" src="https://user-images.githubusercontent.com/56713634/160330275-6def309a-69d9-4b8b-8d3b-f9b8de8d2134.png"></p>

  * D_scr(x) : Discriminator가 실제 이미지인지 가짜 이미지인지 판별 > 확률분포로 0~1사이의 값을 냄
  * G : loss를 최소화
  * D : loss를 최대화

### Domain Classification Loss
주어진 input image x와 target image labal c에 대해 x를 target domain c로 분류된 output image y로 변환한다.

_예를 들어 셀럽 이미지 x를 target domain(금발머리) c로 변환하는 것_
> 이를 만족하기 위해서 auxiliary classifier를 D위에 추가하고 D와 G를 최적화한다.

* 진짜 이미지의 domain classification loss : 판별자를 최적화하기 위함
<p align="center"><img width="170" alt="스크린샷 2022-03-28 오후 2 28 49" src="https://user-images.githubusercontent.com/56713634/160332144-00d86073-2bde-4361-aa0b-0ef1b2b6b290.png"></p> 
  
    D는 real image x를 그것에 대응하는 original domain c'로 분류시키는 것을 학습한다.

* 가짜 이미지의 domain classification loss : 생성자를 최적화하기 위함
<p align="center"><img width="192" alt="스크린샷 2022-03-28 오후 2 33 56" src="https://user-images.githubusercontent.com/56713634/160332648-cccaea60-2638-4bd0-8a3d-82663b541327.png"></p>
  
    G는 target domain c로 분류될 수 있는 image를 생성하도록 학습한다.
  

### Reconstruction Loss
위에서 소개한 loss만으로는 input image의 target domain에 관련한 부분만을 변화시킬때 input image의 본래 형태를 잘 보존할 수 없다. 따라서 generator에 loss를 하나 더 적용한다.
> CycleGAN에서 사용한 cycle-consistency loss 

<p align="center"><img width="210" alt="스크린샷 2022-03-28 오후 2 47 10" src="https://user-images.githubusercontent.com/56713634/160334221-f7053dcc-b9b8-43e8-bc56-b364833b6c41.png"></p>

* G는 변환된 image G(x,c)와 original domain label c'를 input으로 받고 original image x를 복원하려 한다.
* L1 norm으로 계산한다.

### Full Objective

<p align="center"><img width="222" alt="스크린샷 2022-03-28 오후 2 51 41" src="https://user-images.githubusercontent.com/56713634/160334591-69c183f9-11af-40bf-95f2-345082d9c0d9.png"></p>

* λ_cls와 λ_rec는 domain 분류와 reconstruction loss들의 상대적 중요도를 control
* λ_cls = 1, λ_rec = 10을 사용한다.

## Training with Multiple Datasets
StarGAN은 서로 다른 domain을 가진 dataset을 동시에 포함할 수 있다.
> CelebA의 머리색 label을 RaFD dataset에 적용할 수 있다.

그러나 다수의 dataset을 학습시킬때 원하는 label 정보가 각 dataset에 부분적으로만 있다는 문제가 발생한다.
> 모든 데이터 셋이 동등하게 label을 가지고 있는 것이 아니라 어떤 데이터셋은 특정 label만 가지고 있고 다른 데이터 셋은 그 특정 label만을 가지고 있다는 문제

> 변환된 image G(x,c)에서 input image x를 reconstruction을 하려면 label vector c'에 대한 완전한 정보가 필요하기 때문에 문제가 된다.

### Mask Vector
위의 문제를 해결하기 위한 방안으로 StarGAN이 명시되지않은 label에 대해서는 무시하고 명시된 label에 대해 집중하게 해준다.
> n차원의 one-hot vector를 사용한다.
<p align="center"><img width="100" alt="스크린샷 2022-03-28 오후 3 24 45" src="https://user-images.githubusercontent.com/56713634/160338468-75d89ba5-b847-4ef8-854c-4613d5705bbf.png"></p>

* c_i는 i번째 데이터셋의 라벨에 대한 vector
* 알려져 있는 label의 vector c_i는 binary attributes에 대해서는 binary vector로 표현될 수 있고 카테고리 attributes에 대해서는 one-hot vector로 표현될 수 있다.
* 남겨진 n-1개의 알려지지 않은 label에 대해서는 zero값으로 지정한다.
> 본 논문에서는 CelebA와 RaFD dataset을 이용했으므로 n=2가 된다.

### Training Strategy
* generator는 알려지지 않은 라벨에 대해 무시를 하게 되므로 확실하게 주어진 라벨에 초점을 맞춰 학습하게 된다.
* generator의 구조는 input label의 차원을 제외하고는 하나의 데이터셋에 대해 학습할 때와 같은 구조이다.
* discriminator는 classification error만을 최소화 한다.
  > CelebA 데이터셋 이미지에 대해 학습할 때에는 disciminator가 CelebA attributes(성별, 머리색)에 대해서만 classification error를 최소화하게 되고 RaFD의 표정과 같은 특징들은 무시한다.
  
  > 다양한 데이터셋을 번갈아가며 학습하며 discriminator는 모든 데이터셋에 관한 discriminative 특징들을 학습하고 generator는 모든 라벨들을 제어하는 것에 대해 학습한다.

## Implementation
### Improved GAN Training
학습 과정을 안정화시키고 더 향상시키기 위해서 gradient penalty(λ_gp =10)와 Wasserstein GAN의 objective function을 사용하였다.

<p align="center"><img width="253" alt="스크린샷 2022-03-28 오후 3 55 35" src="https://user-images.githubusercontent.com/56713634/160342640-9fc160b5-665a-4beb-9160-ceb897390989.png"></p>

### Network Architecture
CycleGAN의 architecture를 baseline으로 사용

* 2개의 convolutional layers로 구성된 generator network이용
* G만 instance normalization(D는 X)

## Experiments
* 기존의 cross-domain model들은 각 domain의 특성의 적용이 잘 안되는 반면 StarGAN은 특성들의 적용이 좋다.
<p align="center"><img width="682" alt="스크린샷 2022-03-28 오후 4 02 56" src="https://user-images.githubusercontent.com/56713634/160343733-0b08906a-5c67-4d41-b825-6770f5889224.png"></p>

### CelebA & RaFD
<p align="center"><img width="922" alt="스크린샷 2022-03-29 오후 6 41 35" src="https://user-images.githubusercontent.com/56713634/160583055-b3acae2c-d232-49cd-a80a-e11e9d921dd5.png"></p>

### Maskvector
* maskvector가 제대로 적용된 경우와 제대로 적용되지않은 경우의 비교
<p align="center"><img width="586" alt="스크린샷 2022-03-28 오후 4 14 33" src="https://user-images.githubusercontent.com/56713634/160345551-63f486e9-6757-492e-9a63-2ce359c1e138.png"></p>

## Test with Multiple Datasets
<p align="center"><img width="867" alt="스크린샷 2022-03-29 오후 6 38 17" src="https://user-images.githubusercontent.com/56713634/160582385-3a47677d-2e4c-4b41-8c50-b1dac015bd5b.png"></p>


## Conclusion
본 논문에서는 하나의 generator와 discriminator으로 multi-domain image translation을 가능하게 한 StarGAN을 제안했습니다.
이는 scalability(확장성), robustness(견고성)할 뿐만 아니라 기존의 모델의 비해 high visual quality 의 이미지를 생성하도록 했습니다.
