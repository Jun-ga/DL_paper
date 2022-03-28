# StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation

## Abstract
* 최근 연구는 2개의 domain에서의 imageto-image translation에서 큰 성공을 보여줬다.
* 하지만 2개 이상의 domain에서는 scalability(확장성)와 robustness(견고성)이 제한되었다.
  > 각 이미지 domain 쌍에 대해 독립적으로 만들어졌기 때문
* __StarGAN은 단 하나의 모델을 사용하여 여러가지 domain에 대해 imageto-image translation을 할 수 있다__ 

## Introdution

<img width="676" alt="스크린샷 2022-03-28 오전 11 14 55" src="https://user-images.githubusercontent.com/56713634/160315357-3832fc06-5ecf-4e3e-baae-54960ca7c148.png">

* CelebA와 RaFD dataset을 이용하여 얼굴의 특징과 표정을 변화시킴
* 이러한 라벨링 데이터셋을 이용한 multi-domain image translation은 기존 모델에서 비효율적이고 효과적이지 않다.
  > 이는 k개의 domain사이에서의 모든 매핑들을 학습하기 위해서는 k(k-1)개의 generators가 학습되어야 하기 때문이다. 아래 그림 참조

<img width="633" alt="스크린샷 2022-03-28 오후 12 00 26" src="https://user-images.githubusercontent.com/56713634/160319247-fbe38559-974e-44a8-88be-5924362d7bde.png">

* cross-domain models
  * 그림 (a)는 4개의 다른 domain간의 이미지 변환을 위해 12개의 네트워크가 필요하다.
