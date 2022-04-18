# *Very Deep Convolutional Networks for Large-Scale Image Recognition*
# Abstract
* 3x3 convolution filter를 가진 architecture를 이용하여 depth를 증가시킨 netwo


# INTRODUCTION
* ConvNets가 large-scale image, video recognition에서 큰 활약을 보임
* ConvNets가 컴퓨터 비전 분야에서 많이 사용되면서 Krizhevsky의 original architecture를 개선하려는 많은 시도
  > smaller receptive window size, 첫번째 convolutional layer의 작은 stride 활용
* 본문에서는 depth에 대하여 다룸
* parameter를 고정하고 모든 층에서 3x3 convolution filter를 사용하여 network의 depth를 늘림
  > LSVRC classification와 localisation task에서 높은 정확도를 달성함


# CONVNET CONFIGURATIONS
depth 증가에 따른 ConvNet의 공정한 성능측정을 위해 모든 ConvNet 계층구성은 동일한 원칙을 사용해 설계

## ARCHITECTURE

