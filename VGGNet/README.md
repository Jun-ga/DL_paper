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
*13 convolution layers + 3 Fully-connected layers*

### Feature extractor
* input : 224 x 224 RGB 고정
  > preprocessing : 각 픽셀에서 Training set에서 계산된 mean RGB값을 뺌
* filter : 3x3 convolutional
* stride : 1
* padding : 1
* pooling : 5개의 max pooling
  > 2 x 2 window, stride 2 </br>
  > 모든 conv layer 뒤에 pooling layer가 오는 것은 아님
* ReLu activation functions at the end of layers

### Classifier
* Three Fully-Connected layer *(A stack of conv layer 뒤에)*
* 처음 두개의 FC layer : 4096 channel 
* 마지막 FC layer : 1000개 -> 1000way ILSVRC 분류를 수행, 각 clas마다 한개의 channel을 가짐
* Softmax activation function at last
   
구조사진 찾아서 넣기, 표넣기

## CONFIGURATIONS
<p align="center"><img width="375" alt="스크린샷 2022-04-19 오전 10 39 39" src="https://user-images.githubusercontent.com/56713634/163903178-25114e14-d6ef-4cf2-98dc-d71c0a274c1e.png"></p>
  
### table 1
* configuration의 depth는 A부터 E로 증가하며 layer가 더 추가됨

