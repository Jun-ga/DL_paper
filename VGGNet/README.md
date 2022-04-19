# *Very Deep Convolutional Networks for Large-Scale Image Recognition*
# ABSTRACT
* 3x3 convolution filter를 가진 architecture를 이용하여 depth를 증가시킨 network


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

<p align="center"><img width="375" alt="1__Lg1i7wv1pLpzp2F4MLrvw" src="https://user-images.githubusercontent.com/56713634/163918594-324779dc-7568-43e8-a8fd-befa6109488c.png">
</p>

## CONFIGURATIONS
<p align="center"><img width="375" alt="스크린샷 2022-04-19 오전 10 39 39" src="https://user-images.githubusercontent.com/56713634/163903178-25114e14-d6ef-4cf2-98dc-d71c0a274c1e.png"></p>
  
### table 1
* 본 논문에서 평가된 ConvNet 구성
* configuration의 depth는 A부터 E로 증가하며 layer가 더 추가됨
* 첫번째 layer 64개에서 마지막 layer 512개에 도달할 때까지 max pooling 이후에 2배씩 증가

### table 2
* 각 구성에 대한 parameterS
* 더 depth가 깊지만 weight의 수는 많지 않음

## DISCUSSION
* network 전체에 여러 개의 3x3 필터를 사용함 
* layer 사이에 pooling이 없는 2개의 3x3 conv layer stack이 5x5 receptive field를 가짐
* 3개의 3x3 conv layer는 7x7 receptive field를 가짐
* __단일 7x7 layer 대신 3개의 3x3 conv layer를 사용하면서 얻는 이점__
  * 네트워크의 nonlinearity를 늘릴 수 있다. *(결정함수의 비선형성 증가 > 의사결정 기능을 더욱 차별화)*
  * parameter(weight)의 개수를 줄일 수 있다.

# CLASSIFICATION FRAMEWORK
## TRAINING
* batch size: 256, momentum: 0.9
* weight decay에 의해 정규화, dropout은 처음 두개의 FC층에서 일어남(0.5)
* learing rate의 경우 초기에 10^(-2)로 설정, val셋의 acc가 10번 안에 좋아지지않으면 낮아짐
  > 총 3배 감소, 370K 반복 이후 학습 중단(74 epochs)

* __VGGNet의 많은 파리미터와 깊은 층에 비교하여 적은 epoch을 요구__
  > 깊은 층과 작은 컨볼루션 필터사이즈에 의해 시행되는 암시된 정규화와 몇몇층에서 시행되는 사전 초기화 때문

* 네트워크의 가중치 초기화가 가장 중요
  * Deep net에서의 gradient의 불안정성으로 인한 학습 지연을 막기 위해 network 가중치의 초기화가 중요, 학습 지연 막기 위해 무작위 초기화로 교육될 정도로 얕은 구성 A (Table 1)부터 학습 시작
  * 더 깊은 architecture을 훈련할 때, 처음 네 개의 conv layer과 마지막 세 개의 FC layer를 net A의 layer로 초기(중간 layer들은 무작위로 초기화)
  * 무작위로 초기화하기 위해 평균은 0, 분산은 10^(-2)인 정규분포에서 가중치를 sampling biases는 0으로 초기화
* 224x224 크기의 ConvNet input image 얻기 위해 rescaled training image를 무작위로 crop
  > SGD(Stochastic Gradient Descent) iteration 1회당 image 1개 crop
* Training set를 더 augment 하기 위해 crop 시 random horizontal flipping과 random RGB colour shift

* 이미지를 512x512 크기로 변환 후 224x224 크기로 샘플링한 경우 예시
<p align="center"><img width="672" alt="스크린샷 2022-04-19 오후 12 37 37" src="https://user-images.githubusercontent.com/56713634/163915305-e29e619f-80d7-495c-8615-46d5f5ea42ef.png"></p>

### Training image size
* 고정된 224x224 사이즈의 input을 얻기 위해 rescaled training image에서 random하게 crop
* Training scale S: the smallest side of an isotropically-rescaled(원본과 동일한 비가 되도록 scaling) training image 즉, rescaled training image의 가장 작은 쪽 픽셀 수
* Training scale S를 설정하기 위한 두 가지 접근법
  * Single-scale training, S 고정 (S=256, S=384)
    > S=256을 먼저 사용하여 network를 훈련한 다음 S=384 network의 속도 향상을 위해 S=256으로 pre-trained 된 가중치로 초기화하고 더 작은 learning rate인 10^(-3)  사용
  * Multi-scale training
    > [Smin, Smax ] (Smin =256, Smax =512) 에서 무작위로 S 추출해 각 training image를 개별적으로 축소하는 것</br>
    > 원본 image에 있는 물체는 random한 size를 가져 더 다양한 input을 줄 수 있음(scale-jittering)</br>
    > Image에 있는 object가 크기가 다를 수 있기 때문에 training 동안에 고려하는 것이 좋음</br>
    > *(속도 상의 이유로 S=384로 pre-trained된 동일한 configuration을 가진 single-scale model의 모든 layer들을 fine-tuning함으로써 multi-scale model을 train)*


## TESTING
* Input image는 pre-define된 smallest image side로 isotropically rescale되며 test scale Q로 표시됨
   * Q가 training scale S와 같을 필요 없음, 각 S에 대해 몇 가지 Q를 쓰는 것이 성능 향상에 도움

* Fully Connected layer가 conv layer로 변환됨 (첫 번째 FC layer는 7x7 conv layer로 마지막 두 FC layer는 1x1 conv layer로 변환)
  > resulting Fully Convolutional network가 전체 image에 적용
  * class의 개수와 동일한 개수의 channel을 갖는 class score map
  * Input image size에 따라 변하는 spatial resolution 〓 input image size의 제약이 없어짐
  * 하나의 image를 다양한 scale로 사용한 결과를 조합해 image classification accuracy 개선가능
* Image의 class score의 fixed-size vector을 얻기 위해 class score map은 spatially averaged (sum-pooled)
* Image를 horizontal flipping해서 test set을 augment
* Image의 final score을 얻기 위해 원본과 flipped image의 soft-max class의 평균을 구함
* Multi-crop evaluation: input image를 더 정밀하게 sampling시켜서 정확도 향상되지만 각 crop에 대해 network re-computation이 필요해 효율성 떨어짐

## IMPLEMENTATION DETAILS
* 4개의 GPU 사용
* 전체 batch의 gradient를 얻기 위해 각 GPU batch gradients는 계산되어 평균
* 모든 GPU에서 동시에 gradient가 계산되어 한개의 GPU에서 얻는 결과와 정확히 일치
* 4개의 GPU를 사용함으로써 3.75배 속도 향상

# CLASSIFICATION EXPERIMENTS
ILSVRC-2012 dataset에 대해 설명된 ConvNet architecture가 이뤄낸 image classification 결과를 제시 해당 데이터셋에는 1000 클래스가 넘는 이미지를 포함하고 있으며 분류성능은 top-1과 top-5로 평가

## SINGLE SCALE EVALUATION

* test image size
<p align="center"><img width="539" alt="스크린샷 2022-04-19 오후 12 50 48" src="https://user-images.githubusercontent.com/56713634/163916651-7fb6e0d7-5648-4e1c-aaf4-8067a328b3d0.png"></p>

<p align="center"><img width="347" alt="스크린샷 2022-04-19 오후 12 48 44" src="https://user-images.githubusercontent.com/56713634/163916459-63e1629c-002c-4aca-9408-9cba57d3f16b.png"></p>

* B < C : non-linearity가 성능 향상에 도움
* C < D : 같은 depth이지만 성능이 떨어짐 -> conv filter를 사용하여 spatial context 파악 중요
* Fixed S(256, 384) 때보다 scale jittering at training time S∈Smin, Smax  가
성능 좋음 -> scale jittering에 의한 training set augmentation이 multi-scale image
statistics를 파악하는데 도움

## MULTI-SCALE EVALUATION
<p align="center"><img width="337" alt="스크린샷 2022-04-19 오후 12 55 42" src="https://user-images.githubusercontent.com/56713634/163917093-519565b0-d509-4f92-8ab8-0760acf9d07e.png"></p>

* test에서의 scale jittering 효과 확인
* single-scale보다 multi-scale에서 성능향상

## MULTI-CROP EVALUATION

<p align="center">
<img width="348" alt="스크린샷 2022-04-19 오후 12 57 39" src="https://user-images.githubusercontent.com/56713634/163917274-51dad5fc-1377-4495-a67e-3bde684780df.png">
</p>
* Dense evaluation과 mult-crop evaluation 비교
* muit-crop evalution이 연산량이 많지만 더 좋은 결과
* 상호 보완적인 관계이기때문에 둘을 조합해서 사용하면 더 좋은 결과

## COMPARISON WITH THE STATE OF THE ART
<p align="center">
<img width="361" alt="스크린샷 2022-04-19 오후 1 02 19" src="https://user-images.githubusercontent.com/56713634/163917802-f1e22b07-472e-40f3-9823-03252f5115dc.png">
</p>
* 더 단순하고 효과적인 방법으로 인하여 GoogLeNet보다 많이 사용

# CONCLUSION
* large scale image classification을 위한 deep CNN을 개발
* 더 복잡한 네트워크보다 더 좋은 성능을 가짐
* 기존 ConvNet architecture보다 작은 receptive field 사용(3x3, 1 stride)
* 최대 19 depth까지 weight layer를 deep하게 설계하여 좋은 성능 이끌어냄




