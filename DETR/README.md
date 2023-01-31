# _DETR : End-to-End Object Detection with Transformers_

# Introduction

* object detection는 category label과 bounding boxes 집합을 예측하는 것
* 현대의 detectos들은 set prediction task를 간접적으로 다루고 있다.
  > a large set of proposals, achors, window enter등을 통해 정의
  > 이러한 예측은 후처리, anchor set 정의, heuristics하게 target boxes to anchors 할당하는데 크게 의존 함
* 본 모델은 이러한 과정을 간소화 하기 위해서 suurogate task를 패스하고 direct set prediction을 수행하는 방법론을 제안
* end-to-end 방법론은 기계번역과 음성식에서는 큰 발전을 보였지만, object 분야에서는 없었다.

<p align="center"><img width="381" alt="스크린샷 2023-01-31 오후 4 33 20" src="https://user-images.githubusercontent.com/56713634/215695263-69971ce5-a64f-44b1-9d47-c48cfd920c3d.png"></p>


* object 문제를 direct set prediction으로 바라 본다.
* transformer에서 기반한 decoder-encoder 구조를 사용
* self-attention mechanism은 이 구조가 removing duplicate prediction과 같은 set prediction의 제약을 특히 다루기 쉽게 만들어줌
* 한번에 모든 object를 예측하기 위해, 예측 object와 ground-truth object 사이의 양자간 매칭(bipartite matching) 을 수행하는 a set loss function(여러 개을 통해 end-to-end로 학습

추가로 작성해야하는데 아이디어가 안떠오름 일단 비워두기


# Related work

## Set Prediction
* 대부분의 dectector들은 NMS와 같은 post processing을 제거하기 위해 모든 예측에 대하여 상호작용을 모델링하는 global inference가 필요함
  > autoregressive sequence model을 사용
* 본 모델은 post processing 방법을 사용하지않으며, 이분매칭을 통해 Set prediction을 정의한다.

## Transformers and Parallel Decoding
* transformer는 기계번역을 위한 것으로 소개됨
* Attention mechanisms은 전체 입력 squence의 모든 정보를 이용
* transformers는 self-attention layer를 가지고 있고 이것은 각각의 요소들을 검사하여 집합정보 layer를 업데이트 하기에 non-local 네트워크와 비슷
* global한 계산으로 완벽한 메모리를 사용 -> 긴 sqeunce 모델에 적합(RNN)
* transformers는 현재 NLP,음성처리, 컴퓨터 비전 분야에서 RNN의 문제점을 대체할 수 있는 수단
  > seq2seq 모델에 자기추론 모델을 사용
* 본 모델에서도 transformer과 병렬 decoding을 결합하여 사용

## Object detection
* 2-stage detectors : proposal을 이용
* 1-stage detectors : anchor 혹은 object center grid을 이용
  > detector의 최종 성능은 초기 추축에 따라 좌우됨
* 본 모델은 anchor가 아닌 input image를 사용하여 absolute box prediction으로 detection set를 직접 예측함으로써 hand-crafted process를 제거하여 detection process의 효율성을 높임

### Set-based loss
몇몇 object detector는 bipartite matching loss를 사용함, 하지만 최근 model은 NMS와 같은 post processing을 통해 성능을 향상시킴
__direct set losses를 사용한다면, post-processing step은 필요하지않음__

### Recurrent detectors
* DETR에 가장 근접한 것은 end-to-end set prediction
* CNN activation에 기반한 인코더-디코더 아키텍처와 이분 매칭 손실을 사용하여 directly predict set을 수행한다. 
* 그러나 RNN(자기 회귀 모델)을 기반으로 하기 때문에 병렬 디코딩을 수행하지 못한다.

# The DETR model
direct set prediction에 필수적인 요소는 아래와 같다
1) Prediction와 ground truth boxes 사이의 고유한 매칭을 위한 set prediction loss
2) Object set을 예측하고 이들의 relation을 모델링하는 아키텍쳐

<p align="center"><img width="381" alt="스크린샷 2023-01-31 오후 4 34 26" src="https://user-images.githubusercontent.com/56713634/215695509-c8e97ee2-8e6a-41c7-8818-434b1e469fb2.png"></p>


## Object detection set prediction loss
* DETR은 decoder를 통해서 단 한번의 pass로 고정된 개수인 N개의 예측을 반환
* 이때 N은 image내 전형적인 object의 개수보다 훨씬 커야함
* 학습에서 가장 중요한 것중 하나는 ground truth와 관련하여 최적의 predicted object score를 구하는 것
* loss는 예측한 object과 실제 object간의 최적의 이분매칭진행하여 bounding box별 loss 최적화한다.

<p align="center"><img width="160" alt="스크린샷 2023-01-31 오후 4 35 34" src="https://user-images.githubusercontent.com/56713634/215695704-fd5bf9a6-0b5e-4e18-a44b-d776bd177be1.png"></p>

* y : object의 ground truth set, y^ : N개의 prediction set
* N은 이미지내의 객체 수 보다 많아지므로 y에 N크기가 되게끔 ∅(no object)를 추가(N이 이미지의 object 수보다 크다고 가정하고 y도 ∅(no object)로 채워진 N크기의 set으로 간주)
* 두 set 사이의 매칭 loss 값의 합을 가장 적게 만드는 σ를 찾음 __σ^__ 은 매칭 결과 
  > 하나의 요소가 N요소의 순열에 포함 되는 것을 이분 매칭을 통해 찾았을때, cost가 가장 낮게 된다. 
* L_match : ground truth y와 prediction y_hat의 pair-wise matching cost
  > 일대일 매칭시 loss값  
  
<p align="center"><img width="313" alt="스크린샷 2023-01-31 오후 4 35 39" src="https://user-images.githubusercontent.com/56713634/215695749-e510eb6a-ff51-4e88-962a-d75bba1c0c0a.png"></p>

* Hungarian algorithm은 가능한 모든 경우의 매칭 loss를 작게 만들 수 있는 경우를 찾는 방법
* ground truth요소를 yi = (c_i, b_i)라 할때, c_i = class label b_I = bounding box의 중심좌표, 높이, 너비인 4차원 정보
* prediction요소는 class의 확률값 : p^ bounding box의 확률값 : b^
* L_box : bounding box가 얼마나 유사한지에 대한 loss값
* class가 no object일때, 가중치에 10배의 감소를 준다.
  > 이때, prediction에 영향이 없기 때문에 cost는 일정
  > class의 불균형을 해결하기 위함

### Bounding box loss
Region proposal이나 anchor 기반으로 bounding box를 예측하는 기존 방식과 달리 directly box prediction을 수행한다. Scale을 고려하기 위해 prediction과 ground truth의 차이에 대해 L1 loss와 GIoU를 함께 사용한다.
> box를 예측할때 L1 loss는 큰상자와 작은상자의 오차가 유사하더라도 다른 scale을 가지기 때문에 L1 loss와 GIOU loss를 조합하여 scale 을 다양하지 않게 만듦으로 이런 문제를 완화함

## DETR architecture
feature 추출을 위한 CNN backbone, encoder-decoder 구조의 transformer, 최종 detection prediction을 수행하는 FFN(Feed Forward Network)로 총 세가지 구성요소로 이뤄져있다.

<p align="center"><img width="381" alt="스크린샷 2023-01-31 오후 4 34 26" src="https://user-images.githubusercontent.com/56713634/215695509-c8e97ee2-8e6a-41c7-8818-434b1e469fb2.png"></p>

### Backbone
* feature extraction을 하는 cnn 구조 사용
* 본문에서는 ImageNet으로 학습한 ResNet50 혹은 ResNet101
* channels=2048, height과 width는 input의 1/32의 feature map 생성



### Transformer encoder

* 1x1 convolution으로 feature map의 channel을 압축하여 feature map을 sequence data로 변형
  > dxWH 차원의 새로운 feature map으로 변형됨
* loss 계산할 때 permutation 구하므로 invariant 하도록 각 attention layer의 인풋으로 positional encoding을 추가

### Transformer decoder

* permutation-invariant를 위해 positional encoding이 사용됨
* object query : 학습이 가능한 positional embedding
* 각 embedding은 각기 다른 object를 의미
  > output은 FFN을 통해 N개의 box coordinates와 class label
* ground-truth의 object와 동일한 개수의 object를 예측할 수 있도록 FFN과 위의 Hungarian loss는 각 decoder layer에 계산됨
  > FFN은 parameter을 공유된다.
 
### Transformer Feed-forward
* N개의 decoder output을 독립적인 input으로 받아서 각각의 class와 bounding box 값을 예측
* network는 3-layer의 perceptron과 ReLu activation function으로 구성
* 만약 클래스가 없다면 "no object"로 결과가 산출되고 이미지 상 Background 


# Experiments
COCO 2017 detection을 사용하여 Faster R-CNN과 정량적으로 비교

## Comparison with Faster R-CNN

<p align="center"><img width="382" alt="스크린샷 2023-01-31 오후 4 37 57" src="https://user-images.githubusercontent.com/56713634/215708268-ee5cf30a-d4e9-407b-8054-2f7ddedc21cf.png"></p>

* 큰 object의 경우 DETR이 잘 구분하지만, 작은 object는 상대적으로 구분하지못함

## Ablations
### Number of encoder layer
<p align="center"><img width="377" alt="스크린샷 2023-01-31 오후 4 38 04" src="https://user-images.githubusercontent.com/56713634/215709699-2c0c8269-0403-4ca8-9f53-1cf186844f67.png"></p>

* encoder layer의 증가로 AP가 증가함
  > global한 scene reasoning을 수행해서 object 들을 구별하는 데에 중요한 역할

<p align="center"><img width="379" alt="스크린샷 2023-01-31 오후 4 38 10" src="https://user-images.githubusercontent.com/56713634/215709965-6571cf69-2451-4ca6-8704-d30921afc73c.png"></p>

* encoder의 self-attention 과정으로 activation map을 시각화한 것
* 다음과 같이 인스턴스가 잘 나눠진다면 디코더 파트에서 예측하는 것이 쉬워짐

### Number of decoder layers.
object를 detection할 때, decoder의 단계가 매우 중요
  > 충분한 layer를 가진 decoder가 좋은 성능을 가지게 함
  
<p align="center"><img width="204" alt="스크린샷 2023-01-31 오후 4 38 16" src="https://user-images.githubusercontent.com/56713634/215711142-b38a992d-1b64-411d-9f39-34f31fc1096a.png"></p>

<p align="center"><img width="358" alt="스크린샷 2023-01-31 오후 4 38 26" src="https://user-images.githubusercontent.com/56713634/215712311-8d16d6ff-e592-4b90-83a5-0ddb1fa7a89f.png"></p>
* decoder attention은 상당히 지역적
* object의 head나 legs 쪽을 attention
* 인코더가 global하게 attention하여 object의 인스턴스를 분리하는 반면 디코더는 object의 경계를 추출하기 위해 head와 legs에 attention한다

### Importance of positional encodings
상대적인 위치 정보를 처리하기 위해 본 모델에는 두가지 종류의 positional encodings이 존재 spatial positional과 output positional 

<p align="center"><img width="375" alt="스크린샷 2023-01-31 오후 4 38 34" src="https://user-images.githubusercontent.com/56713634/215713268-77684e19-e3e7-41be-98df-262aadefcfbd.png"></p>

* positional encoding시 가장 성능이 좋음

### Loss ablations
Loss function에서 L1 loss와 GIOU loss의 영향을 파악하기 위한 실험
이 결과를 통해 본 논문에서 사용하는 loss function의 타당성을 부여
<p align="center"><img width="358" alt="스크린샷 2023-01-31 오후 4 38 40" src="https://user-images.githubusercontent.com/56713634/215714696-d96db132-867f-421f-a97f-319e19c6eca0.png"></p>

## Analysis
각각의 object query가 개별적으로 역할을 수행 이를 아래와 같이 시각화
<p align="center"><img width="374" alt="스크린샷 2023-01-31 오후 4 38 48" src="https://user-images.githubusercontent.com/56713634/215715186-c5cbdd2e-ef92-4ef4-80f2-e7d265994255.png"></p>

* 점들은 각각의 bounding box의 center값에 해당
* 각각 서로 다른 different area와 box size를 가지게 되도록 학습
  >  N개의 object query는 이미지가 주어졌을 때 각각 다른 영역과 박스 크기에 관심
  
## DETR for panoptic segmentation
DETR은 panoptic segmentation에 이용 될 수 있는데 원래 DETR의 구동 방식 대로, Box를 예측하고 mask head를 달아서 segmentation을 진행한다.
  > panoptic segmentation : stuff와 thing을 모두 segmentation하는 vision의 task
  
<p align="center"><img width="393" alt="스크린샷 2023-01-31 오후 4 38 56" src="https://user-images.githubusercontent.com/56713634/215715954-1327e10c-51a7-47a7-9553-b8c53c4dad02.png"></p>

<p align="center"><img width="374" alt="스크린샷 2023-01-31 오후 4 39 03" src="https://user-images.githubusercontent.com/56713634/215715715-5f6ad1c2-0289-47d1-a1df-c1ee8ef970ad.png"></p>

# Conclusion
 Object detection 분야에서 end-to-end 방식의 새로운 구조를 제안했다. Partite matching (이분매칭) & Transformer encoder-decoder architecture를 활용하여 큰 object에 대해서는 탐지 성능이 좋지만 작은 object에 대해서는 상대적으로 안 좋은 탐지 성능을 보였다.

