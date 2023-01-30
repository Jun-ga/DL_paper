# _DETR : End-to-End Object Detection with Transformers_

# Introduction

* object detection는 category label과 bounding boxes 집합을 예측하는 것
* 현대의 detectos들은 set prediction task를 간접적으로 다루고 있다.
  > a large set of proposals, achors, window enter등을 통해 정의
  > 이러한 예측은 후처리, anchor set 정의, heuristics하게 target boxes to anchors 할당하는데 크게 의존 함
* 본 모델은 이러한 과정을 간소화 하기 위해서 suurogate task를 패스하고 direct set prediction을 수행하는 방법론을 제안
* end-to-end 방법론은 기계번역과 음성식에서는 큰 발전을 보였지만, object 분야에서는 없었다.

[사진 첨부]

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

[사진 첨부]
## Object detection set prediction loss
* DETR은 decoder를 통해서 단 한번의 pass로 고정된 개수인 N개의 예측을 반환
* 이때 N은 image내 전형적인 object의 개수보다 훨씬 커야함
* 학습에서 가장 중요한 것중 하나는 ground truth와 관련하여 최적의 predicted object score를 구하는 것
* loss는 예측한 object과 실제 object간의 최적의 이분매칭진행하여 bounding box별 loss 최적화한다.

[식 첨부]

* y : object의 ground truth set, y^ : N개의 prediction set
* N은 이미지내의 객체 수 보다 많아지므로 y에 N크기가 되게끔 ∅(no object)를 추가(N이 이미지의 object 수보다 크다고 가정하고 y도 ∅(no object)로 채워진 N크기의 set으로 간주)
* 두 set 사이의 매칭 loss 값의 합을 가장 적게 만드는 σ를 찾음 __σ^__ 은 매칭 결과 
  > 하나의 요소가 N요소의 순열에 포함 되는 것을 이분 매칭을 통해 찾았을때, cost가 가장 낮게 된다. 
* L_match : ground truth y와 prediction y_hat의 pair-wise matching cost
  > 일대일 매칭시 loss값  
  
[식 첨부]

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

[사진 첨부]

### Backbone

