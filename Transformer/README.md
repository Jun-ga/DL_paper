# *Transformer : Attention Is All You Need*

# Abstract

* RNN과 CNN을 완전 배제하고 attention 메커니즘만을 사용하는 단순한 architecture인 Transformer을 제안
* 병렬화가 용이하고 적은 학습시간으로 성능이 높다.
* 다른 task에 대해서도 잘 동작함


# Introduction

* RNN, LSTM, GRU는 sequence modeling과 transduction problem같은 언어 modeling 및 기계 번역에서 SOTA로 제안됨
* Recurrent models은 이전 결과를 입력으로 받는 순차적인 특성으로 인하여 병렬처리가 어려움
  > 최근의 연구는 factorization tricks, conditional computation을 통해 효율적인 연산처리를 진행했지만 아직 순차적 제약 발생
* Attention mechanisms은 input, output sequence의 길이에 상관하지않음 하지만 여전히 __recurrent network와 함께 사용됨__
* 본 모델은 recurrent network를 제거하고 input과 ouput간의 global dependency를 뽑아내기 위해 attention mechanism만을 사용하는 __transformer__ 를 제안한다.
  > 이 모델은 8개의 P100 GPU로 12시간동안 학습하여 병렬처리와 SOTA를 달성한다.


# Model Architecture
transformer은 self-attention과 position-wise를 따르며, 크게 Encoder와 Decoder로 구성되어있다.
<p align="center"><img width="286" alt="스크린샷 2023-01-17 오후 8 58 32" src="https://user-images.githubusercontent.com/56713634/212893467-5078ac29-5db2-401d-8fe3-7587d7e126ed.png"><p>


## Encoder and Decoder Stacks
### Encoder

<p align="center"><img width="180" alt="스크린샷 2023-01-17 오후 9 03 24" src="https://user-images.githubusercontent.com/56713634/212894249-22aef785-b6e1-4853-827d-b03d1a36639e.png"><p>  
 
* N = 6개의 layer의 stack으로 구성되어있음 이때, 각 layer는 2개의 sub-layer가 있다.
* 첫번째 layer은 multi-head self-attention mechanism, 두번째 layer은 position-wise fully connected feed-forward network
* 이 2개의 sub-lalyer에 각각 residual connection과 normalization을 적용  
  >  residual connection을 구현하기 위하여 모든 sub-layer(embedding layers 포함)의 output은 512
 
### Decoder

 <p align="center"><img width="185" alt="스크린샷 2023-01-17 오후 9 03 34" src="https://user-images.githubusercontent.com/56713634/212894338-75385eb2-0a3d-401c-a82a-d8f70fc73f7c.png"><p>  
  
* Encoder와 동일하게 N = 6개의 layer의 stack으로 구성되어있음
* 2개의 sub-layer에 세번째 sub-layer를 추가하여 encoder stack의 output에 대해 muti-head attention을 수행
* decoder stack의 self-attention sub-layer를 수정하여 position이 subsequent positions에 attending하는 것을 막음 __masking__
* residual connection과 normalization을 적용

## Attention
   
<img width="626" alt="스크린샷 2023-01-18 오후 12 16 37" src="https://user-images.githubusercontent.com/56713634/213075099-572d07a6-eae5-4a10-81fd-b626de4f1171.png">
   
#### 동작 방식
__encoder__   
<p align="center"><img width="570" alt="스크린샷 2023-01-18 오전 11 35 11" src="https://user-images.githubusercontent.com/56713634/213068164-9317562b-edb6-4904-b30d-dc339eae4245.png"><p> 
   
* time step에 나 는 학생 이다 4개의 단어 입력
* 그에 따라 4개의 hidden state vector
* 각 단어의 hidden state vector를 모두 이용하여 입력된 단어와 같은 수의 vector을 얻을 수 있음

__decoder__
<p align="center"><img width="681" alt="스크린샷 2023-01-18 오전 11 35 17" src="https://user-images.githubusercontent.com/56713634/213068232-af742f1a-2edf-45c9-9897-9103e309f8af.png"><p>
  
__attention__
<p align="center"><img width="659" alt="스크린샷 2023-01-18 오전 11 35 24" src="https://user-images.githubusercontent.com/56713634/213068277-e8f89485-eb5c-441c-baaa-b4dcb2fb65ca.png"><p>
  
#### Attention weight layer
encoder가 출력하는 단어의 hidden state vector에 주목하여 가중치 a 구하는 부분
 > 즉, 디코더에서 출력된 hidden state vector(h)와 인코더에서 넘어온 hidden state vector(hs)를 이용하여 가중치 (a)를 구하는 단계
 
__어떻게 집중하는데? -> 각 단어의 가중치(중요도 or 기여도)를 계산하여 반영__
  
<p align="center"><img width="493" alt="스크린샷 2023-01-18 오전 11 45 04" src="https://user-images.githubusercontent.com/56713634/213069771-a527b79a-b6de-4eab-b364-c3e4877c3be5.png"><p>  
  
* Attention weight layer에서 출력된 vector를 __Attention score__
* Attention score가 클수록 유사도가 높다.
* Attention score값을 바로 사용하지않고 한번 정규화(softmax)를 적용하여 사용
 
  
#### Weight Sum layer
a와 hidden state의 가중합을 구하여 context vector 출력하는 부분
  
<p align="center"><img width="589" alt="스크린샷 2023-01-18 오전 11 47 44" src="https://user-images.githubusercontent.com/56713634/213070111-904cb1f5-9e52-4c18-a378-c5113ef12804.png"><p>  
  
### Scaled Dot-Product Attention

  

<p align="center"><img width="176" alt="스크린샷 2023-01-16 오후 10 59 37" src="https://user-images.githubusercontent.com/56713634/212894466-ae57f911-bad8-4b34-92b1-bfcac00d5106.png"><p> 

* input : Query(Q), Key(K), Value(V)
  
1. 찾고 싶은 Q 입력
2. K-V를 통해 유사도 계산
3. 유사도를 확률로 변환
4. 유사도 깂들의 가중합을 최종 결과로 반환

#### Attention Function
weight는 Q와 K의 조합으로 계산됨 이때, Additive attention과 dot-product attention 두가지의 방법이 존재
* Additive attention은 single hidden layer가 있는 FFN을 사용하여 compatibility function 계산
  > Q,K는 같은 dimension이 가질 필요가 없으며 dimension이 크기에 상관없이 좋은 성능을 보인다.
* dot-product attention에 scaling을 수행
  > matrix를 통해 최적화된 연산을 구현할 수 있기 때문에 훨씬 빠르고 공간 효율적 (hidden layer를 곱하는 과정이 추가되지 않아서 연산 속도와 space 측면에서 효율적)
  
  > q 와 k 의 dimension이 같아야 한다는 제약조건이 있으며, dimension이 클 때 학습에 방해 될 수 있음
 
<p align="center"><img width="218" alt="스크린샷 2023-01-17 오후 9 05 48" src="https://user-images.githubusercontent.com/56713634/213072382-e19d4535-74ed-4b03-b7b5-27030626db5d.png"><p> 
  
* Scaled Dot-Product은 Dot-Product에 scaling 수행한 것
* d_k가 값이 작은 경우에는 dot-product와 scaled dot-product가 유사하게 수행하지만 값이 커지면 scale이 더 우수함
* d_k 값이 클 때, dot-product의 size가 커지면서 softmax를 극도로 작은 gradient를 갖게 된다. 
* 이를 개선하기 위해 __1/√(dk)__ 만큼 스케일링



### Multi-Head Attention
single attention을 d_model 차원에 Q,K,V를 사용하여 수행하는 것보다 d_k, d_k, d_v 차원에 대해 학습된 서로 다른 linear projection을 사용하여 query, key, value를 h회 linear projection하는 것이 유익을 발견

* head란 attention을 수행하는 주체
* multi-head : 여러번 attention을 수행하겠다는 의미
> 동일한 소스에 대해 여러번 수행하여 어디에 얼만큼 집중할지 여러가지 집중 방법을 시도할 수 있음(다른 관점에서 문장을 바라볼 수 있음)
  
<p align="center"><img width="187" alt="스크린샷 2023-01-16 오후 10 59 43" src="https://user-images.githubusercontent.com/56713634/212965640-b703e9ba-d94c-44c2-b84c-c4b9af739274.png"><p> 
  
* query, key, value의 각 projection version에서 attention funtction을 병렬로 수행하여 d_v차원 output을 생성
* 이를 concat하여 다시 d_model 차원의 output이 생성

<p align="center"><img width="474" alt="스크린샷 2023-01-18 오전 2 10 33" src="https://user-images.githubusercontent.com/56713634/212965793-5f5486e9-eb53-455c-91f5-b19c2826f494.png"><p> 
  
* H = 8 parallel attention layer or head
* d_k = d_v = d_model/h = 64
* total computational cost는 single-head attention cost와 유사
  > 각 head마다 차원을 줄이므로

### Applications of Attention in our Model
Transformer는 다음의 방식으로 multi-head attnetion을 사용한다.

#### "Encoder-Decoder Attention layer" 
* decoder에서 self-attention 다음으로 사용되는 layer
* queries는 이전 decoder layer에서 가져오고, keys와 values는 encoder의 output에서 가져옴
* decoder의 모든 position의 vector들로 encoder의 모든 position 값들을 참조함으로써 decoder의 sequence vector들이 encoder의 sequence vector들과 어떠한 correlation을 가지는지를 학습
  
#### "self-attention in encoder"
* encoder에서 사용되는 self-attention으로 queries, keys, values 모두 encoder로부터 가져옴
* encoder의 각 position은 그 전 layer의 모든 positions들을 참조
* 이는 해당 position과 모든 position간의 correlation information을 더해주게 된다. 간단하게 설명해서 어떤 한 단어가 모든 단어들 중 어떤 단어들과 correlation이 높고, 또 어떤 단어와는 낮은지를 학습
* 이는 해당 position과 모든 position간의 correlation information을 더해주게 된다. 간단하게 설명해서 어떤 한 단어가 모든 단어들 중 어떤 단어들과 correlation이 높고, 또 어떤 단어와는 낮은지를 배우게 된다.  
  
#### "self-attention in decoder" 
* 전체적인 과정과 목표는 encoder의 self-attention과 동일
* sequence model의 auto-regressive property를 보존해야하기 때문에 masking vector를 사용하여 해당 position 이전의 벡터들만을 참조한다 _(이후에 나올 단어들을 참조하여 예측하는 것은 일종의 치팅)_

## Position-wise Feed-Forward Networks
인코더와 디코더의 각 layer는 fully connected feed-forward network를 가짐
* 각 position에 개별적으로 동일하게 적용

<p align="center"><img width="195" alt="스크린샷 2023-01-18 오전 2 11 22" src="https://user-images.githubusercontent.com/56713634/212965970-81248c46-e371-480c-8177-e4c848e8f3b3.png"><p> 

* input과 output의 차원은 512, inner-layer의 차원은 2048


## Embeddings and Softmax
* 학습된 embedding을 사용하여 input token과 output token을 d_model차원의 벡터로 변환
* 학습된 linear transformation과 softmax 함수를 사용하여 decoder output을 예측된 다음 token 확률로 변환
* 두 embedding layer와 softmax 이전 linear transformation 사이에서 동일한 weight matrix를 공유
* Inner layer에서 이러한 weight를 가중치에 √(d_model)을 곱한다.

## Positional Encoding
Transformer는 순차적인 특성이 없고 이에 따라 sequence의 위치 정보가 없기 때문에 positional 정보를 추가해야함
* 인코더와 디코더 stack 하단에 positional encodings을 추가
* Positional encoding은 embedding과 동일한 차원을 가짐

<p align="center"><img width="204" alt="스크린샷 2023-01-18 오전 2 12 13" src="https://user-images.githubusercontent.com/56713634/212966119-c4161208-c749-4df3-b903-bb9ffd000d94.png"><p> 
  
* pos : token의 위치, i : 차원
* Positional encoding의 각 차원은 sin파에 해당
* 본 논문에서 이 함수를 사용한 이유는 어떠한 고정된 offset k에 대해서 PE_pos+k를 PE_pos의 linear function으로 나타낼 수 있기 때문에 모델이 쉽게 상대적인 위치를 참조할 수 있을 것이라 가정했기 떄문

# Why Self-Attention
Self-attention layers와 recurrent and convolution layers와의 비교

<p align="center"><img width="455" alt="스크린샷 2023-01-18 오전 2 12 59" src="https://user-images.githubusercontent.com/56713634/212966280-19253faf-84c6-494b-9dfb-2b1d6cd5ab4e.png"><p> 
  
* layer별 총 연산의 complexity.
* sequential parallelize 할 수 있는 계산의 양
* Network에서 long-range dependencies(장거리 의존성) 사이 path length. 
  > long-range dependency의 학습은 번역 업무에서 핵심 task
  
  > long-range dependency을 잘 학습하기 위해서 중요한 것은 forward 및 backward signal의 길이다.
  
  > Input의 위치와 output의 위치의 길이가 짧을수록 dependency 학습이 쉬워짐
  
  > layer types로 구성된 네트워크에서 input과 output 위치 사이 길이가 maximum 길이를 비교

# Training
* WMT 2014 English-German dataset, WMT 2014 English-French dataset
* 8개의 NVIDIA P100 GPU로 학습
* base model은 12시간 동안 (100,000 step) 학습, big model 은 3.5일 동안 (300,000 step) 학습
* Adam optimizer
* Regularization
  > Residual Dropout
  
  > Label Smoothing

# Results
### Machine Translation
English-to-German translation task에 대해서 다른 모델들과 성능을 비교한 실험
  
<p align="center"><img width="476" alt="스크린샷 2023-01-18 오전 2 14 01" src="https://user-images.githubusercontent.com/56713634/212966543-a4ef2c87-da14-4f2c-9dfe-e6ffaa3f9685.png"><p> 
* BLEU는 기계 번역 결과와 사람이 직접 번역한 결과가 얼마나 유사한지 비교하여 번역에 대한 성능을 측정하는 방법
* Transformer가 다른 모델들에 비해서 높은 성능을 가지면서 training cost 또한 낮음
  
## Model Variations

모델의 여러 조건들을 변경해가면서 성능에 어떠한 영향을 주는지를 보는 실험  
  
<p align="center"><img width="483" alt="스크린샷 2023-01-18 오전 2 14 07" src="https://user-images.githubusercontent.com/56713634/212966651-ce6eb8cd-6e01-4d32-9fcb-d9413c99b458.png"><p> 

* (B) : key size를 너무 줄이면 quality가 안좋아짐
* (C) : 큰 모델이 성능이 좋음
* (D) : drop-out이 오버피팅을 방지
  
## English Constituency Parsing

Transformer가 다른 task에서도 잘 동작하는지를 보기 위해서 English Constituency Parsing task에도 적용  
>  Constituency Parsing : 어떠한 단어가 문법적으로 어떠한 것에 속하는지 분류하는 task

<p align="center"><img width="472" alt="스크린샷 2023-01-18 오전 2 14 14" src="https://user-images.githubusercontent.com/56713634/212966720-b41f9db7-eafb-469d-8242-3061e0fb423b.png"><p> 
  
* transformer를 해당 task에 맞게 tuning하지 않았음에도 불구하고 좋은 성능
