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
  > 이 모델은 8개의 P100 GPU로 12시간 호학습하여 병렬처리와 SOTA를 달성한다.


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
attention function은 query와 key-value쌍을 query, keys, values, output이 모두 vectors인 output에 mapping하는 것

#### 동작 방식
   
<p align="center"><img width="357" alt="스크린샷 2023-01-18 오전 12 49 08" src="https://user-images.githubusercontent.com/56713634/212945605-95250bb6-0fa6-4bd4-9916-831d7775c175.png"><p> 
   
* 빨간색 vector : encoder의 매 time step마다의 hidden state
* 초록색 vector : 마지막 hidden state vector는 decoder의 h0이 되어 x와 곱셈을 하여 나온 값
* encoder의 모든 hidden state vector를 고려하기 위해 초록색 vector와 빨간색 vectore들의 내적을 각각 구한다 
* 내적으로 부터 얻은 유사도에 softmax를 취하여 각 hidden state를 얼마나 반영할 것인지를 의미하는 weight를 구한다. 이러한 weight들을 Attention vector 
* Attention모듈의 output은 encoder hidden state들의 가중 평균 vector가 됨 -> __context vector__

<p align="center"><img width="442" alt="스크린샷 2023-01-18 오전 12 49 14" src="https://user-images.githubusercontent.com/56713634/212950836-d2e2a389-403e-4d6c-a3a3-8b2dd44c331a.png"><p>
  
* Attention모듈의 input은 초록색 vector와 빨간색 vector, output은 가장 상단의 vector
* decoder hidden state vector와 Attention output vector는 h1이 되어 다음 단계의 hidden state로 전달
* 새로운 x와 곱셈을 하여 하단의 두번쨰 초록색 vector
* 이 vector로 encoder의 빨간색 hidden state vector들과 내적을 통해 유사도를 계산한 후 가중 평균 벡터를 구한 후, 이 가중 평균 벡터와 concatenate하여 output layer의 input으로 들어가 output을 예측
  
<p align="center"><img width="510" alt="스크린샷 2023-01-18 오전 12 49 19" src="https://user-images.githubusercontent.com/56713634/212951794-82014417-1000-4d86-bd13-217fc7d9258d.png"><p>
  
### Scaled Dot-Product Attention

<p align="center"><img width="176" alt="스크린샷 2023-01-16 오후 10 59 37" src="https://user-images.githubusercontent.com/56713634/212894466-ae57f911-bad8-4b34-92b1-bfcac00d5106.png"><p> 

* input : Query(Q), Key(K), Value(V)

#### Attention Function
weight는 Q와 K의 조합으로 계산됨 이때, Additive attention과 dot-product attention 두가지의 방법이 존재
* Additive attention은 single hidden layer가 있는 FFN을 사용하여 compatibility function 계산
  > Q,K는 같은 dimension이 가질 필요가 없으며 dimension이 크기에 상관없이 좋은 성능을 보인다.
* dot-product attention에 scaling을 수행
  > matrix를 통해 최적화된 연산을 구현할 수 있기 때문에 훨씬 빠르고 공간 효율적 (hidden layer를 곱하는 과정이 추가되지 않아서 연산 속도와 space 측면에서 효율적)
  
  > q 와 k 의 dimension이 같아야 한다는 제약조건이 있으며, dimension이 클 때 학습에 방해 될 수 있음
 
<p align="center"><img width="176" alt="스크린샷 2023-01-16 오후 10 59 37" src="https://user-images.githubusercontent.com/56713634/212894466-ae57f911-bad8-4b34-92b1-bfcac00d5106.png"><p> 
  
* Scaled Dot-Product은 Dot-Produc에 scaleling 수행한 것
* d_k가 값이 작은 경우에는 dot-product와 scaled dot-product가 유사하게 수행하지만 값이 커지면 scale이 더 우수함
* d_k 값이 클 때, dot-product의 size가 커지면서 softmax를 극도로 작은 gradient를 갖게 된다. 
* 이를 개선하기 위해 __1/√(dk)__ 만큼 스케일링



### Multi-Head Attention
single attention을 d_model 차원에 Q,K,V를 사용하여 수행하는 것보다 d_k, d_k, d_v 차원에 대해 학습된 서로 다른 linear projection을 사용하여 query, key, value를 h회 linear projection하는 것이 유익을 발견
  
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

* "Encoder-Decoder Attention layer" 에서 query는 이전 decoder layer에서 얻고, key와 value는 encoder의 output에서 얻는다. 이를 통해 decoder의 모든 position이 input sequence의 모든 position에 배치될 수 있다. 이는 sequence-to-sequence 모델에서 일반적인 encoder-decoder attention 메커니즘과 동일하다.
* Self-Attention layer는 encoder에 존재하며 query, key, value가 동일하며, 이는 encoder에 있는 이전 layer의 output이다. Encoder의 각 position은 이전 layer의 모든 position에 attend 할 수 있다.
* Self-Attention layer는 decoder에도 존재하며, 마찬가지로, decoder의 self-attention layer는 각 position이 해당 position의 위치까지 docoder의 모든 position에 attned하도록 한다. auto-regressive propert를 유지하기 위해 decoder에서 leftward information flow을 막아야 한다 (미래 시점의 단어를 볼 수 없도록 하는 것). 이를 위해 매우 작은 수를 부여하여 softmax 결과 0에 수렴하도록 하여 masking을 수행한다.우리는 잘못된 연결에 해당하는 소프트맥스 입력의 모든 값을 마스킹(-)로 설정)하여 스케일링된 도트 제품 주의의 내부에서 이를 구현한다.


### Position-wise Feed-Forward Networks
인코더와 디코더의 각 layer는 fully connected feed-forward network를 가짐
* 각 position에 개별적으로 동일하게 적용
* 중간에 ReLU activation이 있는 두 가지 linear transformation 구성

<p align="center"><img width="195" alt="스크린샷 2023-01-18 오전 2 11 22" src="https://user-images.githubusercontent.com/56713634/212965970-81248c46-e371-480c-8177-e4c848e8f3b3.png"><p> 
  
* linear transformation은 다른 position에 대해 동일하지만 layer간 parameter는 다름
* input과 output의 차원은 512, inner-layer의 차원은 2048


### Embeddings and Softmax
* 학습된 embedding을 사용하여 input token과 output token을 d_model차원의 벡터로 변환
* 학습된 linear transformation과 softmax 함수를 사용하여 decoder output을 예측된 다음 token 확률로 변환
* 두 embedding layer와 softmax 이전 linear transformation 사이에서 동일한 weight matrix를 공유
* Inner layer에서 이러한 weight를 가중치에 √(d_model)을 곱한다.

### Positional Encoding
Transformer는 순차적인 특성이 없고 이에 따라 sequence의 위치 정보가 없기 때문에 positional 정보를 추가해야함
* 인코더와 디코더 stack 하단에 positional encodings을 추가
* Positional encoding은 embedding과 동일한 차원을 가짐

<p align="center"><img width="204" alt="스크린샷 2023-01-18 오전 2 12 13" src="https://user-images.githubusercontent.com/56713634/212966119-c4161208-c749-4df3-b903-bb9ffd000d94.png"><p> 
  
* pos : token의 위치, i : 차원
* Positional encoding의 각 차원은 sin파에 해당
* 본 논문에서 이 함수를 사용한 이유는 어떠한 고정된 offset k에 대해서 PE_pos+k를 PE_pos의 linear function으로 나타낼 수 있기 때문에 모델이 쉽게 상대적인 위치를 참조할 수 있을 것이라 가정했기 떄문

## Why Self-Attention
Self-attention layers와 recurrent and convolution layers와의 비교

<p align="center"><img width="455" alt="스크린샷 2023-01-18 오전 2 12 59" src="https://user-images.githubusercontent.com/56713634/212966280-19253faf-84c6-494b-9dfb-2b1d6cd5ab4e.png"><p> 
  
* layer별 총 연산의 complexity.
* sequential parallelize 할 수 있는 계산의 양
* Network에서 long-range dependencies(장거리 의존성) 사이 path length. 
  > long-range dependency의 학습은 번역 업무에서 핵심 task
  
  > long-range dependency을 잘 학습하기 위해서 중요한 것은 forward 및 backward signal의 길이다.
  
  >  Input의 위치와 output의 위치의 길이가 짧을수록 dependency 학습이 쉬워짐
  
  >  layer types로 구성된 네트워크에서 input과 output 위치 사이 길이가 maximum 길이를 비교

## Training
* WMT 2014 English-German dataset, WMT 2014 English-French dataset
* 8개의 NVIDIA P100 GPU로 학습
* base model은 12시간 동안 (100,000 step) 학습, big model 은 3.5일 동안 (300,000 step) 학습
* Adam optimizer
* Regularization
  > Residual Dropout
  
  > Label Smoothing

## Results
### Machine Translation

<p align="center"><img width="476" alt="스크린샷 2023-01-18 오전 2 14 01" src="https://user-images.githubusercontent.com/56713634/212966543-a4ef2c87-da14-4f2c-9dfe-e6ffaa3f9685.png"><p> 

## Model Variations

<p align="center"><img width="483" alt="스크린샷 2023-01-18 오전 2 14 07" src="https://user-images.githubusercontent.com/56713634/212966651-ce6eb8cd-6e01-4d32-9fcb-d9413c99b458.png"><p> 
  
## English Constituency Parsing

<p align="center"><img width="472" alt="스크린샷 2023-01-18 오전 2 14 14" src="https://user-images.githubusercontent.com/56713634/212966720-b41f9db7-eafb-469d-8242-3061e0fb423b.png"><p> 

