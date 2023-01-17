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
transformer은 self-attention과 point-wise를 따르며, 크게 Encoder와 Decoder로 구성되어있다.
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
> 이때, output은 values의 weighted sum으로 계산
> 즉, 

[필요시 추가 설명 첨부]

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

[사진 첨부]
* query, key, value의 각 projection version에서 attention funtction을 병렬로 수행하여 d_v차원 output을 생성
* 이를 concat하여 다시 d_model 차원의 output이 생성

[식 첨부]
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

[식 첨부]
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
[식 첨부]
* pos : token의 위치, i : 차원
* Positional encoding의 각 차원은 sin파에 해당
* 본 논문에서 이 함수를 사용한 이유는 어떠한 고정된 offset k에 대해서 PE_pos+k를 PE_pos의 linear function으로 나타낼 수 있기 때문에 모델이 쉽게 상대적인 위치를 참조할 수 있을 것이라 가정했기 떄문

## Why Self-Attention
Self-attention layers와 recurrent and convolution layers와의 비교
[그림 첨부]
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
[표 첨부]

## Model Variations
[표 첨부]

## English Constituency Parsing
[표 첨부]

