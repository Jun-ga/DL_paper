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
[사진 첨부]

## Encoder and Decoder Stacks
### Encoder
[사진 첨부]
* N = 6개의 layer의 stack으로 구성되어있음 이때, 각 layer는 2개의 sub-layer가 있다.
* 첫번째 layer은 multi-head self-attention mechanism, 두번째 layer은 position-wise fully connected feed-forward network
* 이 2개의 sub-lalyer에 각각 residual connection과 normalization을 적용  
  >  residual connection을 구현하기 위하여 모든 sub-layer(embedding layers 포함)의 output은 512
 
### Decoder
[사진 첨부]
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
[사진 첨부]

* input : Query(Q), Key(K), Value(V)
[식 첨부]
*


### Multi-Head Attention
[사진 첨부]
