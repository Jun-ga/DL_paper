# *Deep contextualized word representations*
# Abstract
* EMLo(Deep contextualized word representations) = 2018년도에 발표된 단어 임베딩
* 큰 텍스트 말충치에서 사전 훈련되는 biLM을 이용해 훈련을 진행했다.
* 본 모델은 2가지의 특성을 가지고 있다.
  > 구문 및 의미론적에서 단어의 복잡한 특징을 표현할 수 있다.

  > 다의어경우 언어적 맥락에 따라 표현할 수 있다.
 
 # Introduction
* 사전 훈련된 단어 표현(Mikolov et al., Pennington et al.)은 많은 신경 언어 이해 모델의 핵심 구성요소이며, 높은 수준의 단어 임베딩이 중요하다.
* 해당 방식은 입력으로 제공된 전체문장에 대해 문장을 구성하는 단어들의 임베딩을 제공한다는 점에서 전통적인 단어유형 임베딩과 다르다.
  > 즉 단어 임베딩 전에 전체 문장을 고려하여 임베딩하는 방식
* 양방향 LSTM인 biLM 으로부터 얻은 백터를 사용한다.
* EMLo는 biLM의 모든 layer를 사용하며, 기존 모델에 쉽게 추가될 수 있다는 장점을 가진다.

# ELMo: Embeddings from Language Models

### 엘모의 주요특징
* character convolutions 에서 얻은 biLM의 가장 위 2개 layer의 선형함수(가중함)에 의해 계산
* 대규모로 biLM을 사전학습시켰을때 semisupervised learning 가능
* 다앙햔 NLP에 쉽게 사용될 수 있음.

### Bidirectional language models (biLM)
* forward 모델과 backward 모델의 합으로 나타낸다.
* input 문장은 ![](https://user-images.githubusercontent.com/56713634/148430286-73ac3b4f-40bf-4656-bffa-910ab143071f.png)
개의 token ![](https://user-images.githubusercontent.com/56713634/148430356-38534186-4c1b-4d2c-a52b-fccf8d85efe6.png) 이라고 가정


<p align="center"><img src="https://user-images.githubusercontent.com/56713634/148729939-796cdefd-1cf7-42e6-8ca8-be631c016614.png" alt="text" width="number" /></p>

1. forward LM
    ![CodeCogsEqn](https://user-images.githubusercontent.com/56713634/148430663-0ce6a99b-5a49-4a18-a403-2e4cc1c74257.png)일때 ![CodeCogsEqn (1)](https://user-images.githubusercontent.com/56713634/148430732-5c81fd85-c787-4ffe-ae05-3d5c8a72d9bd.png)가 나올 확률을 예측하는 모델 
    >즉, 이전 단어를 이용하여 미래를 예측하도록 학습 
    
    <p align="center"><img src="https://user-images.githubusercontent.com/56713634/148431045-e2417230-d084-4e7b-805b-65219df439ee.png" alt="text" width="number" /></p>
    
    * ![CodeCogsEqn (3)](https://user-images.githubusercontent.com/56713634/148432198-52ad1710-2e5a-4f80-84ca-fdc9d786b740.png) (cnn이나 token embedding 얻은)를 L개의 layer를 가진 forward LSTM의 input으로 투입한다.

    * 각 ![CodeCogsEqn (4)](https://user-images.githubusercontent.com/56713634/148432429-d38ce1c8-aded-40bc-a589-217ee12fa4e5.png)에 대해서 context-independent ![CodeCogsEqn (5)](https://user-images.githubusercontent.com/56713634/148432512-c51875f1-5d29-482d-bb8f-b698fe6b54a1.png)(![CodeCogsEqn (6)](https://user-images.githubusercontent.com/56713634/148432578-0bc5011f-4a9d-4c23-8ec0-65c66b694640.png)) 가진다.

   * top layer LSTM의 output  ![CodeCogsEqn (7)](https://user-images.githubusercontent.com/56713634/148432665-ff213b71-c325-4335-a85d-52df3f08db79.png)가 ![CodeCogsEqn (8)](https://user-images.githubusercontent.com/56713634/148432728-41cb3c47-9fd8-49a3-b5ec-df9c920a64f6.png)을 예측한다.
    
2. backward LM
    ![CodeCogsEqn (9)](https://user-images.githubusercontent.com/56713634/148434201-70f13e95-18d8-4454-821a-149893f9cf56.png)일때![CodeCogsEqn (10)](https://user-images.githubusercontent.com/56713634/148434298-2a899386-8516-4f6b-b6a5-8da517b9843a.png)가 나올 확률을 예측하는 모델 
    >즉, 미래 단어를 이용하여 이전을 예측하도록 학습
    
    <p align="center"><img src="https://user-images.githubusercontent.com/56713634/148434445-57e160a6-702b-4aed-acab-5db8fa3acea5.png" alt="text" width="number" /></p>

    * 각 ![CodeCogsEqn (4)](https://user-images.githubusercontent.com/56713634/148432429-d38ce1c8-aded-40bc-a589-217ee12fa4e5.png)에 대해서 context-independent ![CodeCogsEqn (13)](https://user-images.githubusercontent.com/56713634/148434688-85005dda-f8aa-44c5-afde-3b236a4b480a.png)(![CodeCogsEqn (6)](https://user-images.githubusercontent.com/56713634/148432578-0bc5011f-4a9d-4c23-8ec0-65c66b694640.png)) 가진다.
    * top layer LSTM의 output ![CodeCogsEqn (14)](https://user-images.githubusercontent.com/56713634/148434767-0b1e1f70-f276-4337-b75f-05371a7eadab.png)가 ![CodeCogsEqn (15)](https://user-images.githubusercontent.com/56713634/148434817-4cabdc23-93cc-490c-9762-a1bdfb0cb63e.png)을 예측한다.
    
3. biLM
   위의 두 LSTM을 결합하여 두 방향에 대한 log likelihood를 최대화 한 모델
   
<p align="center"><img src="https://user-images.githubusercontent.com/56713634/148434978-9706bf2d-2fec-4b68-b7e1-064a05eb006f.png" alt="text" width="number" /></p>
   
<p align="center"><img src="https://latex.codecogs.com/svg.image?\Theta_{x}" title="\Theta_{x}" /> : token representation, <img src="https://latex.codecogs.com/svg.image?\Theta_{s}" title="\Theta_{s}" /> : softmax layer</p>


### ELMo
* biLM에서 중간 매체 레이어들의 표현을 특별한 방식으로 합친 모델
* L개의 layer일때 총 2L + 1개의 representation 사용
  > input representation 1개  각 층의 forward, backward representation 2L개
  
<p align="center"><img src="https://latex.codecogs.com/svg.image?R_{k}" title="R_{k}" /> = {<img src="https://latex.codecogs.com/svg.image?x_{k}^{LM}" title="x_{k}^{LM}" /><img src="https://latex.codecogs.com/svg.image?,\overrightarrow{h_{kj}^{LM}},\overleftarrow{h_{kj}^{LM}}" title=",\overrightarrow{h_{kj}^{LM}},\overleftarrow{h_{kj}^{LM}}" />|<img src="https://latex.codecogs.com/svg.image?j&space;=&space;1,...,L" title="j = 1,...,L" />}</p>

<p align="center"> = { <img src="https://latex.codecogs.com/svg.image?h_{kj}^{LM}" title="h_{kj}^{LM}" />|<img src="https://latex.codecogs.com/svg.image?j&space;=&space;0,...,L" title="j = 0,...,L" />}</p>

* 최종적으로 모든 층에서 생성된 representation을 결합하여 하나의 벡터로 생성
<p align="center"><img src="https://latex.codecogs.com/svg.image?ELMo_{k}^{task}&space;=&space;E(R_{k};\Theta^{task})=\gamma^{task}\sum_{j=0}^{L}{s_{j}^{task}h_{k,j}^{LM}}" title="ELMo_{k}^{task} = E(R_{k};\Theta^{task})=\gamma^{task}\sum_{j=0}^{L}{s_{j}^{task}h_{k,j}^{LM}}" /></p>

<p align="center"><img src="https://latex.codecogs.com/svg.image?{s_{j}^{task}" title="{s_{j}^{task}" /> : softmax-normalized weights, <img src="https://latex.codecogs.com/svg.image?\gamma^{task}" title="\gamma^{task}" /> : scalar parameter</p>

**ELMo 생성과정**
![IMG_KEEP_1641779843](https://user-images.githubusercontent.com/56713634/148719627-e8c2af1e-0cd3-4869-9549-af3093b61263.jpg)


# Evalution
![캡처](https://user-images.githubusercontent.com/56713634/148714350-79e3008e-8991-40af-9df4-1b049ae7dda0.PNG)
* ELMo를 단순히 추가하는 것으로도 baseline model에 비해 성능이 좋아짐

# Analysis
### Alternate layer weight schemes

Regularization parameter인 <img src="https://latex.codecogs.com/svg.image?\lambda&space;" title="\lambda " />의 값이 매우 커지면 weight funciton이 단순히 평균을 내는 역할을 수행, 값이 작아지면 각 층에 대한 weight값이 다양하게 적용된다.

<p align="center"><img src="https://user-images.githubusercontent.com/56713634/148719642-1c0eef8e-77d7-4872-baf6-b1d7b7482bb2.PNG" alt="text" width="number" /></p>

* 모든 층에 represention를 사용했을때 성능이 더 좋다.
* <img src="https://latex.codecogs.com/svg.image?\lambda&space;" title="\lambda " />값을 작게하여 각 층의 weight을 학습하도록 하는 것이 더 효과적이다.

### Where to include ELMo?
본 논문에서 사용한 구조에서는 가장 낮은 층의 input에만 ELMo를 사용했다. 하지만 특정 task에서는 output에도 ELMo를 추가하는 것이 성능향상에 도움이 된다.
<p align="center"><img src="https://user-images.githubusercontent.com/56713634/148719647-c82a1fda-b095-4baa-a9d1-7d578632b6b6.PNG" alt="text" width="number" /></p>

* SNLI & SQuAD의 경우 biRNN 이후에 Attention을 사용하므로 이 layer에 ELMo를 도입하는 것이 성능 향상에 도움이 된다.

### What information is captured by the biLM's represntaions?

<p align="center"><img src="https://user-images.githubusercontent.com/56713634/148721433-77a654bb-72ef-4e1b-8172-b3d862e23639.PNG" alt="text" width="number" /></p>

* GloVe에서는 play에 관련된 단어로 스포츠와 유사한 의미를 갖는 것들이 나온다.
* ELMo는 주어진 문장에서 사용된 play와 유사한 의미를 갖는 문장들이 나온다. 

### sample efficiency
ELMo를 추가시 학습 속도 향상을 보여준다.
<p align="center"><img src="https://user-images.githubusercontent.com/56713634/148721754-617222d6-5353-42f2-b7bb-89f34699d217.PNG" alt="text" width="number" /></p>

### visualization of learned weight
softmax-normalized parameter를 시각화한 것
<p align="center"><img src="https://user-images.githubusercontent.com/56713634/148721759-7757bac8-ca97-42bd-8db9-49c2707114bb.PNG" alt="text" width="number" /></p>

* ELMo가 input에 사용된 경우 첫번째 LSTM의 선호도가 높다
* ELMo가 output에 사용된 경우 weight는 균형있게 분배되었지만 낮은 layer의 선호도가 높다

# Conclusion
biLM을 사용하여 높은 성능의 deep context dependent 단어 표현을 학습하는 ELMo모델을 제안했다. ELMo를 사용하면 대부분의 NLP task에서 성능이 향상되었다. 또한 다양한 실험을 통해 biLM 계층이 문맥 내 단어들에 대한 구문 및 의미정보를 인코딩하여 다의어 표현등을 극복할 수 있었다.
