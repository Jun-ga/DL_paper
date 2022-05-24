# RNN,LSTM,GRU 개념
## 한눈에 보기 
<p align="center"><img width="587" alt="스크린샷 2022-05-24 오전 12 25 54" src="https://user-images.githubusercontent.com/56713634/169880020-d64c83c4-73be-4994-9ed8-9ea697cc0999.png">


## RNN(Recurrent Neural Network,순환 신경망)
입력과 출력을 sequence 단위로 처리하는 모델
시계열 데이터를 처리하기 위한 모델

<p align="center"><img width="368" alt="스크린샷 2022-05-24 오전 12 21 44" src="https://user-images.githubusercontent.com/56713634/169866879-37067c15-6366-42ea-bb43-d08fd3d3fe5b.png"></p>

### 특징
* hidden layer의 메모리셀은 각각의 시점에서 바로 이전의 시점에서의 hidden layer에서 나온 값을 자신의 입력으로 사용 > __재귀적__

<p align="center"><img width="565" alt="스크린샷 2022-05-24 오전 3 10 04" src="https://user-images.githubusercontent.com/56713634/169880995-1d599e26-c046-49f7-aa62-931a130b3225.png"></p>


* 입력 갯수와 출력 갯수에 따라 one to many, many to one, many to many로 나뉨
* 순서대로 처리가 되므로 속도가 느리다.
* 입력데이터의 길이가 길어지게 된다면 gradint가 매우 작아져 전달이 안됨 = Gradient Vanishing
  > 이를 해결한 방법이 LSTM


### 수식

<p align="center"><img width="342" alt="스크린샷 2022-05-24 오후 12 12 35" src="https://user-images.githubusercontent.com/56713634/169941333-9e6b0bfb-537b-4410-82b4-38df77ab0d61.png"></p>

<p align="center"><img width="401" alt="스크린샷 2022-05-24 오전 3 12 12" src="https://user-images.githubusercontent.com/56713634/169881266-71d80474-c597-48da-a44f-96b2c361d8b2.png"></p>




## LSTM(Long Short-Term Memory,장단기 메모리)
RNN의 문제를 해결하기 위해 고안된 방식

<p align="center"><img width="343" alt="스크린샷 2022-05-24 오전 3 18 10" src="https://user-images.githubusercontent.com/56713634/169882171-621700b7-4587-43d3-bb99-61365836acf4.png"></p>

### 특징
* long sequence의 입력을 처리가히게 탁월한 성능을 가짐
* long term state를 위해 Memory cell을 추가
<p align="center"><img width="471" alt="스크린샷 2022-05-24 오후 12 34 27" src="https://user-images.githubusercontent.com/56713634/169943540-296d4caf-3ea6-4b79-8b53-e8103c713919.png"></p>
* hidden state와 Memory cell을 구하기 위해 3개의 gate가 추가됨
  > 이를 통해, 불필요한 기억은 지우고 기억해야할 것들을 정함
  > h_t는 단기상태, c_t는 장기상태

#### input gate
현재 정보를 기억하기 위한 게이트

<p align="center"><img width="276" alt="스크린샷 2022-05-24 오후 12 37 49" src="https://user-images.githubusercontent.com/56713634/169943982-841efb56-a3cb-4592-8f36-b5e32348b01c.png"></p>

* 시그모이드 함수를 지나 0 과 1 사이의 값과 thanh(하이퍼볼릭탄젠트함수)를 지나 -1 과 1 사이의 값 두 개가 나온다.
* 이 값을 통해 선택된 기억할 정보의 양을 정함

#### foget gate
기억을 삭제하기 위한 게이트

<p align="center"><img width="269" alt="스크린샷 2022-05-24 오후 12 41 27" src="https://user-images.githubusercontent.com/56713634/169944865-6f126138-7057-4c6b-b7c2-920b4c335863.png"></p>

* 현재 시점 t의 x값과 이전 시점 t-1의 hidden state가 시그모이드 함수를 거침
* 0과 1 사이의 값이 나오게 되며, 이 값은 삭제 과정을 거친 정보의 양 
* 0에 가까울수록 정보가 많이 삭제, 1에 가까울수록 정보를 온전히 기억
  > 이를 가지고 cell state를 구함


#### cell stata
장기 상태라고 부름
<p align="center"><img width="293" alt="스크린샷 2022-05-24 오후 12 52 15" src="https://user-images.githubusercontent.com/56713634/169945920-585b0c47-6dcd-4db3-ac49-12e2047e51fb.png"></p>

* forget gate의 출력값이 0이라면, 오직 input gate의 결과만 현재시점의 cell 값을 결정
* input gate가 0 이라면 현재시점의 cell값은 이전 시점의 cell값에만 의존
* __forget gate는 이전 시점의 입력을 얼마나 반영할지, input gate는 현재 시점의 입력을 얼마나 반영할지 결정__
  > 정보를 선택적으로 활용 <\br>
  > cell state는 각 gate의 결과를 더함으로 시퀀스가 길더라고 gradient를 잘 전파함

#### output state
Memory cell을 얼마나 사용할지 결정하기 위한 게이트

<p align="center"><img width="272" alt="스크린샷 2022-05-24 오후 1 11 59" src="https://user-images.githubusercontent.com/56713634/169947355-44fba368-496e-48cd-9cca-edfbbdb535a9.png"></p>

* 현재 시점 t의 x갑과 이전 시점 t-1의 hidden sta
* te가 시그모이드 함수를 지난 값은 현재 시점 t의 은닉 상태를 결정
* 은닉상태 = 단기 상태, 장기상태의 값이 tanh함수를 지나 -1과 1사이의 값
* 즉, 현 시점의 hidden state는 현 시점의 cell state와 함께 계산되며 출력됨과 동시에 다음 hidden state로 넘김


## GRU(Gated Reccurent Unit)
LSTM의 구조를 조금 더 간단하게 개선한 모델

<p align="center"><img width="564" alt="스크린샷 2022-05-24 오후 1 30 44" src="https://user-images.githubusercontent.com/56713634/169949300-51faa60c-277a-4842-ac7b-cff4645df703.png"></p>

### 특징
* LSTM보다 간단한 구조면서 긴 데이터를 잘 처리함
* Memory cell 사용 x
  > cell stata와 hidden state가 합쳐져 하나의 hidden state로 표현
* reset gate, update gate 총 2개의 gate만을 사용

#### reset gata
지난 정보를 얼마나 버릴지 결정하기 위한 게이트
* 이전 시점의 hidden state와 현 시점의 x를 시그모이드 함수에 적용하여 구하는 방식 식(2)
  > 결과값은 0과 1 사이 값 =  이전 hidden state의 값을 얼마나 활용할지에 대한 정보
* 식(3)을 통해 현재 시점에 과거의 정보를 얼마나 사용할지 정함
  > 전 시점의 hidden state에 reset gate를 곱함

#### updata gate
정보를 얼마나 반영할지 결정하는 게이트
* 식(1)에서 구한 결과 z는 현재 정보를 얼마나 사용할지 반영
* (1-z)는 과거 정보에 대해 얼마나 사용할지 반영
  > 이는 LSTM의 input, forget gate로 볼 수 있음
* 식(4)를 통해 현 시점의 hidden state를 구할 수 있음


