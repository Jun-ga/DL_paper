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


### 수식

<p align="center"><img width="342" alt="스크린샷 2022-05-24 오후 12 12 35" src="https://user-images.githubusercontent.com/56713634/169941333-9e6b0bfb-537b-4410-82b4-38df77ab0d61.png"></p>

<p align="center"><img width="401" alt="스크린샷 2022-05-24 오전 3 12 12" src="https://user-images.githubusercontent.com/56713634/169881266-71d80474-c597-48da-a44f-96b2c361d8b2.png"></p>




## LSTM(Long Short-Term Memory,장단기 메모리)
RNN의 문제를 해결하기 위해 고안된 방

<p align="center"><img width="343" alt="스크린샷 2022-05-24 오전 3 18 10" src="https://user-images.githubusercontent.com/56713634/169882171-621700b7-4587-43d3-bb99-61365836acf4.png"></p>

### 특징
* long sequence의 입력을 처리가히게 탁월한 성능을 가짐
* long term state를 위해 Memory cell을 추가
<p align="center"><img width="471" alt="스크린샷 2022-05-24 오후 12 34 27" src="https://user-images.githubusercontent.com/56713634/169943540-296d4caf-3ea6-4b79-8b53-e8103c713919.png"></p>
* hidden state와 Memory cell을 구하기 위해 3개의 gate가 추가됨
  > 이를 통해, 불필요한 기억은 지우고 기억해야할 것들을 정함
  > h_t는 단기상태, c_t는 장기상태

#### input gate
