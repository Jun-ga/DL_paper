# RNN,LSTM,GRU 개념
## 한눈에 보기

## RNN(Recurrent Neural Network,순환 신경망)
입력과 출력을 sequence 단위로 처리하는 모델

<img width="368" alt="스크린샷 2022-05-24 오전 12 21 44" src="https://user-images.githubusercontent.com/56713634/169866879-37067c15-6366-42ea-bb43-d08fd3d3fe5b.png">

* hidden layer의 메모리셀은 각각의 시점에서 바로 이전의 시잠에서의 hidden layer에서 나온 값을 자신의 입력으로 사용 _재귀적_
* 입력 갯수와 출력 갯수에 따라 one to many, many to one, many to many로 나뉨

