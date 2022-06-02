# YOU ONLY LOOK ONCE

# YOLOv1

<p align="center"><img width="595" alt="스크린샷 2022-06-02 오후 1 53 21" src="https://user-images.githubusercontent.com/56713634/171555461-b965738f-84be-4c5e-9d0c-0b63a868802c.png"></p>

* 1 stage detector
* single convolutional network로 이미지를 입력받아, 바운딩박스와 각 박스의 class 예측
* image grid를 448 x 448로 resize한 후 CNN 동작

## 특징

__빠른 속도__
* 회귀문제로 진행하기 때문에 복잡하지않고 간단한 신경망 학습을 통해 예측한다. _45fps_

__Fast R-CNN 보다 background error가 두 배이상 적음__
<p align="center"><img width="380" alt="스크린샷 2022-06-02 오후 2 11 22" src="https://user-images.githubusercontent.com/56713634/171557283-36e4be26-7120-4fe3-99bd-52835ec7cb39.png"></p>

* YOLO는 예측할때 이미지 전체를 이용하므로 class와 객체 출현에 대한 cotextual imformation 이용가능
* 반면, fast R-CNN은 제안한 영역만을 이용하여 예측하기 때문에 larger context를 이용하지못함 -> 배경을 객체로 탐지하는 경우 발생

__일반화__
* 객체의 일반화 된 representations를 학습하여 다른 도메인에서 좋은 성능을 지님
* 새로운 도메인 적용에 용이


## 동작
<p align="center"><img width="398" alt="스크린샷 2022-06-02 오후 2 14 57" src="https://user-images.githubusercontent.com/56713634/171557661-3d02ccaa-cedb-4ac5-a8c5-d1af6808a81e.png"></p>

* 입력이미지를 5x5 grid로 분할 
* 각 grid cell은 B개의 bounding box와 bounding box의 confidence score를 가짐 _객체 중심이 grid cell에 맞아 떨어지면 그 grid cell 객체를 탐지했다 판단_
* box는 총 5가지 예측으로 구성
  * x,y : 사각형 중앙의 좌표
  * w : 가로의 길이 = 넓이 h : 세로의 길이 = 높이
  * confidence : 신뢰도(예측된 box와 실제 box의 IOU) 즉, 물체가 있는지 있다면 바운딩박스에 얼마나 포함되어 있는지에 대한 신뢰도

  <p align="center"><img width="172" alt="스크린샷 2022-06-02 오후 2 38 02" src="https://user-images.githubusercontent.com/56713634/171560527-72a7f9e0-7f6b-4dec-a0a5-5ac943f747b5.png"></p>
  
       object(x) : confidence score = 0  object(o) : confidence score = IOU

* C : conditional class probabilities
  * 각각의 grid cell은 c개의 conditional class probability를 가짐 (각 class에 대한 probability)
  * 각 박스에 대한 class-specific confidence scores를 얻을 수 있음
    > 해당 바운딩 박스에서 특정 class의 object가 나타날 확률과 object에 맞게 바운딩 박스를 올바르게 예측했는지에 대해 나타냄

<p align="center"><img width="645" alt="스크린샷 2022-06-02 오후 2 38 27" src="https://user-images.githubusercontent.com/56713634/171560458-f325eea0-c488-4873-b263-fee927d12300.png"></p>



### Network Design
<p align="center"><img width="647" alt="스크린샷 2022-06-02 오후 2 53 52" src="https://user-images.githubusercontent.com/56713634/171562277-5a712a4b-f4a7-42b3-a856-7d59cb4c479c.png"></p>

* GoogLeNet에서 영향을 받아 1x1 layer뒤에 3x3 conv layer 이용
* 24개의 conv layers, 2개의 fully connected layers
* convolutional layer에서 이미지의 특징을 추출하여 FC layer에세 추출된 특징을 기반으로 class 확률과 bounding box의 좌표를 추론

### Loss function

* classification loss : class conditional probabilities의 squared error
* localization loss : 예측된 boundary box의 위치와 크기에 대한 error
* confidence loss : 객체 탐지 여부에 따라 가중치를 다르게 줌

<p align="center"><img width="466" alt="스크린샷 2022-06-02 오후 3 16 34" src="https://user-images.githubusercontent.com/56713634/171565294-483ad6e7-2d15-43df-9f31-73bdee57ee50.png"></p>


1. object가 존재하는 grid cell i의 예측 bounding box j에 대해, x와 y의 loss를 계산.
2. object가 존재하는 grid cell i의 에측 bounding box j에 대해, w와 h의 loss를 계산. 
큰 box에 대해서는 small deviation을 반영하기 위해 제곱근을 취한 후, sum-squared error를 한다.(같은 error라도 larger box의 경우 상대적으로 IOU에 영향을 적게 준다.)
3. object가 존재하는 grid cell i의 예측 bounding box j에 대해, confidence score의 loss를 계산. (Ci = 1)
4. object가 존재하지 않는 grid cell i의 bounding box j에 대해, confidence score의 loss를 계산. (Ci = 0)
5. object가 존재하는 grid cell i에 대해, conditional class probability의 loss 계산. (Correct class c: pi(c)=1, otherwise: pi(c)=0)

## 한계

* 각 grid cell 마다 B개의 bounding box만 추측해내야하므로 objet가 겹쳐있으면 제대로 예측 불가능
* data로 부터 bounding box를 훈련시키기 때문에,학습데이터에는 없는 새로운 형태의 bounding box가 test data로 주어진다면 이를 예측 불가능
* 작은 bounding box의 loss가 IOU에 민감하게 영향을 미쳐 localization에 안좋은 영향을 미침%


## 성능
<p align="center"><img width="366" alt="스크린샷 2022-06-02 오후 3 27 10" src="https://user-images.githubusercontent.com/56713634/171566775-25912cde-361f-40f9-bc9b-e7eda2829fab.png"></p>


