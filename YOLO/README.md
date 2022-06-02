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

<p align="center"><img width="642" alt="스크린샷 2022-06-02 오후 3 28 42" src="https://user-images.githubusercontent.com/56713634/171567004-7eb9e631-343a-4561-b083-4fb3b2e63efb.png"></p>


1. object가 존재하는 grid cell i의 예측 bounding box j에 대해, x와 y의 loss를 계산.
2. object가 존재하는 grid cell i의 에측 bounding box j에 대해, w와 h의 loss를 계산. 
큰 box에 대해서는 small deviation을 반영하기 위해 제곱근을 취한 후, sum-squared error를 한다.(같은 error라도 larger box의 경우 상대적으로 IOU에 영향을 적게 준다.)
3. object가 존재하는 grid cell i의 예측 bounding box j에 대해, confidence score의 loss를 계산. (Ci = 1)
4. object가 존재하지 않는 grid cell i의 bounding box j에 대해, confidence score의 loss를 계산. (Ci = 0)
5. object가 존재하는 grid cell i에 대해, conditional class probability의 loss 계산. (Correct class c: pi(c)=1, otherwise: pi(c)=0)

λcoord: coordinates(x,y,w,h)에 대한 loss와 다른 loss들과의 균형을 위한 balancing parameter
λnoobj: obj가 있는 box와 없는 box간에 균형을 위한 balancing parameter

## 한계

* 각 grid cell 마다 B개의 bounding box만 추측해내야하므로 objet가 겹쳐있으면 제대로 예측 불가능
* data로 부터 bounding box를 훈련시키기 때문에,학습데이터에는 없는 새로운 형태의 bounding box가 test data로 주어진다면 이를 예측 불가능
* 작은 bounding box의 loss가 IOU에 민감하게 영향을 미쳐 localization에 안좋은 영향을 미침%


## 성능
<p align="center"><img width="366" alt="스크린샷 2022-06-02 오후 3 27 10" src="https://user-images.githubusercontent.com/56713634/171566775-25912cde-361f-40f9-bc9b-e7eda2829fab.png"></p>

# YOLOv2

## better
### 1. Batch Normalization
* Batch Normalization은 다른 regularization의 필요성을 없애고 신경망을 더 빠르게 수렴
* YOLO의 convolutional layer에 Batch Normalization을 이용하여 mAP를 2% 상승시켰고, dropout을 제거
### 2. High Resolution Classifier
* YOLOv1은 입력 이미지 224x224로 학습을 한다. 하지만 detection을 할 때는 입력 이미지 448x448을 하기때문에 강제로 cnn 입력을 바꿔 네트워크가 잘 적응하지 못하여 성능이 낮다고 생각함
* 이에, YOLO v2는 ImageNet dataset에서 448x448 이미지로 fine tuning 및 detection을 위한 fune tuning을 해줌
* 4% mAP를 증가

### 3. Convolustional with Anchor Boxes

fully connected layer를 제거하고 anchor box를 활용해서 바운딩 박스를 예측 즉, 1x1 convolutional layer로 예측

* 입력이미지 크기를 448x448에서 416x416으로 변경
 * 416을 32로 down sampling시 13x13의 feature map 생성, 물체가 이미지 중앙에 있는 경우가 많기 때문에 output feature map은 홀수x홀수가 더 좋음
 * grid마다 클래스를 예측하는 것이 아닌 모든 Anchor Box마다 클래스 예측
 * anchor box를 이용하여 mAP은 감소하고 recall이 증가
   >  anchor box(x) : 69.5 mAP, 81% recall
   
   >  anchor box(o) 69.2 mAP, 88% recall
   
### 4. Dimension Clusters
기존의 모델들은 anchor box의 크기와 aspect ratio를 사전에 수작업(hyperparameter)으로 정의, YOLOv2에서는 수작업으로 정의하는 방법보다 더 좋은 사전 anchor box들을 이용하면 더 좋은 detection 성능을 보여줄 것이라 생각

* k-means clustering 방법을 통해 자동으로 training dataset의 ground truth를 clustering해서 anchor box의 크기와 aspect ratio를 정의
* Euclidean distance를 이용할 경우 큰 bounding box가 작은 bounding box에 비해 큰 error를 발생시키는 문제발생, 이에 저자들은 아래의 식을 도입(IOU를 기준으로 k-means clustering)

<p align="center"><img width="305" alt="스크린샷 2022-06-02 오후 3 49 15" src="https://user-images.githubusercontent.com/56713634/171571208-0fa958c8-930f-4915-aadc-d04f938e908b.png"></p>

<p align="center"><img width="593" alt="스크린샷 2022-06-02 오후 3 49 07" src="https://user-images.githubusercontent.com/56713634/171570640-f42186c1-c5c8-45d8-885c-41c625742395.png"></p>
* k = 5로 설정

<p align="center"><img width="410" alt="스크린샷 2022-06-02 오후 3 49 22" src="https://user-images.githubusercontent.com/56713634/171571789-afa7bf7a-6725-4a1b-bd02-8709aa79298c.png"></p>

### 5. Direct location prediction

* anchor box를 활용한 bounding box offset 예측법은 bounding box의 위치를 제한하지 않아서 초기 학습시 불안정하다는 단점 발생
* 예측값의 범위를 제한하지 않으면 bounding box가 이미지 어디에도 나타날 수 있습니다. 
* 안정적으로 바운딩박스를 예측하기 까지 모델이 학습되는데에는 많은 시간 소요
* 이를 해결하기 위해 sigmoid를 사용하여 범위를 0~1로 제한

<p align="center"><img width="409" alt="스크린샷 2022-06-02 오후 4 09 09" src="https://user-images.githubusercontent.com/56713634/171573859-98b06923-e433-4a05-afce-05bd42e4f670.png"></p>


__bounding box 중심 좌표 예측값을 sigmoid 함수로 감싸 위치 제한__
<p align="center"><img width="518" alt="스크린샷 2022-06-02 오후 4 08 53" src="https://user-images.githubusercontent.com/56713634/171574115-e43b3323-6efc-4e4e-9342-7351965980f0.png"></p>

### 6. Fine-Grained Features

YOLOv2의 13x13 feature map은 큰 물체를 탐지하는데 충분할 수 있으나 작은 물체를 잘 탐지하지 못 할 수 있다. 이를 해결하기 위해 13x13 feature map을 얻기 전의 앞 쪽의 layer에서 26x26 해상도의 feature map을 passthrough layer를 통해 얻는다.

### 7. Multi-scale Training

YOLO v2는 다른 크기의 이미지로부터 robust를 갖기 위해 다양한 크기로 학습
* 10갸의 batch들 마다 새로운 이미지 차워을 적용
* 1/32배 downsampling을 진행하므로 학습 이미지는 31의 배수 _다양한 입력크기에도 잘 예측이 가능해짐_




