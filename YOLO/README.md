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

1. 입력이지미를 5x5 grid로 분할
2. 객체의 중심이 grid cell에
