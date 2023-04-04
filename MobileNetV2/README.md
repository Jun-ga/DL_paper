# MobileNetV2: Inverted Residuals and Linear Bottlenecks
# Introdution
* 최신 네트워크는 많은 모바일 및 임베디드 네트워크의 기능을 뛰어넘는 높은 컴퓨팅 리소스가 필요
* 본 논문에서는 모바일 및 리소스가 제한된 환경에 맞게 특별히 설계된 새로운 신경망을 소개
* 주요 기술로는 linear bottleneck layer로 고차원으로 확장되고, 저차원으로 표현되는 특징을 가짐
* 이러한 기술을 통해 모든 최신 프레임워크에서 표준 작업을 사용하여 효율적으로 구현될 수 있으며 표준 벤치마크를 사용하여 여러 성능 포인트에서 최신 모델을 능가
* 또한 모바일 설계에 적합 -> 메모리 공간을 크게 줄일 수 있으므로

# Preliminaries, discussion and intuition
## Depthwise Separable Convolutions
[MobileNetv1](https://github.com/Jun-ga/DL_paper/tree/main/MobileNetV1)

## Linear Bottlenecks
n개의 layer L_i로 구성된 신경망으로 고려
* 이미지에 대해 각 layer에는 h_i x w_i x d_i 크기의 activation tensor가 존재
* 이를 d_i 크기의 차원의 __pixels__ 이 h_i x w_i 개 있다고 생각
* 실제 입력 이미지에 대해 layer activation은 __manifold of interest__ 을 형성
  > manifold : 고차원 데이터가 저차원으로 압축되면서 특정 정보들이 저차원의 어떤 영역으로 매핑되는 것
* 지금까지의 연구들은 신경망에서 manifold of interest은 low-dimensional subspaces으로 embedded될 수 있다 가정



MobileNetv1에서는 bottleneck layer을 적용하여 성공적으로 차원을 줄여 네트워크를 효율적으로 만들었다.

<p align="center"><img width="433" alt="스크린샷 2023-04-04 오후 4 57 16" src="https://user-images.githubusercontent.com/56713634/229726424-a9a05ec1-dc92-41dd-a4ab-bbc1b4c39317.png"></p>

그러나, ReLU에 의해 정보 손실이 일어날 수 있음
* 저차원의 맵핑 정보 손살이 큼, 차원수가 커지면 정보 손실이 줄어든다.

<p align="center"><img width="425" alt="스크린샷 2023-04-04 오후 4 57 53" src="https://user-images.githubusercontent.com/56713634/229726584-2c3942eb-d8a0-4724-87f0-435b375b973b.png">
</p>


1. manifold of interest가 ReLU 변환 이후에도 non-zero volume으로 남아 있다면, 그것은 선형 변환에 부합
2. ReLU가 입력 manifold에 대해 정보를 완전히 보전하는 경우는 입력 manifold가 입력 space의 저차원 subspace에 있을 때

즉, linear layer를 추가하여 정보 파괴를 방지 __linear bottleneck__

## Inverted residuals

<p align="center"><img width="438" alt="스크린샷 2023-04-04 오후 4 58 33" src="https://user-images.githubusercontent.com/56713634/229726701-8dda0670-fad5-4792-ac00-30adf6e7a788.png">
</p>

__residual block__
* 일반적인 residual connection과 반대 __wide - narrow - wide__
* 처음에 들어오는 입력은 wide, 1x1 conv를 이용하여 채널을 줄여 다음 layer bottleneck 형성
* bottleneck에서는 3x3 conv를 이용하여 conv 연산 후 다시 skip connection과 합쳐지기 위해 size 복원

__inverted residual__
* __narrow - wide - narrow__ 형태로 narrow layer끼리 연결되어있음
* 이미 필요한 정보는 narrow에 저장되어있기 떄문에 skip connection으로 사용해도 정보를 더 깊은 layer로 전달 가능
  > 압축된 narrow layer를 skip connection으로 사용함으로써 메모리 사용량을 줄임
* 양끝의 빗금이 쳐져 있는 layer는 linear bottleneck을 의미하고, Relu를 사용하지않음

## Information flow interpretation
본 구조의 흥미로운 점은 building blocks (bottleneck layers)의 input/output domains와 transformation 사이의 분리가 자연스럽게 이뤄진다는 점

* 전자는 capacity 후자는 expressiveness 부분
* 기존의 convolution이 capacity와 expressiveness가 서로 합쳐져 있는 것에 비해 대조적임



# Model Architecture


<p align="center"><img width="460" alt="스크린샷 2023-04-04 오후 4 59 04" src="https://user-images.githubusercontent.com/56713634/229726845-45b5e579-1440-4f00-9078-7dad31471a3d.png">
</p>

* 32개의 filters를 갖는 fully conv layer
* 19개의 residual bottleneck layer
* activate function ReLU6


<p align="center"><img width="431" alt="스크린샷 2023-04-04 오후 4 59 37" src="https://user-images.githubusercontent.com/56713634/229726935-2ae24179-6614-438c-92ce-986d142f57dc.png">
</p>


* input size는 Imagenet data로 224 x 224 x 3 사이즈, 최종 k는 1000
* 모든 실험에서 inverted residual의 확장 비율은 6으로 고정
  > input 채널 64 -> wide layer 386
* t = expansion factor
* c = channal
* n = iteration
* s = stride

### Trade-off hyper parameter
입력이미지 해상도와 폭 배율을 원하는 정확도/성능 trage off에 따다 조정할 수 있도록 fine-tunung가능하도록 hyper parameter 사용

__basic model__

* multiplier 1
* input 224 x 224
* 300M개의 multiply-adds
* 3.4M개의 parameter

<p align="center"><img width="440" alt="스크린샷 2023-04-04 오후 5 00 10" src="https://user-images.githubusercontent.com/56713634/229727117-08dea5af-4d92-40d5-ab83-e74a8874afba.png">
</p>


Mobilenet v2의 채널수가 적고(bottleneck 에서만 expansion시키고 다시 projection하기 때문) 메모리 사용량도 더 적기 때문에 embadded system에 가장 적합

# Experiments
## ImageNet Classification
__Training setup__
* TensorFlow
* RMSPropOptimizer (decay and momentum set to 0.9)
* 모든 layer에 batch normalization
* 16 GPU, 96개의 batchsize

__Results__

<p align="center"><img width="423" alt="스크린샷 2023-04-04 오후 5 01 04" src="https://user-images.githubusercontent.com/56713634/229727307-ab2923d9-a041-4b01-a35d-dbe936cdb223.png">
</p>
<p align="center"><img width="437" alt="스크린샷 2023-04-04 오후 5 00 53" src="https://user-images.githubusercontent.com/56713634/229727392-4318f0b3-44ed-4d03-9458-22db5c2f6345.png">
</p>

## Object Detection
__SSDLite__
<p align="center"><img width="459" alt="스크린샷 2023-04-04 오후 5 02 33" src="https://user-images.githubusercontent.com/56713634/229728071-524728b9-5940-424a-ad2f-30177220a250.png">
</p>
* MobileNet과 잘 어울리는 모델로 기존 SSD의 conv를 separable conv로 변경
* 기존보다 계산적인 효율을 높여 준다. 기존 SSD와 비교하여 parameter 수와 계산량을 획기적

__Results__
<p align="center"><img width="453" alt="스크린샷 2023-04-04 오후 5 02 37" src="https://user-images.githubusercontent.com/56713634/229728142-a3295122-66e3-4268-8ab0-7467b7d2279b.png">
</p>

# Conclusions and future work
* 고효율 모바일 모델 제품군을 구축할 수 있는 네트워크 아키텍처인 MobileNetV2를 제안
* ssdlite와 결합했을때 yolov2보다 적은 매개변수와 계산량을 가짐
* capacity와 expressiveness를 분리할 수 있다는 점에서 향수 연구의 중요한 방향일 것임




