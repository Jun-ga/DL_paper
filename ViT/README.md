# An Image is Worth 16 x 16 Words : Transformer for Image Recognition at Scale

# Abstract
* 자연어처리분야(NLP)에서 transformer은 널리 사용됨
* 컴퓨터비전분야에서는 아직 제한적임 
  > Attention은 cnn과 같은 convolution network를 같이 사용해야함
* ViT(Vision Transformer)는 이러한 cnn의 의존을 제거하고 image pathch를 통해 Transformer를 직접 수행
* 대용량 데이터셋으로 pre-training한 뒤 작은 양의 데이터로 transferred하는 방식으로 기존 모델들 보다 좋은 성능 및 적은 계산량을 가짐

# Introdution
* NLP에서 Transformer는 large text corpus에서 사전 훈련을 수행, specific dataset에 대해 fine-tuning하는 방식
* 컴퓨터비전 분야에서 cnn과 혹은 유사한 구조와 self-attention을 결합하려고 노력
  > cnn 등에 의존적임
  > 본 논문의 저자들은 Transformer를 직접 사용해보는 실험 진행
* 이미지를 patch로 분할하고 이 patch들의 embedding sequence를 입력으로 설정
  > 이미지의 patch들은 NLP의 token과 동일한 방식으로 처리
* ImageNet과 같은 mid-sized의 데이터셋에서 학습을 진행했을때 resnet보다 몇 퍼센트 낮은 정확도를 달성
  > __inductive bias__ 를 transformer에서는 고려할 수 없기 때문에 일반화가 어렵다는 문제
* large-scale 데이터 세트에서 학습할때엔 inductive bias를 능가
  > 즉, 충분한 규모로 사전 학습되고 더 적은 데이터로 fine tuning 할때 좋은 결과 도출
* ImageNet : 88.36%, CIFAR-100 : 94.55% 등의 성능


# METHOD
모델 설계시 원래의 Transformer를 가능한 가깝게 따름

## VISION TRANSFORMER (ViT)
<p align="center"><img width="600" alt="스크린샷 2023-02-20 오후 11 46 44" src="https://user-images.githubusercontent.com/56713634/220137650-b62d91ea-c78f-4ff8-8991-17f2530ba098.png"></p>


* Transformer은 1D token embedding sqeunce를 input으로 받음
* 2D 이미지 <img width="90" alt="스크린샷 2023-02-20 오후 5 28 31" src="https://user-images.githubusercontent.com/56713634/220052481-817875f6-db40-4f00-87c4-8df968f5df1b.png"> 를 flatten 된 2D 이미지 패치인 <img width="103" alt="스크린샷 2023-02-20 오후 5 28 26" src="https://user-images.githubusercontent.com/56713634/220052590-24072529-f75d-404e-a94e-d106d22771fe.png"> 로 재구성
  > 이미지를 (P x P) 크기의 패치 N = HW/P^2로 분할하여 구축 <br>
  > ex) P = 128, H,W가 256이면 N = 4 즉 128x128xC의 patch가 총 4개
  
  > (H, W) : 원본 이미지의 해상도 | (P, P) : 각 이미지 패치의 해상도 | C : 채널 개수 | N : 패치 개수
* Trainable linear projection을 통해 x_p의 각 패치를 flatten한 백터 D차원으로 변환 후 사용 이를 __patch embedding__ 
  > 모든 layer에 고정된 백터 크기 D를 사용하기 때문에
* position embedding 은 위치 정보를 유지하기 위해 patch embedding에 더해짐
* BERT의 [class]토큰과 비슷하게 임베딩(Learnable class)된 패치의 시퀀스에 z0 = x_class 임베딩을 추가로 붙여 넣음
* 임베딩을 인코더에 입력, 이를 통해 output으로 image representation 출력
  > L번의 encoder를 거친 후의 output 중 learnable class 임베딩과 관련된 부분
* MLP Head에 image representation을 입력시켜 분류

#### 오른쪽 그림에 대한 추가 설명

* Transformer 인코더는 Multi-headed self-attention 및 MLP block
* Layernorm(LN)은 모든 block 이전에 적용되고 residual connection 은 모든 block 이후에 적용

<p align="center"><img width="871" alt="스크린샷 2023-02-20 오후 11 47 49" src="https://user-images.githubusercontent.com/56713634/220137900-05325066-c7a3-4f2f-a5db-0070e369dbe3.png"></p>

1) 각 patch 값
2) Muti-head attention
3) MLP(Muti-Layer Perceptron)
4) LN(Layer Norm=Normalization Layer)에 z^0_L 넣어 y 획득

![image1](https://user-images.githubusercontent.com/56713634/220147446-72d6f2cb-8ce9-4eaf-a7f3-2fb01c7fc35b.gif)

#### Inductive Bias
_학습과정에서 보지 못한 데이터 또한 추론할 수 있도록 모델이 가지고 있는 가정_

* ViT가 CNN보다 image별 inductive bias가 적다는 것에 주목
* transformer은 self-attention을 기반으로 하고 있기에 낮은 Inductive Bias 가짐
* ViT의 Multi-Head self Attention 또한 Inductive Bias가 낮음

__이를 극복하기 위한 2가지 방법__
* Patch Extration : 이미지를 여러 개의 패치로 분할 및 순서대로 입력
* Resolution adjustment : 패치의 크기가 동일하지만 생성되는 패치 개수는 다르기 때문에, fine-tuning 단계에서 positional embedding을 할 때 조절

#### Hybrid Architecture
mage patch 대신 CNN featrue map을 flatten하여 transformer 차원으로 projection하여 사용 가능

## Fine-Tuning And Higher Resolution
* Large dataset에 pre-trained한 다음, down stream tasks에 fine-tuning
* down stream tasks에 적용하기 위하여 pre-train 된 prediction head 를 제거하고 0으로 초기화 된 D x K feedforward layer 를 연결
  > K : downstream class 의 수
* pre-trained할 때의 이미지 해상도보다 높은 해상도로 fine-tuning 을 하는 것이 성능에 도움
* 고해상도의 이미지가 주어질 때 패치의 크기는 동일하게 유지되므로 시퀀스의 길이가 더 커짐
  > ViT는 가변적 길이의 패치를 처리할 수는 있지만 pre-trained position embeddings는 의미를 잃게 될 수 있음
  > 이 경우  pre-trained position embedding을 원본 이미지의 위치에 따라 2D interpolation 수행
  
# Experiments
* ResNet, Vision Transformer(ViT) 및 하이브리드의 represiontation learning capability 를 평가
* 각 모델의 데이터 요구사항을 이해하기위해 다양한 크기의 데이터 세트를 사전 학습하고 많은 벤치 마크에서 작업을 수행
* __모델을 pre-train 하는 계산 비용을 고려하여 ViT 는 더 낮은 pre-train 비용으로 대부분의 recognition 벤치마크에서 SOTA 를 달성__

## Setup
#### Datasets
아래 3개의 dataset으로 pre-train 
* ILSVRC-2012 ImageNet 1k classes 및 1.3M images 
* ImageNet-21k 21k classes 및 14M images 
* JFT 18k classes 및 303M images

pre-trained 한 모델들을 benchmark tasks에 transfer 
* ReaL labels, CIFAR-10/100, Oxford-IIIT Pets, Oxford Flowers-102
* 19-task VTAB classification suite

#### Model Variants
3개의 size로 실험 진행 각 사이즈에서 다양한 패치 크기에 대한 실험 진행
<p align="center"><img width="590" alt="스크린샷 2023-02-20 오후 11 48 29" src="https://user-images.githubusercontent.com/56713634/220138063-eaa300a7-7c03-4c3a-972f-e2c94d0425ee.png"></p>

#### Traing & Fine-tuning
pre-train
* Adam optimizer : β1 = 0.9, β21 = 0.999 batchsize = 4096
* weight decay : 0.1

Fine-turning
* SGD with momentum
* Higher resolution :512 for ViT-L/16 , 518 for ViT-H/14

#### Mtrics
평가 지표로는 few-shot accuracy와 fine-tuning accuracy를 고려
* Few-shot accuracy: Training set에 없는 클래스를 맞추는 문제에 대한 정확도
  > closed form으로 구할 수 있어서, 간혹 fine-tuning accuracy의 연산량이 부담될 때만 사용했음.
* Fine-tuning accuracy: Fine-tuning 후의 정확도

## Comparison to State of The Art
저자들의 모델은 ViT-H/14, ViT-L/16 기존의 SOTA는 CNN 기반 모델
* Big Transfer (BiT) : large ResNet을 이용해 supervised transfer learning 수행
* Noisy Student : large EfficientNet을 이용해 semi-supervised learning 수행(ImageNet과 라벨이 지워진 JFT-300M 데이터셋)

<p align="center"><img width="859" alt="스크린샷 2023-02-20 오후 11 49 31" src="https://user-images.githubusercontent.com/56713634/220138264-74f849c2-90a0-4faf-8d34-81a531b4160c.png"></p>

* 거의 모든 데이터셋에서 ViT-H/14 모델이 가장 높은 성능
* 기존 SOTA 모델인 BiT-L 보다도 높은 성능이며 더 적은 시간
  > 더 작은 모델인 ViT-L/16 또한 BiT-L 보다 높은 성능과 적은 시간
  
<p align="center"><img width="844" alt="스크린샷 2023-02-20 오후 11 49 38" src="https://user-images.githubusercontent.com/56713634/220138376-33acea7f-af13-4684-a248-2893c8614a7e.png"></p>

## Pre-training Data Requirements
ViT는 대규모 사이즈의 dataset인 JFT-300M에 대해 pre-train하였을 때 더 좋은 성능을 보여줌

ResNet 보다 vision 에 대한 inductive bias 가 적을 때 데이터 세트의 크기가 얼마나 중요한지에 대한 실험을 수행

<p align="center"><img width="850" alt="스크린샷 2023-02-20 오후 11 50 43" src="https://user-images.githubusercontent.com/56713634/220138570-ef5552f5-8d26-4773-9c54-b49d0267fb05.png"></p>

#### Fiure 3
* 가장 작은 dataset에서는 Base보다 Large모델의 성능이 떨어짐
* 21K를 사용시 성능이 비슷해짐

#### Fiure 4
JFT 데이터셋을 각각 다른 크기로 랜덤 샘플링한 데이터셋을 활용하여 진행
* 작은 dataset에서는 확실히 inductive bias 효과가 있는 CNN 계열의 BiT가 높은 성능
* 큰 dataset으로 갈수록 ViT 성능이 더 좋아지는 것을 확인

## Scaling Study
FT-300M 데이터세트에서 transfer 성능에 대해 다양한 모델로 확장된 연구를 수행한 결과


<p align="center"><img width="850" alt="스크린샷 2023-02-20 오후 11 50 52" src="https://user-images.githubusercontent.com/56713634/220138646-bebeac48-238b-464d-a6ae-9d0cbdec1637.png"></p>


* 같은 시간이 소모되었을 때 ViT가 더 높은 성능
* __성능과 cost의 trade-off에서 ViT가 BiT보다 우세__
* Cost가 낮을 때는 Hybrid가 ViT보다 유리한 듯 하지만 Cost가 높아지면서 trade-off 차이가 감소

## Inspecting Vision Transformer
ViT 가 이미지를 처리하는 방법을 이해하기 위해 분석

<p align="center"><img width="866" alt="스크린샷 2023-02-20 오후 11 52 06" src="https://user-images.githubusercontent.com/56713634/220138851-edaffac7-904f-4b21-a82c-34a604b2b5c4.png"></p>

#### 왼쪽
* ViT 의 첫번째 레이어는 flatten patch 를 더 낮은 차원 공간에 projection
* 왼쪽은 학습된 embedding filter 의 구성 요소
* 구성요소는 각 patch 내 미세 구조의 low-dimensional representation 에서 basic function 과 유사
* projection 된 이후 학습된 position embedding 이 patch representation 에 추가

#### 가운데
* 모델이 position embedding 의 유사성에서 이미지 내 거리를 인코딩 하는 방법을 학습한다는 것을 보여줌
* 구성요소는 각 patch 내 미세 구조의 low-dimensional representation 에서 basic function 과 유사
* 더 가까운 patch 는 더 유사한 position embedding 을 갖는 경향이 있으며 행-열 구조
* Self-attention 을 통해 ViT 는 가장 낮은 레이어에서도 전체 이미지에 대한 정보를 보여줌

#### 오른쪽 
* Self-attention 의 weight 를 기반으로 정보가 통합되는 이미지 공간의 평균 거리를 오른쪽 그림과 같이 계산
* attention distance 는 CNN의 receptive field size 와 유사
* 일부 head 에서는 최하위 레이어에 있는 대부분의 이미지에 attention 을 적용하여 global 하게 모델을 사용할 수 있다는 것을 보여줌
* 구성요소는 각 patch 내 미세 구조의 low-dimensional representation 에서 basic function 과 유사


<p align="center"><img width="278" alt="스크린샷 2023-02-20 오후 11 51 57" src="https://user-images.githubusercontent.com/56713634/220138938-4557d9e9-cd0f-4da5-b045-d457a61f2ba3.png"></p>

# Conclusion

* 이미지 인식 분야에서 Transformer 를 직접 적용하는 방법을 제안
* 논문에서는 구조에 image-specific inductive bias 를 사용 X
* patch 로 해석하고 NLP에서 사용되는 standard transformer encoder 로 처리
* 대규모의 데이터 세트에 대해 pre-train 될 때 잘 작동 
* ViT 는 많은 이미지 분류 데이터세트에서 SOTA 달성, pre-train 비용이 굉장히 저렴하다는 장점
