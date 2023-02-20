# An Image is Worth 16 x 16 Words : Transformer for Image Recognition at Scale

# Abstract
* 자연어처리분야(NLP)에서 transformer은 널리 사용됨
* 컴퓨터비전분야에서는 아직 제한적임 
  > Attention은 cnn과 같은 convolution network를 같이 사용해야함
* ViT(Vision Transformer)는 이러한 cnn의 의존을 제거하고 image pathch를 통해 Transformer를 직접 수행
* 대용량 데이터셋으로 pre-training한 뒤 작은 양의 데이터로 transferred하는 방식으로 기존 모델들 보다 좋은 성능 및 적은 게산량을 가짐

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
[사진 넣기]

* Transformer은 1D token embedding sqeunce를 input으로 받음
* 2D 이미지 <img width="90" alt="스크린샷 2023-02-20 오후 5 28 31" src="https://user-images.githubusercontent.com/56713634/220052481-817875f6-db40-4f00-87c4-8df968f5df1b.png"> 를 flatten 된 2D 이미지 패치인 <img width="103" alt="스크린샷 2023-02-20 오후 5 28 26" src="https://user-images.githubusercontent.com/56713634/220052590-24072529-f75d-404e-a94e-d106d22771fe.png"> 로 재구성
  > 이미지를 (P x P) 크기의 패치 N = HW/P^2로 분할하여 구축
  
  > (H, W) : 원본 이미지의 해상도 | (P, P) : 각 이미지 패치의 해상도 | C : 채널 개수 | N : 패치 개수
* Trainable linear projection을 통해 x_p의 각 패치를 flatten한 백터 D차원으로 변환 후 사용 이를 __patch embdding__ 
  > 모든 layer에 고정된 백터 크기 D를 사용하기 때문에
* BERT의 [class]토큰과 비슷하게 임베딩(Learnable class)된 패치의 시퀀스에 z0 = x_class 임베딩을 추가로 붙여 넣음
  > 식 (4)에 해당
* 패치에 대해 나온 인코더 output은 representation으로 해석하여 분류에 사용
  > L번의 encoder를 거친 후의 output 중 learnable class 임베딩과 관련된 부분
* position embedding 은 위치 정보를 유지하기 위해 patch embedding에 더해짐
* Transformer 인코더는 Multi-headed self-attention 및 MLP block
* Layernorm(LN)은 모든 block 이전에 적용되고 residual connection 은 모든 block 이후에 적용

[식 첨부]

#### Inductive Bias
ViT는

#### Hybrid Architecture


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
[테이블 1 첨부]

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
  > 최소제곱회귀문제
  
  > closed form으로 구할 수 있어서, 간혹 fine-tuning accuracy의 연산량이 부담될 때만 사용했음.
* Fine-tuning accuracy: Fine-tuning 후의 정확도

## Comparison to State of The Art
저자들의 모델은 ViT-H/14, ViT-L/16 기존의 SOTA는 CNN 기반 모델
* Big Transfer (BiT) : large ResNet을 이용해 supervised transfer learning 수행
* Noisy Student : large EfficientNet을 이용해 semi-supervised learning 수행(ImageNet과 라벨이 지워진 JFT-300M 데이터셋)

[table2 첨부]

* 거의 모든 데이터셋에서 ViT-H/14 모델이 가장 높은 성능
* 기존 SOTA 모델인 BiT-L 보다도 높은 성능이며 더 적은 시간
  > 더 작은 모델인 ViT-L/16 또한 BiT-L 보다 높은 성능과 적은 시간
  
[사진 2]

