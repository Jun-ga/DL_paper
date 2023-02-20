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
  > __inductive bias__를 transformer에서는 고려할 수 없기 때문에 일반화가 어렵다는 문제
* large-scale 데이터 세트에서 학습할때엔 inductive bias를 능가
  > 즉, 충분한 규모로 사전 학습되고 더 적은 데이터로 fine tuning 할때 좋은 결과 도출
* ImageNet : 88.36%, CIFAR-100 : 94.55% 등의 성능


# METHOD
## VISION TRANSFORMER (VIT)
[사진 넣기]
