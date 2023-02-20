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
* 이미지를 patch로 분할하고 이 patch들을 sequence
