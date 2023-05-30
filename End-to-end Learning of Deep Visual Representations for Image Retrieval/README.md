# End-to-end Learning of Deep Visual Representations for Image Retrieval

# Introdution
* Instance-level image retrieval은 query image가 인가되었을때 image database내에서 query와 동일한 개체를 찾는 것을 목표함
* deep learning(특히, CNN)은 computer vision에서 강력한 tool
  > 하지만 Instance-level image retrieval에 대해서는 낮은 성능을 보임
* 대부분의 retrieval methods는 네트워크를 local feature extractors로 사용
  > ImageNet같은 대규모 dataset에서 사전학습된 모델을 활용
  > 이는 결국에는 이미지 representation에 적합한 이미지를 설계하는 곳에 focus
  > Instance-level image에 대한 지도학습이 부족하기 때문에 성능 저하
* 본 논문에서는 __learning representations that are well suited for the retrieval task__ 에 초점을 맞춤
  > 동일한 class에 속하더라도 그 속에서의 특정 객체를 구별하는 것에 관심이 있는 task이므로
* R-MAC과 triplet loss을 기반으로 함
* 다른 input image resolution에 대해 single descripton으로 encording하는 방법을 제안


# Leveraging large-scale noisy data
instance-level image retrieval을 위한 informative, efficient representation을 학습하기 위해서는 dataset이 중요
  > 기존 dataset을 활용, 자동으로 정리하는 방법은 제안

그림 1

* 214k images of 672 famous landmark sites을 포함하는 대규모 dataset인 Landmarks dataset을 활용
* 이는 landmark와 완전히 관련없는 이미지도 포함됨
* 이러한 dataset에서 image가 너무 적은 class를 제거하며, test시 사용하는 Oxford 5k, Paris 6k 및 Holidays dataset와 겹치는 class를 제거함
* 이를 통해 586개의 landmark로 나눠진 192,000개의 image data를 얻음
