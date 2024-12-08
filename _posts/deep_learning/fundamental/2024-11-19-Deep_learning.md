---
layout : post
title : Deep Learning
categories : [Deep Learning, Fundamental]
use_math : true
---

## 0. Paper

&nbsp;[Gardner, M. W., & Dorling, S. R. (1998). Artificial neural networks (the multilayer perceptron)—a review of applications in the atmospheric sciences. Atmospheric environment, 32(14-15), 2627-2636.](https://www.sciencedirect.com/science/article/pii/S1352231097004470?casa_token=f8_QOZhLyhoAAAAA:ssCaDodlVSNfVRx4AAE078T2Sl6aTlloL8sJnvyOLRELSh6GK3qCQlujLo2PCkEDdHZEnTc9rAc){:target="_blank"}

## 1. Introduction

&nbsp;최근 생성형 AI와 컴퓨터 비전 기술의 발전으로 딥러닝이 주목받고 있으며, 다양한 딥러닝 모델이 개발되고 있습니다.<br>
해당 포스트에서는 딥러닝의 개념, 장단점 그리고 전반적인 학습과정에 대해 다루겠습니다.


## 2. Terminology

* #### 퍼셉트론 (Perceptron)
: 퍼셉트론은 신경과학자들이 인간의 정보 처리 방식을 모방하여 고안한 초기 인공 신경망 모델로, 현대 딥러닝 모델의 근간이 되는 기초적인 아이디어를 제공한 모델입니다. <br><br>: 퍼셉트론은 XOR 문제를 해결하지 못한다는 치명적인 단점이 있었습니다.
<figure class="Perceptron">
    <img src = "../assets/img/perceptron.png" width="90%" height="90%" alt="Alt text">
    <div style="text-align : center;">
        <figcaption> The Perceptron </figcaption>
    </div>
</figure>

* #### 다층 퍼셉트론 (Multi Layer Perceptron, MLP)
: XOR문제를 해결하기 위해 고안된 모델로써, 입력층과 출력층을 제외한 은닉층이 하나 이상 존재하는 모델입니다.
<figure class="MultiLayerPerceptron">
    <img src = "../assets/img/multi_layer_perceptron.png" width="90%" height="90%" alt="Alt text">
    <div style="text-align : center;">
        <figcaption> Multi Layer Perceptron, MLP </figcaption>
    </div>
</figure>

* #### 순방향 신경망 (Feed Forward Network, FNN)
: 보통 딥러닝 모델은 데이터의 흐름이 순방향(입력층 → 은닉층 → 출력층)입니다. 이러한 특성 때문에 순방향 신경망(feed forward network)라고 부릅니다.

* #### 완전 연결 신경망 (Fully Connected Network, FCN)
: 완전 연결 신경망(fully connected network)은 모델 내 모든 뉴런이 연결된 기본적인 신경망 구조입니다.<br><br>: 구조가 단순하여 구현이 쉬워 딥러닝 모델의 기본적인 형태로 자주 사용되지만, 매우 많은 파라미터를 가지게 되어 과적합(overfitting) 문제를 일으킬 수 있습니다.

## 3. Learning procedure

| (1) 딥러닝 모델의 파라미터를 초기화합니다. <br><br> (2) 입력 데이터를 모델의 입력층에 전달합니다. <br><br> (3) 모델의 파라미터를 이용해 입력 데이터에 대한 출력 값을 계산합니다. <br><br> (4) 실제 값과 모델의 출력 값 사이의 오차를 계산합니다. <br><br> (5) 오차를 기반으로 파라미터를 업데이트합니다. <br><br> (6) 오차가 충분히 줄어들 때까지 이 과정을 반복합니다.

## 4. Pros and Cons

* #### 딥러닝의 장점

| (1) 딥러닝 모델은 비선형적인 관계를 학습할 수 있으므로, 통계적 모델 (선형 회귀, 로지스틱 회귀)과 달리<br> &nbsp; &nbsp; &nbsp; 데이터의 분포에 대한 가정 없이 학습이 가능합니다.<br><br> (2) 실제 문제의 대부분은 비선형적이기 때문에, 딥러닝이 통계적 모델보다 우수한 성능을 보이는 경우가 많습니다.<br><br> (3) 딥러닝은 raw data에서 복잡한 패턴을 학습할 수 있기 때문에, feature extraction 과정이 필요하지 않습니다.

* #### 딥러닝의 단점

| (1) 훌륭한 모델을 만들기 위한 규칙 (① 레이어의 깊이, ② 노드의 개수)이 존재하지 않습니다.<br><br> (2) 은닉층에서 feature extraction과정을 처리해주므로 어떤 feature를 추출하는지 알 수 없습니다.<br> &nbsp; &nbsp; &nbsp; 이러한 해석하기 어렵다는 특징 때문에 딥러닝을 Black box라고 부릅니다.

## 5. Conclusion

&nbsp; 딥러닝은 복잡한 패턴과 비선형적인 관계를 학습할 수 있어 ① 기존의 통계적 모델에 비해 우수한 성능을 보이며, ② 데이터의 분포나 특징에 대한 가정이 필요 없습니다. 그러나, 은닉층이 학습한 패턴을 해석하기 어려운 단점이 있어 상황에 따라 기존의 통계적 모델과 딥러닝을 적절히 선택하여 사용하는 것이 바람직합니다.