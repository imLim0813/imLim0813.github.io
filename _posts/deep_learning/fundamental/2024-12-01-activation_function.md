---
layout : post
title : Activation Function
categories : [Deep Learning, Fundamental]
use_math : true
---

## 0. Introduction

&nbsp; 딥러닝은 실제 데이터의 복잡한 패턴 혹은 비선형적 관계를 학습하는 알고리즘입니다.<br>
복잡한 패턴을 학습하기 위해 모델은 비선형성을 가져야하고, 이를 위해 활성화 함수(Activation function)을 사용합니다.<br>

## 1. Activation function이란?

&nbsp; 활성화 함수(Activation function)은 딥러닝 모델에 비선형성을 부여하기 위해 사용되는 함수입니다.<br>
활성화 함수는 아래 식에서 $\sigma$&nbsp;에 해당하며, 가중치(weight; $W$)와 편향(bias; $b$)의 선형조합에 비선형성을 부여합니다.<br>

|$Y = \sigma(W^{T}X + b)$

## 2. Activation function 종류

* #### 시그모이드 함수 (Sigmoid)
: 시그모이드 함수는 모든 입력값을 0과 1 사이의 값으로 변환하는 함수입니다.<br>
&nbsp; 이 값은 확률($p$)의 범위와 동일하기 때문에, 이진 분류 문제에서 출력층 활성화 함수로 자주 사용됩니다.
<br><br>: 수식은 $\sigma(x) = \frac {1} {1 + e^{-x}}$이고, 미분은 $\sigma^{\prime}(x) = \sigma(x)(1-\sigma(x))$입니다.
<br>

<figure class="Sigmoid">
    <img src = "../assets/img/sigmoid.png" width="90%" height="90%" alt="Alt text">
    <div style="text-align : center;">
        <figcaption> Sigmoid 함수 그림 </figcaption>
    </div>
</figure>

* #### 하이퍼블릭 탄젠트 함수 (tanh)
: 하이퍼블릭 탄젠트 함수는 모든 입력값을 -1과 1 사이의 값으로 변환하는 함수입니다.<br>
&nbsp; 이 함수는 시그모이드 함수의 출력이 항상 양수라는 단점을 극복하여 더 큰 범위를 표현하도록 고안되었습니다.
<br><br>: 수식은 $\sigma(x) = \frac {e^{x} - e^{-x}} {e^{x} + e^{-x}}$이고, 미분은 $\sigma^{\prime}(x) = 1-\sigma^{2}(x)$입니다.
<br>

<figure class="tanh">
    <img src = "../assets/img/tanh.png" width="90%" height="90%" alt="Alt text">
    <div style="text-align : center;">
        <figcaption> tanh 함수 그림 </figcaption>
    </div>
</figure>