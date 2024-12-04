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
이 값은 확률($p$)의 범위와 동일하기 때문에, 이진 분류 문제에서 출력층 활성화 함수로 자주 사용됩니다.
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
이 함수는 시그모이드 함수의 출력이 항상 양수라는 단점을 극복하여 더 큰 범위를 표현하도록 고안되었습니다.
<br><br>: 수식은 $\sigma(x) = \frac {e^{x} - e^{-x}} {e^{x} + e^{-x}}$이고, 미분은 $\sigma^{\prime}(x) = 1-\sigma^{2}(x)$입니다.
<br>

<figure class="tanh">
    <img src = "../assets/img/tanh.png" width="90%" height="90%" alt="Alt text">
    <div style="text-align : center;">
        <figcaption> tanh 함수 그림 </figcaption>
    </div>
</figure>

* #### ReLU 함수 (ReLU)
: ReLU 함수는 입력값이 0보다 크면 그 값 자체를, 0보다 작으면 0을 반환하는 함수입니다.<br>
이 함수는 이전에 소개한 두 활성화 함수(Activation function)의 편미분 값이 1보다 작거나 같아서 발생하는<br>
기울기 소실(Vanishing Gradient) 문제를 해결하기 위해 고안되었습니다.
<br><br>: 수식은 $\sigma(x) = max(0, x)$이고, 미분은 $\sigma^{\prime}(x) = 1\;\text{if}\; x > 0, 0\;\text{if}\; x \leq 0$입니다.
<br>

<figure class="ReLU">
    <img src = "../assets/img/ReLU.png" width="90%" height="90%" alt="Alt text">
    <div style="text-align : center;">
        <figcaption> ReLU 함수 그림 </figcaption>
    </div>
</figure>

* #### Swish 함수 (Swish)
: Swish 함수는 ReLU 도함수의 입력이 0보다 작을 경우 기울기가 0인 문제를 해결하고자 고안됐습니다.<br>
이 함수는 도함수의 입력이 0보다 작더라도 기울기가 음수를 유지하여 Dying neuron 문제를 방지합니다.
<br><br>: 수식은 $Swish(x) = x * \sigma(x) \; (\sigma(x) = sigmoid(x))$ 이고,
<br> 미분은 $Swish^{\prime}(x) = \sigma(x) * (1 + x (1-\sigma(x)))$입니다.
<br>

<figure class="Swish">
    <img src = "../assets/img/Swish.png" width="90%" height="90%" alt="Alt text">
    <div style="text-align : center;">
        <figcaption> Swish 함수 그림 </figcaption>
    </div>
</figure>

## 3. Vanishing Gradient / Exploding Gradient
: 딥러닝 모델의 성능 개선을 위해 우선적으로 고려해볼 수 있는 방법은 모델의 레이어를 깊게 쌓는 것입니다.<br>
그러나 이 경우, 오차 역전파 과정에서 그래디언트 값(Backward Value)이 급격히 작아지거나 커질 위험이 있습니다. <br>

<figure class="Vanishing Gradient">
    <img src = "../assets/img/vanishing_gradient.png" width="90%" height="90%" alt="Alt text">
    <img src = "../assets/img/vanishing_gradient_2.png" width="90%" height="90%" alt="Alt text">
    <div style="text-align : center;">
        <figcaption> Back propagation 연산 </figcaption>
    </div>
</figure>

* #### 기울기 소실 (Vanishing Gradient)
: Sigmoid 함수와 tanh 함수의 그림을 보면, 편미분 최댓값이 각각 0.25, 1입니다. <br>
따라서, 레이어가 깊어질수록 1보다 작거나 같은 값들이 반복적으로 곱해지게 되어 Backward Value가 <br> 0으로 수렴하게 되는 현상을 기울기 소실(Vanishing Gradient) 현상이라고 합니다. <br>

* #### 기울기 폭발 (Exploding Gradient)
: Back propagation 그림을 보면, Backward Value 계산을 위해, 가중치 값도 사용되는 것을 알 수 있습니다. <br>
만약, 이 가중치들이 충분히 큰 값들이라면, 레이어가 깊어질수록 Backward Value가 폭발적으로 증가하게됩니다.<br>
이러한 현상을 기울기 폭발(Exploding Gradient) 현상이라고 합니다.

## 4. Appendix

| (1) Leaky ReLU, FReLU, PReLU, ELU 함수 등의 활성화함수가 존재합니다. (ReLU의 변형이라 제외하였습니다) <br><br> (2) Softmax 함수는 출력층에만 사용되므로 해당 포스트에서는 제외시켰습니다.<br><br> (3) 기울기 소실(Vanishing Gradient) 문제를 해결하기 위해, 여러 기법이 사용됩니다.<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;① ReLU함수 사용 : 편미분 값이 0 혹은 1이므로 기울기 소실을 방지 <br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;② Batch Normalization : 활성화 함수의 입력을 정규화시켜 출력 값이 Saturation region에 빠지지 않도록 유지<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;③ Weight Initialization : 적절한 초기 가중치 값을 통해 활성화 함수가 Saturation region에 빠지지 않도록 유지 <br><br> (4) 기울기 폭발(Exploding Gradient) 문제를 해결하기 위해서 위 방법과 더불어 여러 테크닉을 사용할 수 있습니다.<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;① Adam, RMSProp 등의 옵티마이저를 통한 적절한 학습률 선택<br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;② 기울기 클리핑 (Gradient Clipping) : 기울기가 일정 값을 넘어가면 잘라버리는 기법 <br><br> (5) ReLU는 기울기 소실 문제는 해결하지만, 기울기 폭발 문제는 해결하지 못합니다.

## 5. Conclusion

&nbsp; 현실에 존재하는 데이터는 종종 복잡한 패턴을 가지고 있습니다. 이러한 데이터를 학습하기 위해, 딥러닝 모델은 각 레이어 출력에 활성화 함수(Activation function)를 추가하여 모델에 비선형성을 부여합니다. 이를 통해 모델은 복잡한 데이터의 패턴을 학습합니다.