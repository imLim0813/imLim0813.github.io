---
layout : post
title : Gradient Descent
categories : [Deep Learning, Fundamental]
use_math : true
---

## 0. Introduction

&nbsp;딥러닝(Deep Learning)은 실제 값과 예측 값 사이의 오차를 줄여나가는 과정을 통해 학습을 진행합니다.<br>
오차를 줄이기 위해 파라미터를 반복적으로 업데이트해야하고, 이를 위해 경사하강법(Gradient Descent)을 사용합니다.

## 1. Gradient Descent

&nbsp; 경사하강법(Gradient Descent)이란, 딥러닝 모델의 에러를 최소화하기 위해 손실함수의 기울기(경사)를 따라 파라미터를 점진적으로 업데이트해 나가는 최적화 방법을 의미합니다 &nbsp;( $ w = w - \eta \nabla J(w)$ ).<br><br>

<figure class="GradientDescent">
    <img src = "../assets/img/gradient_descent_simple.png" width="50%" height="50%" alt="Alt text">
    <div style="text-align : center;">
        <figcaption> Gradient Descent 설명 </figcaption>
    </div>
</figure>
<br>

## 2. Gradient Descent 방해 요인

&nbsp; 딥러닝의 손실함수는 일반적으로 아래 그림과 같이 볼록이 아닌(non-convex) 형태를 가집니다.<br>
이러한 형태때문에 딥러닝 학습에 방해가 되는 국소적 최소점(Local Minimum)과 안장점(Saddle Point)가 발생합니다.

* #### 전역적 최소점(Global Minimum)
: 손실함수(loss function)에서 값이 가장 작은 지점이며, 이상적인 모델은 이 지점에 도달합니다.

* #### 국소적 최소점(Local Minimum)
: 전역적 최소점이 아닌 극소점을 의미합니다. <br> : 국소적 최소점은 $\nabla J(w)$가 0이 되어 파라미터 업데이트가 더 이상 이루어지지 않으므로 문제가 됩니다.

* #### 안장점(Saddle Point)
: 모든 파라미터의 기울기는 0이지만, 극소점은 아닌 지점을 의미합니다. <br> : 국소적 최소점과 같이 $\nabla J(w)$가 0이 되어 파라미터 업데이트가 더 이상 이루어지지 않으므로 문제가 됩니다.

<br>

<figure class="SaddlePoint">
    <img src = "../assets/img/saddle_point.png" width="50%" height="50%" alt="Alt text">
    <div style="text-align : center;">
        <figcaption> Global Minimum, Local Minimum, Saddle Point </figcaption>
    </div>
</figure>
<br>

## 3. Appendix

| (1) 손실함수가 non-convex 형태를 띄는 이유는 예측값($\hat y$)의 비선형성 때문입니다. <br><br> (2) 딥러닝이 뉴턴메소드가 아닌 경사하강법을 사용하는 이유는 &nbsp;① 계산량, &nbsp;② 손실함수가 non-convex이기 때문입니다. <br><br> (3) 실제 딥러닝 학습에 문제가 되는 것은 국소적 최소점(Local Minimum)이 아니라, 안장점(Saddle Point)이라고 합니다. <br><br> (4) 딥러닝은 &nbsp;① 학습률 조정(Learning Rate), &nbsp;② 옵티마이저(Optimizer) 등 여러 기법을 활용해 안장점 문제를 해결합니다.


## 4. Conclusion

&nbsp; 딥러닝은 경사하강법을 통해 파라미터를 반복적으로 업데이트하여 에러를 줄여나가도록 학습합니다.<br>
손실함수는 딥러닝의 모델 구조 특성상 볼록하지 않은(non-convex) 형태를 띄게 되며, 이는 딥러닝 학습에 방해가 되는 <br>
국소적 최소점(Local Minimum)과 안장점(Saddle Point)을 발생시킵니다.<br>이를 해결하기 위해, 딥러닝은 &nbsp;① 학습률 조정(Learning Rate), &nbsp;② 옵티마이저(Optimizer) 등 여러 기법을 활용합니다.<br><br>
&nbsp; <b>결국, 딥러닝은 경사하강법 기반의 다양한 옵티마이저를 활용해 모델의 파라미터를 최적화하고, 은닉층을 통해 데이터를 학습하여 패턴과 특징을 추출하는 알고리즘입니다.</b>