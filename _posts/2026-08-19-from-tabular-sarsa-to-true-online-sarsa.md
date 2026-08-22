---
layout: post
title: "从表格 Sarsa 到 True Online Sarsa(λ)"
title_html: "从表格 Sarsa 到 True Online Sarsa(λ)"
date: 2026-08-19 12:00:00 +0800
categories: [Reinforcement Learning]
tags: [Sarsa, function-approximation, eligibility-trace, true-online]
series: "强化学习基础"
math: true
---

> 本文重点停在：**为什么传统 accumulating trace 在 online 情况下不能精确匹配 online forward view**。

---

## 1. 整体链路

这一阶段的学习主线是：

$$
\boxed{
\text{表格 Sarsa}
\rightarrow
\text{线性函数近似}
\rightarrow
\text{半梯度 Sarsa}
\rightarrow
\text{函数近似下的资格迹 / Sarsa}(\lambda)
\rightarrow
\text{online forward view 与传统 backward view 的 mismatch}
}
$$

这一条链路主要解决三个问题：

1. **状态—动作空间太大时，如何不用 Q 表也能表示动作价值，并让相似状态之间共享经验。**
2. **函数近似以后，如何把当前 TD error 更快地分配给过去参与过预测的参数方向。**
3. **在线更新时参数持续变化，为什么传统 accumulating trace 不能在一般步长下精确复现 online forward view。**

---

## 2. 为什么需要函数近似

表格 Sarsa 为每个状态—动作对保存一个独立的：

$$
Q(s,a).
$$

这种表示很直观，但在连续状态空间或非常大的状态空间中会遇到两个问题：

- 无法为所有状态—动作对分别存储一个独立表项；
- 相似状态之间不能自动共享经验。

因此，我们使用参数化函数：

$$
\boxed{
\hat q(s,a,\mathbf w)
}
$$

来近似动作价值。

在线性函数近似中：

$$
\boxed{
\hat q(s,a,\mathbf w)
=
\mathbf w^\top \mathbf x(s,a)
}
$$

其中：

$$
\mathbf x(s,a)
=
\begin{bmatrix}
x_1(s,a)\\
x_2(s,a)\\
\vdots\\
x_d(s,a)
\end{bmatrix}
$$

表示状态—动作对的特征向量，而：

$$
\mathbf w
=
\begin{bmatrix}
w_1\\
w_2\\
\vdots\\
w_d
\end{bmatrix}
$$

是模型真正学习的参数向量。

因此：

$$
\hat q(s,a,\mathbf w)
=
w_1x_1(s,a)+w_2x_2(s,a)+\cdots+w_dx_d(s,a).
$$

几个量需要严格区分：

- $\mathbf x(s,a)$：描述当前状态—动作具有哪些特征；
- $\mathbf w$：模型学习每个特征应该如何影响价值；
- $\hat q(s,a,\mathbf w)$：模型对动作价值的当前估计。

---

### 2.1 直观例子：连续位置的机器人

假设机器人的位置是连续变量。

表格法会把：

$$
1.20
$$

和：

$$
1.21
$$

视为两个不同状态。

即使在位置 $1.20$ 已经学到“向右移动很好”，位置 $1.21$ 也不能直接利用这条经验。

函数近似则可以让两个状态共享诸如：

- 接近目标；
- 当前速度较低；
- 向右动作；
- 前方没有障碍；

等特征。

因此，当共享参数 $\mathbf w$ 更新后，多个具有相似特征的状态—动作价值都会发生变化。

![多个相互错开的瓦片编码共同表示二维连续状态]({{ '/assets/images/tile-coding-overlapping-tilings.jpg' | relative_url }})
<p class="figure-caption">图 1：二维连续状态空间中的重叠瓦片编码。同一个状态会在每个 tiling 中激活一个 tile；相邻状态共享的 active tiles 越多，线性函数近似产生的泛化越强。图源：Richard S. Sutton、Andrew G. Barto，《Reinforcement Learning: An Introduction》（第 2 版），Figure 9.9。</p>

所以共享参数会同时带来两个结果：

$$
\boxed{
\text{generalization}
\quad\text{和}\quad
\text{interference}
}
$$

我的理解是：

<span style="color:#d1242f"><strong>共享参数的意义不是单纯换一种方式保存 Q，而是用有限参数表达大量状态—动作之间的共同结构。</strong></span>

---

## 3. 半梯度 Sarsa

一步 Sarsa 的 TD error 仍然由：

$$
\boxed{
\text{即时奖励}
+
\text{下一状态—动作估计}
-
\text{当前估计}
}
$$

构成。

严格写成：

$$
\boxed{
\delta_t
=
R_{t+1}
+
\gamma
\hat q(S_{t+1},A_{t+1},\mathbf w_t)
-
\hat q(S_t,A_t,\mathbf w_t)
}
$$

需要注意：

$$
\hat q(S_t,A_t,\mathbf w_t)
$$

和：

$$
\hat q(S_{t+1},A_{t+1},\mathbf w_t)
$$

在计算当前 $\delta_t$ 时都使用同一组参数：

$$
\mathbf w_t.
$$

函数近似以后，真正被更新的不再是某一个独立的 Q 表项，而是参数向量：

$$
\mathbf w_t.
$$

半梯度 Sarsa 的更新为：

$$
\boxed{
\mathbf w_{t+1}
=
\mathbf w_t
+
\alpha
\delta_t
\nabla_{\mathbf w}
\hat q(S_t,A_t,\mathbf w_t)
}
$$

在线性函数近似下：

$$
\hat q(S_t,A_t,\mathbf w_t)
=
\mathbf w_t^\top\mathbf x_t
$$

其中：

$$
\boxed{
\mathbf x_t
=
\mathbf x(S_t,A_t)
}
$$

于是：

$$
\boxed{
\nabla_{\mathbf w}
\hat q(S_t,A_t,\mathbf w_t)
=
\mathbf x_t
}
$$

因此：

$$
\boxed{
\mathbf w_{t+1}
=
\mathbf w_t
+
\alpha\delta_t\mathbf x_t
}
$$

![Mountain Car 任务及半梯度 Sarsa 学到的 cost-to-go 曲面]({{ '/assets/images/semi-gradient-sarsa-mountain-car.jpg' | relative_url }})
<p class="figure-caption">图 2：Mountain Car 任务以及半梯度 Sarsa 配合 tile coding 在线性动作价值函数上逐步学到的 cost-to-go 曲面。它展示了有限参数如何覆盖连续的 position–velocity 状态空间。图源：Richard S. Sutton、Andrew G. Barto，《Reinforcement Learning: An Introduction》（第 2 版），Figure 10.1。</p>

---

### 3.1 如何理解半梯度

假设：

$$
\hat q_t=2
$$

而一步 Sarsa 的 target 为：

$$
4.6.
$$

当前更新希望让预测：

$$
2
$$

朝：

$$
4.6
$$

靠近。

半梯度并不是说 target 中的下一步价值以后永远不会变化。

它真正表示的是：

> 在**当前这一次求梯度**时，把 bootstrap target 暂时当作固定参考，不沿 target 分支继续求导。

也就是说，当前更新只通过：

$$
\hat q(S_t,A_t,\mathbf w)
$$

这一支计算梯度。

但是，如果当前状态和下一状态共享部分参数，那么更新：

$$
\mathbf w_t
$$

以后，下一状态的价值估计：

$$
\hat q(S_{t+1},A_{t+1},\mathbf w)
$$

仍然可能被间接改变。

因此：

<span style="color:#d1242f"><strong>半梯度只是当前求导时不沿 bootstrap target 求导，并不意味着 target 中的价值估计在后续参数更新后保持不变。</strong></span>

这也为后面 True Online 中的 online prediction drift 埋下了伏笔。

---

## 4. 从表格资格迹到参数资格迹

在表格 Sarsa($\lambda$) 中，每个过去访问过的状态—动作对都有资格迹：

$$
\boxed{
z_t(s,a)
}
$$

它表示：

> 过去某个 $(s,a)$ 在当前时刻还有多少资格接收新的 TD error。

对于 accumulating trace：

$$
\boxed{
z_t(s,a)
=
\gamma\lambda z_{t-1}(s,a)
+
\mathbb I\{S_t=s,A_t=a\}
}
$$

函数近似以后，真正被修改的是共享参数：

$$
\mathbf w_t.
$$

因此，资格迹也需要从状态—动作空间搬到参数空间。

一般形式：

$$
\boxed{
\mathbf z_t
=
\gamma\lambda\mathbf z_{t-1}
+
\nabla_{\mathbf w}
\hat q(S_t,A_t,\mathbf w_t)
}
$$

在线性函数近似下：

$$
\nabla_{\mathbf w}
\hat q(S_t,A_t,\mathbf w_t)
=
\mathbf x_t
$$

所以：

$$
\boxed{
\mathbf z_t
=
\gamma\lambda\mathbf z_{t-1}
+
\mathbf x_t
}
$$

当前 TD error 再通过资格迹更新参数：

$$
\boxed{
\mathbf w_{t+1}
=
\mathbf w_t
+
\alpha\delta_t\mathbf z_t
}
$$

---

### 4.1 如何区分 $\delta_t$、$\mathbf z_t$ 和 $\mathbf w_t$

我现在把几个量区分为：

$$
\boxed{
\delta_t
=
\text{当前发现“错了多少”}
}
$$

$$
\boxed{
\mathbf z_t
=
\text{过去哪些参数方向目前还有多少 credit}
}
$$

$$
\boxed{
\mathbf w_t
=
\text{真正被修改的模型参数}
}
$$

更准确地说：

<span style="color:#d1242f"><strong>$\mathbf z_t$ 是参数空间中的向量，它压缩记录当前和过去预测梯度的衰减历史，并决定当前 TD error 沿哪些参数方向产生多大影响。</strong></span>

---

### 4.2 直观例子：延迟奖励

假设轨迹：

$$
A
\rightarrow
B
\rightarrow
C
\rightarrow
\text{terminal}
$$

最后得到奖励：

$$
5.
$$

如果在最后一步，A、B、C 对应的资格大约为：

$$
0.52,\quad 0.72,\quad 1.00,
$$

那么最后出现的 TD error 会按照这些资格影响过去参与过预测的参数方向。

这里不是：

> 把奖励 5 平均分给 A、B、C。

而是：

> 同一个当前 TD error，根据不同资格值，对不同历史参数方向产生不同程度的更新。

---

## 5. Forward view 与 backward view 的联系

资格迹可以从两个方向理解。

### 5.1 Forward view

站在过去的状态—动作：

$$
(S_k,A_k)
$$

向未来看：

> 后面出现哪些 TD error 应该影响当前这个过去状态—动作？

在用于建立直觉的固定参数情形下，可以写成：

$$
\boxed{
G_k^\lambda
-
\hat q_k
=
\delta_k
+
\gamma\lambda\delta_{k+1}
+
(\gamma\lambda)^2\delta_{k+2}
+
\cdots
}
$$

所以过去的预测梯度：

$$
\mathbf x_k
$$

会收到后续多个 TD error 的加权影响。

---

### 5.2 Backward view

站在当前 TD error：

$$
\delta_t
$$

向过去看：

> 当前这个 prediction error 应该影响哪些过去状态—动作，或者哪些过去的参数方向？

资格迹：

$$
\mathbf z_t
$$

把过去梯度的衰减历史压缩在一个递推变量中。

因此不需要在每一个时间步重新遍历整条轨迹。

传统 backward view 的参数更新为：

$$
\boxed{
\mathbf w_{t+1}
=
\mathbf w_t
+
\alpha\delta_t\mathbf z_t
}
$$

可以把两种视角概括成：

> **Forward view：站在过去，看未来哪些 error 应该回来。**
> **Backward view：站在当前 error，看过去哪些 credit 应该收到它。**

![TD(lambda) 中当前 TD error 沿资格迹影响过去状态的后向视图]({{ '/assets/images/td-lambda-backward-view.jpg' | relative_url }})
<p class="figure-caption">图 3：TD(λ) 的 backward view。当前 TD error 与逐步衰减的资格迹结合，将更新分配给过去参与预测的状态或参数方向。图源：Richard S. Sutton、Andrew G. Barto，《Reinforcement Learning: An Introduction》（第 2 版），Figure 12.5。</p>

---

## 6. Online 参数更新破坏了“看起来完全相同的 Q”

一开始推导 forward / backward view 时，我看到望远镜消除中一正一负、形式相同的价值估计，很容易认为：

> 它们就是同一个数，可以直接抵消。

但真正的 online 学习中：

$$
\mathbf w_0
\rightarrow
\mathbf w_1
\rightarrow
\mathbf w_2
\rightarrow
\cdots
$$

参数会在轨迹进行过程中持续更新。

因此，同一个状态—动作在两个相邻时间点上的估计可能分别是：

$$
\hat q(S_t,A_t,\mathbf w_{t-1})
$$

和：

$$
\hat q(S_t,A_t,\mathbf w_t).
$$

虽然：

$$
(S_t,A_t)
$$

完全相同，但因为：

$$
\mathbf w_{t-1}
\neq
\mathbf w_t,
$$

所以一般可能有：

$$
\boxed{
\hat q(S_t,A_t,\mathbf w_{t-1})
\neq
\hat q(S_t,A_t,\mathbf w_t)
}
$$

因此：

<span style="color:#d1242f"><strong>看到两个代数形式相同的 $\hat q(S_t,A_t)$，不能立刻认为它们是同一个数，必须检查它们分别由哪一组参数计算。</strong></span>

这是我之前理解 forward / backward equivalence 时忽略的一点：

> 我只关注了数学形式，没有对参数版本和时间索引保持足够敏感。

---

## 7. 为什么 traditional accumulating trace 在 online 情况下不能精确匹配 forward view

最容易理解的是共享参数。

假设两个不同状态：

$$
S_0,\quad S_1
$$

都只依赖同一个参数：

$$
w.
$$

因此：

$$
\hat v(S_0,w)=w,
\qquad
\hat v(S_1,w)=w.
$$

最初：

$$
w=0.
$$

如果先因为 $S_0$ 的学习把参数更新为：

$$
w:0\rightarrow0.25,
$$

那么由于 $S_1$ 使用同一个参数：

$$
\hat v(S_1,w)
$$

也会自动从：

$$
0
$$

变成：

$$
0.25.
$$

也就是说：

> 过去一次参数更新，已经提前改变了当前或未来状态的 prediction。

如果接下来 $S_1$ 的 target 是：

$$
1,
$$

那么此时真正的 prediction error 应该基于当前已经更新后的：

$$
0.25
$$

计算。

所以当前剩余误差是：

$$
1-0.25=0.75.
$$

这里需要特别注意：

> conventional Sarsa($\lambda$) 并不是“忘了重新计算 TD error”，也不是仍然机械地把 error 当成 1。

真正的问题在于：

> 前面发生的参数更新已经改变了后续 prediction，而 traditional accumulating trace 仍然只按简单的 $\gamma\lambda$ 方式累计历史梯度，因此整个 online backward-view 参数更新轨迹不能在一般步长下精确复现 online forward view。

传统 accumulating trace：

$$
\boxed{
\mathbf z_t
=
\gamma\lambda\mathbf z_{t-1}
+
\mathbf x_t
}
$$

默认历史 credit 从过去传到当前，只经历：

$$
\gamma\lambda
$$

的时间衰减。

但在 online 函数近似中，历史 credit 在传到当前的过程中，还会受到中间参数更新的影响。

因此问题不是：

$$
\boxed{
\text{TD error 本身被放大}
}
$$

而是：

$$
\boxed{
\text{整套 online credit bookkeeping 不再精确}
}
$$

或者更直接地说：

$$
\boxed{
\text{conventional backward view}
\neq
\text{online forward view}
}
$$

在一般有限步长下，两者不能严格一致。

<span style="color:#d1242f"><strong>关键矛盾不在于 TD error 是否被重新计算，而在于传统资格迹没有完整追踪在线参数变化对历史 credit 的影响。</strong></span>

![Mountain Car 上 True Online Sarsa(lambda) 与常规 Sarsa(lambda) 的性能对比]({{ '/assets/images/true-online-sarsa-mountain-car-comparison.jpg' | relative_url }})
<p class="figure-caption">图 4：Mountain Car 上多种 Sarsa(λ) 的早期性能对比。True Online Sarsa(λ) 在该实验中优于使用 accumulating traces 和 replacing traces 的常规版本，为下一部分引入精确匹配 online forward view 的修正提供了经验动机。图源：Richard S. Sutton、Andrew G. Barto，《Reinforcement Learning: An Introduction》（第 2 版），Figure 12.11。</p>

---

## 8. 学习后的几个理解与反思

### 8.1 半梯度不是假设下一步 Q 永远不变

半梯度只表示：

> 当前更新求梯度时，不沿 bootstrap target 的参数路径继续求导。

参数更新以后，如果当前状态和下一状态共享参数，下一状态的 prediction 仍然可能改变。

---

### 8.2 函数近似的关键不只是“压缩 Q 表”

真正重要的是：

$$
\boxed{
\text{parameter sharing}
}
$$

它带来：

- generalization；
- interference。

共享参数使相似状态能够互相利用经验，同时也意味着一次局部更新可能影响其他状态的预测。

---

### 8.3 我之前对参数版本不够敏感

看望远镜消除时，我以前只注意到：

$$
+\hat q(S_t,A_t)
$$

和：

$$
-\hat q(S_t,A_t)
$$

形式一样。

现在需要进一步检查：

$$
\hat q(S_t,A_t,\mathbf w_{t-1})
$$

和：

$$
\hat q(S_t,A_t,\mathbf w_t)
$$

是否真的由同一组参数计算。

---

### 8.4 True Online 的核心问题不是“传统 trace 太大”

更准确的说法是：

$$
\boxed{
\text{True Online 不是为了解决“trace 太大”，而是为了解决 online forward/backward mismatch。}
}
$$

在 online 学习中，参数持续变化。

traditional accumulating backward view 只记录：

$$
\gamma\lambda
$$

意义上的历史衰减，却没有完整记录中间在线参数更新如何改变已经存在的历史影响。

因此它一般不能在有限步长下精确复现 online forward view。

---

## 9. 这一阶段的总结

**1. 从独立表项到共享参数**

表格 Sarsa 为每个状态—动作对单独学习 $Q(s,a)$。这种表示清晰直接，但面对连续或大规模状态空间时，既难以存储，也无法自然地在相似状态之间泛化。

线性函数近似把动作价值写成：

$$
\hat q(s,a,\mathbf w)
=
\mathbf w^\top\mathbf x(s,a).
$$

特征 $\mathbf x(s,a)$ 描述状态—动作，参数 $\mathbf w$ 则在不同状态—动作之间共享。这样既产生了泛化，也可能带来相互干扰。

**2. 半梯度 Sarsa：沿当前预测方向更新**

一步 TD error 为：

$$
\delta_t
=
R_{t+1}
+
\gamma\hat q(S_{t+1},A_{t+1},\mathbf w_t)
-
\hat q(S_t,A_t,\mathbf w_t).
$$

半梯度方法在当前这一步把 bootstrap target 当作固定参考，只沿当前预测的梯度方向更新。在线性函数近似下：

$$
\mathbf w_{t+1}
=
\mathbf w_t+\alpha\delta_t\mathbf x_t.
$$

**3. 资格迹：保存历史参数方向**

Sarsa($\lambda$) 使用参数资格迹：

$$
\mathbf z_t
=
\gamma\lambda\mathbf z_{t-1}
+
\mathbf x_t,
$$

把当前与过去预测梯度的衰减历史压缩到一个向量中。当前 TD error 再通过 $\mathbf z_t$ 影响仍保有 credit 的参数方向，从而更快地完成延迟 credit assignment。

**4. 传统 accumulating trace 记录了什么**

传统 accumulating trace 显式记录的是历史梯度按 $\gamma\lambda$ 产生的时间衰减。它能够高效实现传统 backward view，但递推式本身没有完整编码轨迹中每次在线参数更新造成的 prediction drift。

**5. Online mismatch 从哪里产生**

在真正的在线学习中，参数沿轨迹持续变化：

$$
\mathbf w_0
\rightarrow
\mathbf w_1
\rightarrow
\mathbf w_2
\rightarrow
\cdots
$$

由于参数是共享的，过去的更新会提前改变当前与未来状态—动作的 prediction。因此，即使两个价值估计写成同样的 $\hat q(S_t,A_t)$，只要参数版本不同，它们就不一定相等。

**6. 这一阶段的核心结论**

<span style="color:#d1242f"><strong>在一般有限步长下，traditional accumulating backward view 不能精确复现 online forward view；要实现真正的 online equivalence，还需要修正资格迹和参数更新。</strong></span>

---

## 10. 下一步留下的问题

这一阶段最后留下的核心问题是：

> **既然历史 credit 从过去传到当前时，不只是按 $\gamma\lambda$ 做时间衰减，还会受到中间 online 参数更新的影响，那么“正确的 online 资格迹”应该怎样递推？**

下一部分将沿着这条问题链继续：

- Dutch trace 如何修正传统资格迹；
- $Q_{\mathrm{old}}$ correction 为什么会出现在参数更新中；
- 两项修正如何组合成 True Online Sarsa($\lambda$)，并精确对应 online forward view。
