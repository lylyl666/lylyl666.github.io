---
layout: post
title: "从 True Online Sarsa(λ) 到 IDBD 与 TIDBD"
date: 2026-09-18 13:49:00 +0800
categories: [Reinforcement Learning]
tags: [Sarsa, true-online, dutch-trace, IDBD, TIDBD, meta-learning]
series: "强化学习基础"
math: true
---

> 上一篇[《从表格 Sarsa 到 True Online Sarsa(λ)》]({{ '/posts/from-tabular-sarsa-to-true-online-sarsa/' | relative_url }})停在了一个问题：**传统 accumulating trace 只显式记录历史梯度按 $\gamma\lambda$ 的衰减，为什么无法在一般有限步长下精确复现 online forward view？**
>
> 这一篇先补上这个问题的答案：Dutch trace 和 $Q_{\mathrm{old}}$ correction。然后沿着一个新问题继续：**既然参数已经能够正确地在线更新，为什么仍要给所有 feature 指定同一个学习率？学习率本身能不能学习？** 这就把我们带到 IDBD 和 TIDBD。

---

## 1. 本篇主线：两个不同层次的学习问题

上一篇已经介绍了表格 Sarsa、线性函数近似、半梯度、参数资格迹及 online forward/backward mismatch。这里不再从头重复，只保留必要的符号和衔接。

$$
\boxed{
\text{online mismatch}
\;\longrightarrow\;
\text{Dutch trace + }Q_{\mathrm{old}}\text{ correction}
\;\longrightarrow\;
\text{True Online Sarsa}(\lambda)
}
$$

$$
\boxed{
\text{固定步长 }\alpha
\;\longrightarrow\;
\text{IDBD}
\;\longrightarrow\;
\text{TIDBD}(0)
\;\longrightarrow\;
\text{TIDBD}(\lambda)
}
$$

这两段并不是一条算法直接替换另一条算法的关系：**True Online 主要讨论在线更新与 forward view 的对应；IDBD/TIDBD 主要讨论如何调整每个 feature 的步长。** 后续 SwiftTD 才会进一步把相关思想组合起来。

### 1.1 记号约定

沿用 Sutton & Barto 的习惯：$S_t,A_t,R_{t+1}$ 表示状态、动作、奖励；$\mathbf w_t$ 为参数向量；$\mathbf x_t$ 为特征；$\mathbf z_t$ 为资格迹；$\delta_t$ 为 TD error；$\gamma$ 为折扣率；$\lambda$ 控制 trace 的时间衰减。$i$ 表示第 $i$ 个特征坐标，不表示时间。

前半篇讨论 **Sarsa 动作价值**，此时

$$
\mathbf x_t=\mathbf x(S_t,A_t),\qquad
Q_t\doteq\hat q(S_t,A_t,\mathbf w_t)=\mathbf w_t^\top\mathbf x_t.
$$

后半篇介绍原始 **TIDBD 状态价值预测**，届时将显式切换成 $\mathbf x_t=\mathbf x(S_t)$ 和 $\hat v(S_t,\mathbf w_t)$。不能把状态价值预测的算法公式不加说明就称为 Sarsa 控制算法。

---

## 2. Dutch trace：历史 credit 为什么不只是按 $\gamma\lambda$ 衰减？

上一篇结尾留下的 conventional accumulating trace 为

$$
\mathbf z_t=\gamma\lambda\mathbf z_{t-1}+\mathbf x_t,
\qquad
\mathbf w_{t+1}=\mathbf w_t+\alpha\delta_t\mathbf z_t.
$$

它能够让当前 TD error 沿历史特征方向更新参数，但不能保证整条在线参数轨迹精确等于 online forward view。这里**不是**说传统算法忘记重算 TD error，也不是说 trace 数值只要大就一定错误；关键是更新后的共享参数会改变后续预测，而传统 trace 的递推没有完整反映这种影响。

### 2.1 从一个共享参数例子开始

设两个不同的状态—动作对都使用同一个参数 $w_1$，它们当前的预测均为 $0$。如果前一个状态—动作的学习将 $w_1$ 更新为 $0.25$，后一个状态—动作的预测也会因为参数共享变为 $0.25$。

如果后者的 target 为 $1$，现在的预测误差应该基于**新参数**计算：

$$
1-0.25=0.75.
$$

常规 Sarsa($\lambda$) 也会重新计算 TD error。真正需要进一步处理的是：过去更新对当前预测产生的影响，如何在整条 forward/backward 更新轨迹中被正确记账。

### 2.2 先做一个单次更新的代数观察

暂时把一个 forward-view target 固定为 $G$，看线性半梯度更新映射：

$$
F_t(\mathbf w)
=\mathbf w+\alpha\bigl(G-\mathbf w^\top\mathbf x_t\bigr)\mathbf x_t.
$$

设目前的参数可以拆成一个基础参数加上过去学习留下的扰动：

$$
\mathbf w_t=\mathbf w_t^{\mathrm{base}}+\Delta\mathbf w.
$$

将这两份起点代入**同一个 target、同一个步长**的更新映射，比较更新之后的差异：

$$
\begin{aligned}
&F_t(\mathbf w_t^{\mathrm{base}}+\Delta\mathbf w)
-F_t(\mathbf w_t^{\mathrm{base}})\\
&=\Delta\mathbf w
-\alpha(\Delta\mathbf w^\top\mathbf x_t)\mathbf x_t\\
&=\bigl(\mathbf I-\alpha\mathbf x_t\mathbf x_t^\top\bigr)
\Delta\mathbf w.
\end{aligned}
$$

因此定义一个**仅用于推导的辅助矩阵**：

$$
\boxed{\mathbf M_t\doteq\mathbf I-\alpha\mathbf x_t\mathbf x_t^\top}.
$$

不记成 $A_t$，因为在 Sutton & Barto 的符号中，$A_t$ 已经表示动作。$\mathbf M_t$ 也不是要额外学习的模型参数；它刻画本轮线性更新如何传播已经存在的参数扰动。

注意其中的内积：

$$
\Delta\mathbf w^\top\mathbf x_t.
$$

如果它为 $0$，说明这份历史扰动不影响当前预测，本轮更新不会改变该扰动；如果不为 $0$，旧扰动会沿当前特征方向发生变化。

**推导边界：** 这是固定单步 target 时的精确代数观察，用于说明修正项的来源；它本身尚不是完整的 True Online forward/backward 等价性证明。严格证明还需要处理沿时间变化的 online $\lambda$-return 及其参数版本。

### 2.3 从扰动传播理解 Dutch trace

传统 trace 对过去的部分只做 $\gamma\lambda$ 衰减；考虑刚才的传播矩阵后，可以把 Dutch trace 写成

$$
\mathbf z_t
=\mathbf x_t+\gamma\lambda\mathbf M_t\mathbf z_{t-1}.
$$

代入 $\mathbf M_t$ 并展开：

$$
\begin{aligned}
\mathbf z_t
&=\mathbf x_t+\gamma\lambda
\bigl(\mathbf I-\alpha\mathbf x_t\mathbf x_t^\top\bigr)\mathbf z_{t-1}\\
&=\gamma\lambda\mathbf z_{t-1}+\mathbf x_t
-\alpha\gamma\lambda
\bigl(\mathbf z_{t-1}^\top\mathbf x_t\bigr)\mathbf x_t.
\end{aligned}
$$

所以，Sutton & Barto 记号下的 **Dutch trace** 是

$$
\boxed{
\mathbf z_t
=\gamma\lambda\mathbf z_{t-1}
+\bigl(1-\alpha\gamma\lambda\mathbf z_{t-1}^\top\mathbf x_t\bigr)\mathbf x_t
}.
$$

其中 $\mathbf z_{t-1}^\top\mathbf x_t$ 表示过去资格与当前特征的**带符号重叠**：内积为正，说明大致同向；为零，说明正交；为负，则说明含有反向分量。**“避免重复记账”是直觉；在线更新如何传播历史影响，才是上面代数推导的来源。**

### 2.4 我的理解方式：把抽象公式还原成向量几何

> **我的理解方式：先看方向，再看系数。**
>
> 如果只逐项阅读 Dutch trace 的公式，我很容易把它记成“旧 trace 衰减、加入新 feature、再减去一个 correction”，却不容易理解为什么 correction 恰好是这个内积形式。我后来发现，把这些量画成参数空间中的向量会直观得多：先沿当前特征方向分解历史向量，再观察在线更新究竟改变哪一个分量。这样，内积、外积矩阵和修正项不再是三件分散的事情，而是在描述同一个几何过程。

这个视角并不是要用图代替推导，而是给代数式提供一个可以在脑中保持的图像。它还会立即排除一个常见误解：Dutch trace 并不是把整个历史向量统一缩小，而是只对**与当前特征方向重叠的部分**做额外调整。

因此，我不再先盯着公式中的三个加减项，而是先问：**过去留在参数空间中的向量，哪些部分能改变当前预测？**

当前预测是 $\mathbf w^\top\mathbf x_t$。假如某次历史学习留下参数扰动 $\Delta\mathbf w$，它对当前预测的影响恰好是：

$$
\Delta\hat q_t=\Delta\mathbf w^\top\mathbf x_t.
$$

只要 $\mathbf x_t\neq\mathbf0$，就可以沿 $\mathbf x_t$ 建立一个方向，将历史扰动拆成：

$$
\begin{aligned}
\Delta\mathbf w&=\Delta\mathbf w_{\parallel}+\Delta\mathbf w_{\perp},\\
\Delta\mathbf w_{\parallel}
&=\frac{\mathbf x_t\mathbf x_t^\top}{\|\mathbf x_t\|^2}\Delta\mathbf w,\\
\mathbf x_t^\top\Delta\mathbf w_{\perp}&=0.
\end{aligned}
$$

- $\Delta\mathbf w_{\parallel}$：沿当前特征方向的分量，**能够改变当前预测**。
- $\Delta\mathbf w_{\perp}$：与当前特征正交的分量，**不改变这一次预测**；但它可能影响其他状态—动作的预测，并不是无用信息。

![向量在一维子空间上的正交投影与正交残差]({{ '/assets/images/orthogonal-projection-one-dimensional-subspace.jpg' | relative_url }})
<p class="figure-caption">图 1：向量在一维子空间上的正交投影。左图的 $\pi_U(\mathbf x)$ 是沿基向量 $\mathbf b$ 的平行分量，红色虚线是与该子空间正交的残差；右图用 $\cos\omega$ 与 $\sin\omega$ 展示同一分解。图源：Deisenroth、Faisal 与 Ong，《Mathematics for Machine Learning》（2020），Figure 3.10 [5]。</p>

需要特别说明：教材图只提供标准的正交投影几何。下面把“投影分解、$\mathbf M_t$ 对平行分量的调整、$\gamma\lambda$ 衰减以及加入当前 $\mathbf x_t$”串成一个过程，**是我在推导 Dutch trace 时形成的理解，不是教材或论文中的原句。**

现在看之前出现的矩阵乘积：

$$
\begin{aligned}
\mathbf x_t\mathbf x_t^\top\Delta\mathbf w
&=\mathbf x_t(\mathbf x_t^\top\Delta\mathbf w)\\
&=\|\mathbf x_t\|^2\Delta\mathbf w_{\parallel}.
\end{aligned}
$$

这一步是几何解释的关键：**$\mathbf x_t\mathbf x_t^\top$ 会消去垂直分量，只留下平行分量，并乘上 $\|\mathbf x_t\|^2$。** 严格来说，真正的正交投影矩阵是 $\mathbf x_t\mathbf x_t^\top/\|\mathbf x_t\|^2$；不应直接把 $\mathbf x_t\mathbf x_t^\top$ 称作投影矩阵，除非 $\|\mathbf x_t\|=1$。[5]

把这个结果代回历史扰动的传播式：

$$
\begin{aligned}
\Delta\mathbf w_{\mathrm{new}}
&=(\mathbf I-\alpha\mathbf x_t\mathbf x_t^\top)\Delta\mathbf w\\
&=\underbrace{\Delta\mathbf w_{\perp}}_{\text{垂直部分不变}}
+\underbrace{(1-\alpha\|\mathbf x_t\|^2)\Delta\mathbf w_{\parallel}}_{\text{平行部分被重新调整}}.
\end{aligned}
$$

因此，$\mathbf M_t=\mathbf I-\alpha\mathbf x_t\mathbf x_t^\top$ 的几何意义不是“把整个历史向量缩小”：**它只调整沿当前特征方向的分量，垂直分量原样通过这一次更新。**

若 $0<\alpha\|\mathbf x_t\|^2<1$，平行部分会缩小；等于 $1$ 时会被消去；大于 $1$ 时系数为负，平行分量可能翻转方向。因而“压缩”只是常见步长范围内的直觉，不是无条件成立的性质。

### 2.5 把相同几何思想用在整个 Dutch trace 上

上面分解的是历史参数扰动 $\Delta\mathbf w$，但资格迹 $\mathbf z_{t-1}$ 也是一个位于**同一参数空间**的向量，因此可以用相同坐标系来读它。令

$$
\mathbf z_{t-1}=\mathbf z_{t-1,\parallel}+\mathbf z_{t-1,\perp},
$$

其中平行、垂直都**相对于当前的 $\mathbf x_t$** 定义，而不是相对于上一时刻的 $\mathbf x_{t-1}$。于是：

$$
\begin{aligned}
\mathbf z_t
&=\mathbf x_t+\gamma\lambda\mathbf M_t\mathbf z_{t-1}\\
&=\underbrace{\mathbf x_t}_{\text{加入当前资格}}
+\underbrace{\gamma\lambda(1-\alpha\|\mathbf x_t\|^2)
\mathbf z_{t-1,\parallel}}_{\text{历史资格的平行部分：时间衰减 + 在线更新调整}}\\
&\quad+\underbrace{\gamma\lambda\mathbf z_{t-1,\perp}}_{\text{历史资格的垂直部分：仅时间衰减}}.
\end{aligned}
$$

**这比“旧 trace + 新 feature − 一个修正项”更直观：** 当前特征方向上，历史资格已经参与改变当前预测，所以需要按在线更新的传播关系重新调整；与当前特征正交的历史资格，在这一步并未改变当前预测，因而不受该矩阵额外修正，仍只按 $\gamma\lambda$ 衰减。

还有一个容易忽略的细节：$\mathbf x_t$ 本身就在平行方向上。因此 Dutch trace 的**最终平行分量**是“当前新增 $\mathbf x_t$ + 被调整的历史平行分量”，并不是整个 $\mathbf z_t$ 都乘以 $1-\alpha\|\mathbf x_t\|^2$。例如 $\mathbf z_{t-1}$ 与 $\mathbf x_t$ 正交时，$\mathbf M_t\mathbf z_{t-1}=\mathbf z_{t-1}$，Dutch trace 与 accumulating trace 在这一步完全相同。

![从向量分解理解 Dutch trace 的完整递推]({{ '/assets/images/dutch-trace-vector-geometry.svg' | relative_url }})
<p class="figure-caption">图 2：先把历史资格分解为相对当前特征 $\mathbf x_t$ 的平行与垂直分量；$\mathbf M_t$ 只重新缩放平行分量，随后两部分共同乘以 $\gamma\lambda$，最后再加上当前特征。图中画的是 $0<1-\alpha\|\mathbf x_t\|^2<1$ 的常见情形；其他步长下平行分量也可能被消去或反向。这张图表达的是本文作者在整理公式时形成的向量理解，由 AI 辅助绘制，并依据本文推导与符号人工核对；正交投影的几何基础参考 [5]，Dutch trace 递推参考 [1]。</p>

对我来说，这个几何视角最大的价值，是把“整个向量被统一缩小”的模糊印象，替换成“历史向量相对当前特征方向分别传播”的具体图像。以后再看到 $\mathbf z_{t-1}^\top\mathbf x_t$，我会先把它读成“历史资格与当前方向有多少带符号重叠”，而不只是一个需要机械计算的内积。

### 2.6 两维手算：究竟削弱了向量的哪一部分？

设当前特征、上一步资格迹和参数分别为：

$$
\mathbf x_t=\begin{bmatrix}1\\0\end{bmatrix},\qquad
\mathbf z_{t-1}=\begin{bmatrix}0.6\\0.4\end{bmatrix},\qquad
\gamma\lambda=0.8,\qquad\alpha=0.5.
$$

当前只使用第一个参数方向。于是旧 trace 分成：

$$
\mathbf z_{t-1,\parallel}=\begin{bmatrix}0.6\\0\end{bmatrix},
\qquad
\mathbf z_{t-1,\perp}=\begin{bmatrix}0\\0.4\end{bmatrix}.
$$

由 $\|\mathbf x_t\|^2=1$，传播矩阵为

$$
\mathbf M_t
=\begin{bmatrix}0.5&0\\0&1\end{bmatrix}.
$$

所以，先穿过当前在线更新：

$$
\mathbf M_t\mathbf z_{t-1}=
\begin{bmatrix}0.3\\0.4\end{bmatrix}.
$$

再做时间衰减，最后加入当前特征：

$$
\boxed{
\mathbf z_t
=\underbrace{\begin{bmatrix}1\\0\end{bmatrix}}_{\text{新资格}}
+0.8\underbrace{\begin{bmatrix}0.3\\0.4\end{bmatrix}}_{\text{调整过的旧资格}}
=\begin{bmatrix}1.24\\0.32\end{bmatrix}
}.
$$

如果使用 accumulating trace，则得到

$$
\mathbf z_t^{\mathrm{acc}}
=\begin{bmatrix}1\\0\end{bmatrix}
+0.8\begin{bmatrix}0.6\\0.4\end{bmatrix}
=\begin{bmatrix}1.48\\0.32\end{bmatrix}.
$$

比较两者：**只有第一个坐标不同，第二个坐标同样是 $0.32$。** 这不是“Dutch trace 把所有历史 credit 打折”，而是它额外调整了**与本轮特征重叠的那一部分**。

此前的一维自检也可以保留：设 $z_{t-1}=x_t=1$、$\gamma\lambda=0.5$、$\alpha=0.5$，传统 trace 给出 $z_t=1.5$，Dutch trace 给出 $z_t=1.25$。一维例子只有平行方向，所以看不出“垂直部分保持不变”的区别；二维例子恰好补上这一点。

### 2.7 这个视角的适用边界

几何分解有助于理解，但有四个限定需要保留：

1. **正交是对“当前特征”而言。** 到下一时刻，特征方向变了，今天的垂直分量明天可能又有平行分量。
2. **内积有符号。** 若 $\mathbf z_{t-1}^\top\mathbf x_t<0$，Dutch correction 在当前特征方向的符号也会改变；Dutch trace 不保证每个坐标或整个向量范数都比 accumulating trace 小。
3. **矩阵调整与时间衰减不同。** “垂直分量不变”只针对 $\mathbf M_t$；完整 trace 仍会把垂直历史分量乘以 $\gamma\lambda$。$\mathbf x_t=\mathbf0$ 时，$\mathbf M_t=\mathbf I$，也就只剩时间衰减。
4. **这解释了 Dutch trace 的形式，不是完整等价性证明。** 单步更新的扰动传播是精确代数事实；True Online 与 online forward view 的精确等价还需要连同不断变化的 target、参数版本以及后文的 $Q_{\mathrm{old}}$ correction 一起处理。[1]

**我现在的理解是：Dutch trace 不是单纯给历史资格“减一点”，而是先问“历史向量的哪一部分影响了当前预测”，再只对这部分进行由在线更新决定的修正。** 这个向量视角把内积、矩阵 $\mathbf M_t$ 和 Dutch correction 三个看似不同的东西连到了一起。

---

## 3. $Q_{\mathrm{old}}$ correction：资格迹改了，为什么参数更新还需要修正？

Dutch trace 改的是**历史资格怎样传播**。还有一个不同的问题：在相邻时间步中，**同一个状态—动作对**可能已经因为参数更新而改变预测。

例如，时刻 $t-1$ 将即将到达的状态—动作对作为 next-Q 计算时，有

$$
Q_{\mathrm{old}}=0.50.
$$

上一时刻更新 $\mathbf w$ 后，来到当前 $(S_t,A_t)$，重新计算得到

$$
Q_t=0.69,
\qquad Q_t-Q_{\mathrm{old}}=0.19.
$$

这里比较的是**同一个 $(S_t,A_t)$、两套参数版本**，不是把当前 Q 与下一状态 Q 相减。$0.19$ 是参数更新造成的 prediction drift。

在线性 True Online Sarsa($\lambda$) 中，完整权重更新可写为

$$
\boxed{
\mathbf w_{t+1}
=\mathbf w_t+\alpha\delta_t\mathbf z_t
+\alpha(Q_t-Q_{\mathrm{old}})(\mathbf z_t-\mathbf x_t)
}.
$$

等价地，

$$
\mathbf w_{t+1}
=\mathbf w_t
+\alpha(\delta_t+Q_t-Q_{\mathrm{old}})\mathbf z_t
-\alpha(Q_t-Q_{\mathrm{old}})\mathbf x_t.
$$

这里的 TD error 为

$$
\delta_t=R_{t+1}
+\gamma\hat q(S_{t+1},A_{t+1},\mathbf w_t)
-\hat q(S_t,A_t,\mathbf w_t),
$$

终止状态的 bootstrap 项取 $0$。每次更新前，在旧参数 $\mathbf w_t$ 下先得到下一状态—动作对的预测，并在当前更新后将其保存为下一步使用的 $Q_{\mathrm{old}}$。

两项修正的分工可以压缩为：

- **Dutch trace：** 历史 credit 穿过当前在线更新时应如何累计。
- **$Q_{\mathrm{old}}$ correction：** 同一个当前预测在两次参数版本之间已经发生多少漂移，如何在更新式中正确补偿。

因此 True Online 的核心不是“让更新更保守”，而是在其线性设置与定义好的 online forward view 下，使高效 backward view 在每个时间点对应 forward view。[1]

![19-state Random Walk 上 online 与 offline lambda-return 的性能对比]({{ '/assets/images/online-lambda-return-comparison.jpg' | relative_url }})
<p class="figure-caption">图 3：19-state Random Walk 上 online λ-return 与 offline λ-return 的比较。左图的 online λ-return 在线性函数近似下可由 True Online TD(λ) 精确、高效地实现；即使评价指标取 episode 结束时的 RMS error，online 方法仍略占优势。图源：Richard S. Sutton、Andrew G. Barto，《Reinforcement Learning: An Introduction》（第 2 版），Figure 12.8。</p>

把 Dutch trace 与 $Q_{\mathrm{old}}$ correction 放回完整的动作价值控制循环，可以得到教材中的 True Online Sarsa($\lambda$) 伪代码：

![True Online Sarsa(lambda) 完整算法]({{ '/assets/images/true-online-sarsa-algorithm.jpg' | relative_url }})
<p class="figure-caption">图 4：True Online Sarsa(λ) 的完整算法框。资格迹更新中的内积修正对应 Dutch trace；权重更新中的 $Q-Q_{\mathrm{old}}$ 两项负责补偿同一预测跨参数版本的漂移。图源：Richard S. Sutton、Andrew G. Barto，《Reinforcement Learning: An Introduction》（第 2 版），第 307 页。</p>

---

## 4. 新问题：即使 credit 正确了，为什么学习率仍然要手动设定？

到这里，我们回答的是“**这一次误差应该怎样更新参数**”。但参数更新通常还要乘一个人为设定的全局步长 $\alpha$。不同特征的尺度、相关性以及环境变化程度可能不同，一个共享步长未必适合每一维。

Incremental Delta-Bar-Delta（**IDBD**）提出另一层问题：**不要只学 $w_i$，还要学每个 $w_i$ 对应的 step size。** 为了把两层问题分清，先离开 TD，回到最简单的监督式线性预测。[2]

令

$$
\hat y_t=\mathbf w_t^\top\mathbf x_t,
\qquad
\delta_t=y_t-\hat y_t.
$$

把普通更新里的全局 $\alpha$ 换成每个特征自己的 $\alpha_i$：

$$
\boxed{
w_{i,t+1}=w_{i,t}+\alpha_i\delta_tx_{i,t}
}.
$$

IDBD 使用指数参数化：

$$
\boxed{\alpha_i=e^{\beta_i}>0}.
$$

于是 $\beta_i$ 是第 $i$ 维学习率的对数参数。调高 $\beta_i$ 等于按比例调高 $\alpha_i$，而指数形式还保证 $\alpha_i$ 为正。为了减轻时间索引的拥挤，下面局部推导里的 $\beta_i,\alpha_i$ 指这一轮用于权重更新的值；稍后单独交代实际更新顺序。

---

## 5. IDBD 的核心：$h_i$ 记录什么？

如果直接问“学习率应不应该调大”，还没有足够的信息。我们需要知道：**如果以前用过稍大一点的学习率，那么现在的权重会有什么不同？**

这正是敏感度 $h_i$ 的含义：

$$
\boxed{
h_{i,t}\approx\frac{\partial w_{i,t}}{\partial\beta_i}
}.
$$

这里的 $\partial w/\partial\beta$ 是导数，**不是** $w/\beta$。由于学习率本身也会在线调整，实际 $h_i$ 是算法递推维护的局部近似敏感度，而非整个训练过程对某个永远固定的 $\beta_i$ 的精确总导数。

### 5.1 一个一维的反事实数值例子

令 $x_0=1$、$y_0=1$、$w_0=0$、$\alpha_0=0.1$，所以 $\beta_0=\ln 0.1$。第一次更新：

$$
w_1=w_0+e^{\beta_0}(1-w_0)=0.1.
$$

如果在这一次更新中将 $\beta_0$ 微微调大 $\varepsilon$，那么反事实结果会变成 $w_1'=0.1e^\varepsilon$。因此在 $\varepsilon\to0$ 时，

$$
\boxed{
h_1=\frac{\partial w_1}{\partial\beta_0}
=e^{\beta_0}=0.1
}.
$$

所以 $h_1>0$ 是说“这一轮的步长如果稍大，得到的 $w_1$ 就会稍大”。它**不是 overshoot 检测器**：单看 $h$ 无法判断学习率设得是否合适。

### 5.2 要把 $h_i$ 和当前误差放在一起看

$\delta_tx_{i,t}$ 是当前样本希望修改 $w_i$ 的方向；$h_{i,t}$ 是过去若把步长调大，现在的 $w_i$ 将怎样变化。两者相乘：

$$
\boxed{\delta_tx_{i,t}h_{i,t}}.
$$

若为正，过去使用更大步长所带来的改变与当前希望的更新方向一致；若为负，则相反。因此 IDBD 会依其符号调整学习率。**这只是局部适应信号**：一次反向更新可能源于噪声、目标变化或参数过冲，不能仅凭符号断言已经发生 overshoot。

---

## 6. IDBD：meta-gradient 和 $h$ 的递推从何而来？

### 6.1 什么叫 meta-gradient？

普通梯度下降调整 $w_i$，问“模型参数怎么改”；meta-gradient 调整控制学习过程的 $\beta_i$，问“学习率参数怎么改”。取监督学习平方损失：

$$
L_t=\frac12\delta_t^2,
\qquad \delta_t=y_t-\mathbf w_t^\top\mathbf x_t.
$$

优化 $\beta_i$：

$$
\beta_i\leftarrow\beta_i
-\theta\frac{\partial L_t}{\partial\beta_i},
$$

其中 $\theta$ 是 **meta-step-size**，控制步长本身调整的速度，不是模型权重的步长 $\alpha_i$。

链式法则得到

$$
\frac{\partial L_t}{\partial\beta_i}
=\delta_t\frac{\partial\delta_t}{\partial\beta_i}
=-\delta_t\sum_jx_{j,t}
\frac{\partial w_{j,t}}{\partial\beta_i}.
$$

最后的和式为什么没有直接变成 $-\delta_tx_{i,t}h_{i,t}$？因为虽然 $\beta_i$ 直接控制自己的步长 $\alpha_i$，其他权重也可能通过**共享的预测误差**间接受影响。完整追踪需要维护一个 $d\times d$ 的敏感度 Jacobian。

IDBD 采用**对角近似**，只保留自身的影响：

$$
\frac{\partial w_{j,t}}{\partial\beta_i}\approx0
\quad (j\ne i),
\qquad
\frac{\partial w_{i,t}}{\partial\beta_i}\approx h_{i,t}.
$$

于是

$$
\frac{\partial L_t}{\partial\beta_i}
\approx-\delta_tx_{i,t}h_{i,t},
$$

并得到 IDBD 的 meta-update：

$$
\boxed{
\beta_i\leftarrow
\beta_i+\theta\delta_tx_{i,t}h_{i,t}
}.
$$

注意：$-\delta_tx_ih_i$ 是近似 **meta-gradient**；更新式中 $+\theta\delta_tx_ih_i$ 来自对这个梯度做**下降**。

### 6.2 为什么 $h_i$ 要自己更新？

定义了 $h_i$ 以后，下一步还得维护它，因为 $w_i$ 每次都在变化。由

$$
w_{i,t+1}=w_{i,t}+e^{\beta_i}\delta_tx_{i,t}
$$

对 $\beta_i$ 求导。当前输入与监督目标视为固定，应用乘积法则：

$$
\begin{aligned}
h_{i,t+1}
&\approx h_{i,t}
+\alpha_i\delta_tx_{i,t}
+\alpha_ix_{i,t}\frac{\partial\delta_t}{\partial\beta_i}\\
&\approx h_{i,t}+\alpha_i\delta_tx_{i,t}
-\alpha_ix_{i,t}^2h_{i,t}\\
&=h_{i,t}(1-\alpha_ix_{i,t}^2)
+\alpha_i\delta_tx_{i,t}.
\end{aligned}
$$

算法采用正部截断的形式：

$$
\boxed{
h_{i,t+1}
=h_{i,t}[1-\alpha_ix_{i,t}^2]^+
+\alpha_i\delta_tx_{i,t}
},
\quad [u]^+\doteq\max(u,0).
$$

**$[\,]^+$ 不是求导推出来的**，它是更新规则另外加入的处理。你可以读成“旧敏感度经过本轮更新还保留多少 + 本轮又产生了多少新敏感度”。[2]

---

## 7. 从 IDBD 到 TIDBD(0)：把固定 target 换成 bootstrap target

前面是监督学习 $\delta_t=y_t-\hat y_t$。TIDBD 把独立步长适应移到 TD learning。这里需要**明确切换任务**：原始 TIDBD 的基本公式是状态价值预测，定义

$$
\hat v(S_t,\mathbf w_t)=\mathbf w_t^\top\mathbf x_t,
\qquad
\mathbf x_t\doteq\mathbf x(S_t).
$$

TD error 为

$$
\boxed{
\delta_t=R_{t+1}
+\gamma\mathbf w_t^\top\mathbf x_{t+1}
-\mathbf w_t^\top\mathbf x_t
}.
$$

和监督学习的区别是：$R_{t+1}+\gamma\hat v(S_{t+1},\mathbf w_t)$ 这个 **bootstrap target 本身依赖 $\mathbf w_t$**。

TIDBD 使用 semi-gradient：在本轮优化时把 bootstrap target 视为固定参考，不沿 target 一侧继续求导。因此，结合前面的对角近似，仍采用

$$
\frac{\partial\delta_t}{\partial\beta_i}
\approx-x_{i,t}h_{i,t},
\qquad
\boxed{\Delta\beta_i=\theta\delta_tx_{i,t}h_{i,t}}.
$$

这不是完整 TD-error 平方损失的真实全梯度，而是**semi-gradient meta-update**。它的更新形式接近 IDBD，但优化问题已经变成有 bootstrap target 的 TD prediction。[3]

当 $\lambda=0$ 时，$\mathbf z_t=\mathbf x_t$，因此

$$
w_{i,t+1}=w_{i,t}+\alpha_i\delta_tx_{i,t}.
$$

从**更新形式**上说，TIDBD(0) 可理解为“IDBD + TD error”；但两者的 target 定义、依赖关系及算法性质并不等同。

“每个 feature 有自己的 $\alpha_i$”不只是记号变化。TIDBD 原论文在 Mountain Car 预测实验中同时加入任务相关特征与随机噪声特征：相关特征的步长随学习明显增大，而无关特征的步长保持在较小水平。这给出了一个很直观的结果——独立步长适应也在进行一种简单的 representation learning。[3]

![TIDBD 为相关与无关特征学习到不同的步长]({{ '/assets/images/tidbd-relevant-irrelevant-feature-step-sizes.jpg' | relative_url }})
<p class="figure-caption">图 5：TIDBD 在 Mountain Car 预测任务中学到的 feature-specific step sizes。蓝线对应任务相关特征，其步长随 episode 增大；绿线对应随机无关特征，步长基本保持不变。图源：Kearney 等，<em>TIDBD: Adapting Temporal-difference Step-sizes Through Stochastic Meta-descent</em>（2018），Figure 3 [3]。</p>

---

## 8. TIDBD($\lambda$)：为什么权重用 $z_i$，步长却用 $x_ih_i$？

当 $\lambda>0$ 时，TIDBD 使用 accumulating eligibility trace：

$$
\boxed{z_{i,t}=\gamma\lambda z_{i,t-1}+x_{i,t}}.
$$

权重更新变成

$$
\boxed{w_{i,t+1}=w_{i,t}+\alpha_i\delta_tz_{i,t}}.
$$

这时三个量必须分开：

| 量 | 它回答的问题 |
|---|---|
| $x_{i,t}$ | 当前状态的预测依赖 $w_i$ 多少？ |
| $z_{i,t}$ | 这次 TD error 应该给过去的 $w_i$ 分配多少资格？ |
| $h_{i,t}$ | 过去稍微调整 $\beta_i$，今天的 $w_i$ 将怎样变化？ |

### 8.1 一个两状态的数值例子

设

$$
S_0\xrightarrow{R_1=0}S_1
\xrightarrow{R_2=1}\text{Terminal},
$$

取 $\gamma=1$、$\lambda=0.8$，使用 one-hot 特征：

$$
\mathbf x(S_0)=(1,0)^\top,
\qquad
\mathbf x(S_1)=(0,1)^\top.
$$

初始化 $\mathbf w=\mathbf0$、$\mathbf h=\mathbf0$、$\mathbf z=\mathbf0$，两个步长均为 $0.1$。第一步 TD error 为 $0$，留下 $\mathbf z_0=(1,0)^\top$。第二步得到奖励 $1$，因此

$$
\delta_1=1,
\qquad
\mathbf z_1
=0.8(1,0)^\top+(0,1)^\top
=(0.8,1)^\top.
$$

那么权重变化为

$$
\Delta w_1=0.1\times1\times0.8=0.08,
\qquad
\Delta w_2=0.1\times1\times1=0.1.
$$

虽然此时人在 $S_1$，当前 $x_{1,1}=0$，但前一状态留下的 $z_{1,1}=0.8$ 仍使 $w_1$ 获得更新。在本例的初始化下，$h_1$ 也新增了 $0.08$ 的敏感度；不过这一时刻 $\Delta\beta_1=\theta\delta_1x_{1,1}h_{1,1}=0$。如果以后再次访问 $S_0$，此前留在 $h_1$ 中的信息才可能进入 $\beta_1$ 的直接更新。

### 8.2 手推：为什么 $h$ 递推中的 $x_i^2$ 变成了 $x_iz_i$？

从真正的权重更新出发：

$$
w_{i,t+1}=w_{i,t}+e^{\beta_i}\delta_tz_{i,t}.
$$

普通 accumulating trace 由输入、$\gamma$、$\lambda$ 递推，本身不显式依赖步长，因此在当前推导中 $\partial z_{i,t}/\partial\beta_i=0$。对 $\beta_i$ 求导并利用乘积法则：

$$
\begin{aligned}
h_{i,t+1}
&\approx h_{i,t}
+\alpha_i\delta_tz_{i,t}
+\alpha_iz_{i,t}\frac{\partial\delta_t}{\partial\beta_i}\\
&\approx h_{i,t}
+\alpha_i\delta_tz_{i,t}
-\alpha_iz_{i,t}x_{i,t}h_{i,t}\\
&=h_{i,t}(1-\alpha_ix_{i,t}z_{i,t})
+\alpha_i\delta_tz_{i,t}.
\end{aligned}
$$

再加上算法的正部截断：

$$
\boxed{
h_{i,t+1}
=h_{i,t}[1-\alpha_ix_{i,t}z_{i,t}]^+
+\alpha_i\delta_tz_{i,t}
}.
$$

**两个因子的来源不能混淆：**

$$
\underbrace{x_{i,t}}_{\text{当前预测对 }w_i\text{ 的敏感度}}
\underbrace{z_{i,t}}_{\text{这次更新 }w_i\text{ 使用的资格}}.
$$

IDBD 中实际权重更新用 $x_i$，所以出现 $x_i^2$；TIDBD($\lambda$) 中更新权重改用 $z_i$，但当前预测对 $w_i$ 的导数仍是 $x_i$，因此**只替换了其中一个 $x_i$**。令 $\lambda=0$，便有 $z_i=x_i$，立即退回 IDBD 的形式。

### 8.3 为什么 $\beta_i$ 的直接更新不是 $\delta_tz_ih_i$？

因为两个更新正在回答不同的问题：

$$
\underbrace{\Delta w_i=\alpha_i\delta_tz_{i,t}}_{\text{把误差信息分配给过去参数}},
$$

$$
\underbrace{\Delta\beta_i=\theta\delta_tx_{i,t}h_{i,t}}_{\text{衡量当前 TD 预测误差对步长参数的敏感度}}.
$$

直接 meta-gradient 中的 $x_{i,t}$ 来自当前预测 $\mathbf w_t^\top\mathbf x_t$ 对 $w_i$ 的导数；$h_{i,t}$ 则是 $w_i$ 对 $\beta_i$ 的敏感度。$z_i$ 并没有和步长适应无关：它进入 $h_{i,t+1}$，然后**间接影响未来的** $\beta_i$ 更新。

### 8.4 实际算法中的更新顺序

以上推导为突出来源省略了部分临时下标。按 TIDBD 的在线执行顺序，可以这样读：先用当前 $\delta_t,x_{i,t},h_{i,t}$ 更新 $\beta_i$；得到本轮要使用的 $\alpha_i=e^{\beta_i}$；更新 $z_{i,t}$；随后用同一个本轮 $\alpha_i$ 更新 $w_i$ 和 $h_i$。不要把更新前后的 $\beta_i$ 混在一条等号里。[3]

这也说明 **TIDBD 同时存在两条相互耦合的 credit 路线**：$\delta_t\rightarrow\mathbf z_t\rightarrow\mathbf w$ 是 temporal credit；$\delta_t,\mathbf x_t,\mathbf h_t\rightarrow\boldsymbol\beta\rightarrow\boldsymbol\alpha$ 是步长适应；$\mathbf z_t\rightarrow\mathbf h_{t+1}$ 将两者联系起来。

---

## 9. 学完这一段，我修正了哪些理解？

1. **传统 accumulating trace 不是不会重算 TD error。** 它与 online forward view 的问题在于有限步长下整套历史 credit bookkeeping 不精确，而非某次误差被无故改成了另一个数。
2. **Dutch trace 和 $Q_{\mathrm{old}}$ correction 不是同一个修正。** 前者处理资格递推，后者处理同一预测跨参数版本的漂移。
3. **$h_i$ 不是判断是否 overshoot 的开关。** 它是过去学习率选择对当前权重的局部敏感度；只有结合当前更新方向才能构成步长适应信号，且信号本身存在噪声。
4. **$x_i$、$z_i$、$h_i$ 分别回答三个问题。** 当前预测使用谁、过去哪些参数有资格、过去学习率如何影响当前权重。它们不能相互替代。
5. **对角近似是工程取舍，不是独立性事实。** 它忽略经共享误差传递的跨特征敏感度，用线性规模的状态代替完整敏感度矩阵。
6. **几何视角不是另一套公式，而是一种阅读公式的方法。** 先确定当前特征定义的方向，再把历史向量分成平行与垂直分量，我就能直接看出内积测量什么、矩阵改变什么，以及为什么 correction 只作用在重叠方向上。这比单独记忆每个代数项更容易形成稳定的理解。

---

## 10. 总结与下一步留下的问题

上一篇的悬念已经得到回答：**历史 credit 不仅会随时间衰减，也会穿过不断发生的在线参数更新。** 从向量角度看，当前在线更新对历史资格的额外作用集中在与 $\mathbf x_t$ 平行的方向，垂直方向仍只按 $\gamma\lambda$ 衰减；Dutch trace 和 $Q_{\mathrm{old}}$ correction 分别处理资格传播与预测漂移，共同形成 True Online Sarsa($\lambda$)。

而 IDBD/TIDBD 打开另一层：不只让模型学到合适的权重，还用 meta-gradient 让每个 feature 的步长随数据适应。对 TIDBD($\lambda$)，必须区分当前表示 $\mathbf x$、时间资格 $\mathbf z$、步长敏感度 $\mathbf h$；尤其要理解为什么更新 $w_i$ 用 $\delta z_i$，但直接调整 $\beta_i$ 用 $\delta x_ih_i$。

接下来会遇到两个问题：**当 $\lambda>0$，一步 TD-error 式的 meta-objective 是否与 $\lambda$-return 学习目标匹配？如果自适应步长变得过大，如何控制 overshoot？** 这两条问题线会在 SwiftTD 中汇合。这里先停在问题本身，不在这一篇提前展开 SwiftTD 的全部变量和算法。[4]

---

## 参考文献

1. van Seijen, H., et al. *True Online Temporal-Difference Learning*. JMLR, 2016. <https://www.jmlr.org/papers/v17/15-599.html>.
2. Sutton, R. S. *Adapting Bias by Gradient Descent: An Incremental Version of Delta-Bar-Delta*. AAAI, 1992. <https://www.incompleteideas.net/papers/sutton-92a.pdf>.
3. Kearney, A., et al. *TIDBD: Adapting Temporal-difference Step-sizes Through Stochastic Meta-descent*. 2018. <https://arxiv.org/abs/1804.03334>.
4. Javed, K., Sharifnassab, A., & Sutton, R. S. *SwiftTD: A Fast and Robust Algorithm for Temporal Difference Learning*. 2024. <https://rlj.cs.umass.edu/2024/papers/RLJ_RLC_2024_111.pdf>.
5. Deisenroth, M. P., Faisal, A. A., & Ong, C. S. *Mathematics for Machine Learning*. Cambridge University Press, 2020, §3.8. <https://mml-book.github.io/book/mml-book.pdf>.
