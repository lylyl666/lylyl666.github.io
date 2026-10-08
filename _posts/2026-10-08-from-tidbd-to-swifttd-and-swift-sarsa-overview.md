---
layout: post
title: "从 TIDBD 到 SwiftTD 与 Swift-Sarsa：时间信用、步长学习与控制的统一视角"
date: 2026-10-08
categories: [Reinforcement Learning]
tags: [TD, Sarsa, true-online, IDBD, TIDBD, SwiftTD, Swift-Sarsa, meta-learning]
series: "强化学习基础"
math: true
---

> 上一篇[《从 True Online Sarsa(λ) 到 IDBD 与 TIDBD》]({{ '/posts/from-true-online-to-idbd-tidbd/' | relative_url }})讨论了两件不同的事：**True Online 怎样处理在线更新中的历史信用与预测版本，IDBD/TIDBD 怎样学习每个特征的步长。**
>
> 这一篇继续追问：当这两套机制放进同一个学习器时，**哪些量也必须跟着改变**？为什么已经有了 $h$，还会出现 $p$ 和 $\bar z$？学出来的步长过大时怎么办？最后，这套预测机制又怎样**进入动作选择**？

我希望通过这篇文章建立的，不是一份算法名词表，而是一种阅读在线学习算法的方法：先明确它在优化什么，再检查误差如何传播、需要保存哪些状态，以及这些状态最终进入哪一个更新。

文末还将从表格 Sarsa 开始回顾这一系列的完整演进，区分各算法的新增机制与相互依赖。本文的算法依据是 TIDBD、SwiftTD 与 Swift-Sarsa 原论文。推导按这一系列的符号重新组织；数值例子用于解释公式，不是复现实验。文末的模块化理解和迁移问题属于整理与总结。

## 1. 从前面的三篇文章走到这里

这个系列到目前为止，可以沿着三个问题回看。

[第一篇]({{ '/posts/from-sarsa-to-eligibility-traces/' | relative_url }})从轨迹回报、价值函数和 Sarsa 出发，走到多步回报与资格迹：**一次较晚出现的反馈，如何影响较早做出的预测？**

[第二篇]({{ '/posts/from-tabular-sarsa-to-true-online-sarsa/' | relative_url }})加入线性函数近似，并发现新的问题：多个状态—动作对共享参数，在线更新会不断改变后续预测；传统 accumulating trace 不会因此自动与 online forward view 精确一致。

[第三篇]({{ '/posts/from-true-online-to-idbd-tidbd/' | relative_url }})用 Dutch trace 和旧预测修正解释 True Online，再引入 IDBD/TIDBD：**即使知道参数怎样更新，为什么仍让所有特征使用相同的学习速度？**

现在，SwiftTD 把 True Online 与逐特征步长优化结合起来，再加入步长保护；Swift-Sarsa 将这套机制用于离散动作控制。[1–4]

[![SwiftTD 与 True Online TD 在四个 Atari 任务中的预测对比]({{ '/assets/images/swifttd-atari-predictions.png' | relative_url }})]({{ '/assets/images/swifttd-atari-predictions.png' | relative_url }})
<p class="figure-caption">图 1：学习约两小时后，SwiftTD（左）与 True Online TD($\lambda$)（右）在四个 Atari 任务中的预测片段，灰色虚线为实际回报。这是固定策略采样下的预测实验，评价的是价值预测的准确性。图源：Khurram Javed、Arsalan Sharifnassab、Richard S. Sutton，<a href="https://rlj.cs.umass.edu/2024/papers/RLJ_RLC_2024_111.pdf#page=2">SwiftTD（2024），Figure 1</a> [2]。由论文原图裁切，点击图片可查看大图。</p>

这里最值得保留的区分是：

| 问题 | 对应的机制 | 不应混为一谈的东西 |
|---|---|---|
| 用什么表示价值？ | 表格或函数近似 | 改变表示不等于改变 TD target |
| 当前误差向过去传播多远、怎样传播？ | $\lambda$-return、资格迹、True Online | 历史信用不等于学习率 |
| 每个参数学多快？ | IDBD/TIDBD、Swift 的步长优化 | 学习率敏感度不等于预测误差 |
| 步长组合过大时怎样约束？ | bound、decay | 局部约束不等于全局收敛保证 |
| 预测怎样参与行为选择？ | 动作价值、Sarsa target、策略 | prediction 与 control 不是同一个任务 |

**这些是不同的设计维度，不是一条算法必然取代另一条算法的排行榜。** 接下来先沿预测主线说明 SwiftTD，最后再进入控制。

## 2. 先把时间、坐标和资格迹的约定对齐

### 2.1 时间与坐标

沿用前文的 $S_t,A_t,R_{t+1}$、$\mathbf w_t,\mathbf x_t,\mathbf z_t$。分量统一写成 $w_{i,t},x_{i,t},h_{i,t}$：$t$ 是**时间**，$i$ 是**特征/参数坐标**，动作索引单独写作 $a$。 汇报笔记中写作 $x_{t,i}$ 的量，在这里统一为 $x_{i,t}$。

预测部分令 $\mathbf x_t=\mathbf x(S_t)$，并约定 $\mathbf w_t$ 是处理转移 $S_t\to S_{t+1}$ 前的权重：

$$
\hat v_t=\hat v(S_t,\mathbf w_t)=\mathbf w_t^\top\mathbf x_t,
\qquad
\delta_t=R_{t+1}+\gamma\hat v(S_{t+1},\mathbf w_t)-\hat v_t.
$$

对于固定目标的导数推导，输入特征视为固定；涉及 TD 目标时会明确采用半梯度近似，即暂不对 bootstrap target 求导。终止状态的 bootstrap 值取 $0$。

### 2.2 一个不能省略的转换：步长放在资格迹外面，还是里面？

前文普通 TD($\lambda$)/TIDBD 的资格迹写成

$$
\mathbf z_t^{\mathrm{acc}}
=\gamma\lambda\mathbf z_{t-1}^{\mathrm{acc}}+\mathbf x_t,
\qquad
\Delta w_{i,t}=\alpha_i\delta_tz_{i,t}^{\mathrm{acc}}.
$$

其中，步长在权重更新式外面。SwiftTD 则把步长纳入资格迹的递推，因而权重更新里不再额外乘同一个 $\alpha_i$。

先用**固定标量步长**说明这种记号转换。将前文未吸收外部步长的 Dutch trace 暂记为 $\mathbf z_t^{\mathrm{std}}$，定义

$$
\mathbf z_t=\alpha\mathbf z_t^{\mathrm{std}}.
$$

前文公式

$$
\mathbf z_t^{\mathrm{std}}
=\gamma\lambda\mathbf z_{t-1}^{\mathrm{std}}
+\left(1-\alpha\gamma\lambda
(\mathbf z_{t-1}^{\mathrm{std}})^\top\mathbf x_t\right)\mathbf x_t
$$

就变成

$$
\boxed{
\mathbf z_t
=\gamma\lambda\mathbf z_{t-1}
+\alpha\mathbf x_t
\left(1-\gamma\lambda\mathbf z_{t-1}^\top\mathbf x_t\right).
}
$$

从这里开始，除非显式标注 $\mathrm{acc}$ 或 $\mathrm{std}$，$\mathbf z$ 都采用 Swift 的**“步长纳入递推”约定**。 这解释了为什么后面出现 $\delta_tz_{i,t}$，却没有少乘一个学习率。

当步长随时间、特征变化时，不能把完整历史迹简单理解成“普通迹乘当前步长”；必须按实际递推保留每次注入时的步长影响。

## 3. 先明确 TIDBD 留给我们的是什么

原始 TIDBD($\lambda$) 使用普通 accumulating trace，并维护

$$
\alpha_i=e^{\beta_i},
\qquad
h_i\approx\frac{\partial w_i}{\partial\beta_i}.
$$

其核心关系是

$$
\begin{aligned}
\Delta w_i&=\alpha_i\delta_tz_{i,t}^{\mathrm{acc}},\\
\Delta\beta_i&=\theta\delta_tx_{i,t}h_{i,t},\\
h_i^+&=h_i[1-\alpha_ix_{i,t}z_{i,t}^{\mathrm{acc}}]^+
+\alpha_i\delta_tz_{i,t}^{\mathrm{acc}}.
\end{aligned}
$$

这里省略了临时变量的覆盖顺序；$[u]^+=\max(u,0)$ 是算法加入的正部截断，不是链式法则本身推出来的项。[1]

原始 TIDBD **并没有** SwiftTD 这里单独维护的 $p$。 它的权重更新使用 $z$，直接步长更新使用 $xh$；$z$ 又通过 $h$ 的递推影响后续步长。因此不能说 TIDBD 的步长学习“完全没有历史”，也不能把 SwiftTD 附录中的“TD($\lambda$) with step-size optimization”直接当成原始 TIDBD。

从这里走向 SwiftTD，需要重新处理三件事：让步长优化利用 $\lambda$-return 对应的时间信息；对包含 Dutch trace 和旧预测修正的更新追踪敏感度；限制过大的新资格迹注入。[2]

下面先解释前两件事中的特殊变量。

## 4. $h$：它追踪的是学习过程，不是当前权重的大小

### 4.1 从“改变步长会发生什么”开始

$\beta_i$ 是对数步长参数。一个小扰动 $\varepsilon$ 对应

$$
\beta_i\to\beta_i+\varepsilon
\quad\Longrightarrow\quad
\alpha_i\to\alpha_ie^\varepsilon
\approx\alpha_i(1+\varepsilon).
$$

它改变的是学习速度，而不是直接给 $w_i$ 加上 $\varepsilon$。如果这个微小改变作用于学习过程，累积得到的权重也会不同。$h$ 用来近似追踪这种影响：

$$
\boxed{h_{i,t}\approx\frac{\partial w_{i,t}}{\partial\beta_i}.}
$$

**这里的导数简写指向学习历史。** 只在程序里修改当前 $\beta_i$，已经保存的 $w_i$ 不会立刻变动；$h$ 也不是对两个彼此独立的当前内存变量做一次求导。它是在递推中估计：若相关步长选择略有不同，现在的权重会受到怎样的影响。

完整敏感度包含跨坐标项 $\partial w_{j,t}/\partial\beta_i$。IDBD/Swift 的对角近似主要保留 $j=i$，避免维护完整的敏感度矩阵。这是计算上的取舍，**不是特征真的彼此独立**。[1,2]

### 4.2 为什么 $h$ 必须自己递推？

先用一维监督预测解释，暂时不加入 TD 和 True Online：

$$
w^+=w+\alpha e x,
\qquad e=y-wx,
\qquad\alpha=e^\beta.
$$

固定 $x,y$，对学习过程中的 $\beta$ 求导：

$$
\begin{aligned}
h^+
&=h+\frac{\partial(\alpha e x)}{\partial\beta}\\
&=h+\alpha ex+\alpha x\frac{\partial e}{\partial\beta}\\
&=h+\alpha ex-\alpha x^2h\\
&=h(1-\alpha x^2)+\alpha ex.
\end{aligned}
$$

第一部分是历史敏感度经过本轮更新后留下的部分，第二部分是本轮学习新产生的敏感度。所以 $h$ 本身已经是**一种递归记忆**。 后面不能用“$h$ 没有历史，$p$ 才有历史”来区分两者。

以 $w_0=0,x=y=1,\alpha=0.1$ 为例：第一步得到 $w_1=0.1$、$h_1=0.1$；第二步得到 $w_2=0.19$，以及

$$
h_2=0.1(1-0.1)+0.1(1-0.1)=0.18.
$$

对这个固定步长的两步例子，$w_2=2\alpha-\alpha^2$，直接求 $\partial w_2/\partial\beta=\alpha(2-2\alpha)=0.18$，可以核对递推。

### 4.3 为什么真正进入预测损失的是 $xh$？

当前预测为 $\hat v_t=\sum_jw_{j,t}x_{j,t}$，因此完整链式法则是

$$
\frac{\partial\hat v_t}{\partial\beta_i}
=\sum_jx_{j,t}\frac{\partial w_{j,t}}{\partial\beta_i}
\approx x_{i,t}h_{i,t}.
$$

<span style="color:#d1242f">$h$<strong> 是权重敏感度，</strong>$xh$<strong> 才是当前预测对步长参数的敏感度。</strong></span> $x_{i,t}h_{i,t}$ 是两个标量相乘；对整个向量而言，它对应逐元素乘积，不是把所有坐标相加成一个内积。

例如 $h_i=0.1$，当前 $x_i=2$，则预测敏感度为 $0.2$。步长参数微增 $\varepsilon=0.01$，对应的局部预测差约为 $0.002$。如果当前 $x_i=0$，即使 $h_i\ne0$，该坐标也不直接影响这次预测。

但敏感度仍不等于好坏判断。固定目标的平方损失满足

$$
-\frac{\partial\mathcal L_t}{\partial\beta_i}
\approx (\text{目标}-\text{预测})\,x_{i,t}h_{i,t}.
$$

只有将误差与敏感度结合，才得到步长应该怎样调整的局部信号。单看 $h_i$ 的正负，不能判断是否过冲，也不能判断特征是否重要。

## 5. $p$：为什么已经有历史敏感度 $h$，还要再维护一个迹？

### 5.1 两种不同的“历史”

$h_{i,t}$ 汇总的是：过去的学习怎样形成了**现在这个权重对步长参数的敏感度**。

但当步长优化的目标涉及多步回报时，还需要另一类信息：**过去每一次预测对步长有多敏感，这些预测现在还应不应该接收新到来的误差？**

前者是学习过程的导数状态，后者是预测敏感度的时间信用分配。当前的一个 $h_{i,t}$，并不能代替过去各个状态上的 $x_{i,s}h_{i,s}$。例如某个权重现在仍有敏感度，但当前特征不使用它；只看当前 $xh$，就看不到此前状态在这个方向上的预测敏感度。

### 5.2 从多步目标到时间换序

取一个平方损失，目标一侧按半梯度处理：

$$
\mathcal L_s=\frac12(G_s^\lambda-\hat v_s)^2.
$$

令 $g_{i,s}$ 表示**该次预测被记录时**，预测对 $\beta_i$ 的局部敏感度，即相应参数版本下的 $xh$。于是

$$
-\frac{\partial\mathcal L_s}{\partial\beta_i}
\approx(G_s^\lambda-\hat v_s)g_{i,s}.
$$

为单独看清时间结构，先冻结同一个预测函数。在有限轨迹、终点取零的设置下，有熟悉的 TD-error 展开：

$$
G_s^\lambda-\hat v_s
=\sum_{u=s}^{N}(\gamma\lambda)^{u-s}\delta_u.
$$

**这个等式的冻结预测约定不能省略。** 在线权重持续变化时，不能直接把任意版本的 $\delta_u$ 塞进来，便声称得到了精确的 online forward view。下面先做时间求和的代数换序，之后再说明实现里的版本对应。

固定某个坐标 $i$，看三次预测留下的敏感度 $g_{i,0},g_{i,1},g_{i,2}$：

$$
\begin{aligned}
&g_{i,0}\bigl[\delta_0+\gamma\lambda\delta_1+(\gamma\lambda)^2\delta_2\bigr]\\
&\quad+g_{i,1}\bigl[\delta_1+\gamma\lambda\delta_2\bigr]
+g_{i,2}\delta_2.
\end{aligned}
$$

从每次预测的角度读，这是“这一份敏感度将来接收哪些误差”。改为按误差到来的时刻分组：

$$
\begin{aligned}
&\delta_0[g_{i,0}]\\
&\quad+\delta_1[\gamma\lambda g_{i,0}+g_{i,1}]\\
&\quad+\delta_2[(\gamma\lambda)^2g_{i,0}
+\gamma\lambda g_{i,1}+g_{i,2}].
\end{aligned}
$$

括号内就是需要保留的量。因此定义

$$
\boxed{
p_{i,t}=\sum_{s=0}^{t}(\gamma\lambda)^{t-s}g_{i,s}
=\gamma\lambda p_{i,t-1}+g_{i,t}.
}
$$

<span style="color:#d1242f">$p$<strong> 保存的是过去预测敏感度的加权和；当前误差到来以后，才与 </strong>$p$<strong> 相乘。</strong></span> 这就是从 forward view 转向 backward view 的核心。[2，附录 A.1]

### 5.3 $p$ 不是 TD error，也不是未来误差的预测

下面两件事必须分开：

$$
\underbrace{p_{i,t}}_{\text{保留下来的历史预测敏感度}}
\qquad\text{与}\qquad
\underbrace{\delta_t}_{\text{当前转移产生的误差信号}}.
$$

$p$ 的新增对象是 $xh$，不是 reward 或 $\delta$；它也不会预测未来误差有多大。不过，$h$ 由过去的学习更新形成，因此 $p$ 会**间接受历史误差影响**，不能说它与 TD error 完全无关。

另一个容易误解的地方是：$p$ 不是把过去每个 $\beta_{i,s}$ 单独存起来，事后逐个修改。它把相关影响压缩到一个递推量里，并据此调整**现在可用的步长参数**。

### 5.4 接回 SwiftTD：$p$ 最终进入哪一行？

SwiftTD Algorithm 1 的直接步长更新是

$$
\boxed{
\beta_i\leftarrow\beta_i+
\frac{\theta}{e^{\beta_i}}(\delta'-v^\delta)p_i.
}
$$

这里保留了一行论文的缓存记号：$\delta'$ 使用之前保存的预测基准，$v^\delta$ 记录相应的预测漂移。后文将它们分别对应为 $\tilde\delta_t$ 和 $d_t$；在这一对应下，$\delta'-v^\delta=\delta_t$。$\theta$ 控制元参数更新速度，$e^{-\beta_i}$ 是论文采用的元步长归一化。[2，Algorithm 1]

因而最直接的作用链是

$$
\boxed{
p_i\longrightarrow\Delta\beta_i
\longrightarrow\alpha_i=e^{\beta_i}
\longrightarrow\text{后续资格迹与权重更新}.
}
$$

$p$ 不直接出现在 $\Delta w$ 中，但它通过**改变后续步长**影响 $w$。 若某次 $p_i=0$，这条元梯度更新为零；若有历史 $p_i\ne0$，当前 $x_i=0$ 也不必使这条信号消失。论文的稀疏实现还会按其活动资格迹集合执行更新，不能只看孤立的一行就忽略外层循环。

例如，某次已经形成 $g_{i,0}=0.2$，下一次这个坐标没有新的预测敏感度，取 $\gamma\lambda=0.8$，则 $p_{i,1}=0.16$。若随后用于元更新的误差为 $0.5$，$\theta=0.001$、当前 $\alpha_i=0.1$，那么

$$
\Delta\beta_i=\frac{0.001}{0.1}\times0.5\times0.16=0.0008.
$$

忽略 clipping 和 decay，步长变成 $0.1e^{0.0008}\approx0.100080$。历史敏感度让这次误差仍可调整步长，但不会凭空生成一份当前特征激活。

### 5.5 $xh$ 中的 $h$ 到底是哪一个时间版本？

前面的 $g$ 是为了先分离“时间累积”和“缓存版本”。实现时必须把二者重新对齐。

Swift 的接收—更新循环会在上一轮更新前，对新到达的状态 $S_t$ 计算并保存预测。按本文的时间约定，该保存值使用 $\mathbf w_{t-1}$，所以相应记录的预测敏感度是

$$
g_{i,t}\approx x_{i,t}h_{i,t-1},
\qquad
p_{i,t}=\gamma\lambda p_{i,t-1}+x_{i,t}h_{i,t-1}.
$$

这不与第 4 节 $\partial\hat v(S_t,\mathbf w_t)/\partial\beta_i\approx x_{i,t}h_{i,t}$ 矛盾：**两个式子评价的是不同参数版本的预测。**

SwiftTD 伪代码先轮换 $h$ 的缓存，再执行 `p[i] += x[i] * h[i]`；Swift-Sarsa 的缓存组织不同，对应新增项写为 `x[i] * h_old[i]`。不能脱离前面的赋值顺序，把程序变量名直接当成数学时间下标。[2,4]

## 6. $\bar z$：学习率也改变资格迹，求导时不能跳过这条路径

### 6.1 先看它最终在哪里被使用

从简化的权重更新出发：

$$
w_i^+=w_i+\delta z_i.
$$

对步长参数求导时，乘积法则给出

$$
h_i^+
=h_i+z_i\frac{\partial\delta}{\partial\beta_i}
+\delta\frac{\partial z_i}{\partial\beta_i}.
$$

因此引入

$$
\boxed{\bar z_i\approx\frac{\partial z_i}{\partial\beta_i}.}
$$

它最终进入 $h$ 的递推，而 $h$ 又生成新的预测敏感度、进入 $p$，最终影响步长。

<span style="color:#d1242f"><strong>计算普通资格迹本身不需要 </strong>$\bar z$<strong>；对“使用该资格迹的学习过程”求元梯度，才需要它。</strong></span> 这是 $\bar z$ 与 $p$ 最本质的区别：前者补齐链式法则中的路径，后者累积不同时间的预测敏感度。

### 6.2 为什么普通 TIDBD 没有同样的问题？

原始 TIDBD 的 accumulating trace 不含步长：

$$
z_{i,t}^{\mathrm{acc}}=\gamma\lambda z_{i,t-1}^{\mathrm{acc}}+x_{i,t}.
$$

在固定输入轨迹、零初始化的推导中，$\partial z_{i,t}^{\mathrm{acc}}/\partial\beta_i=0$。但 Swift 的资格迹递推吸收了步长，Dutch trace 又含依赖旧迹的修正，所以不能照搬这个零导数结论。[1,2]

### 6.3 $1-T$ 与前文的 Dutch trace 矩阵有什么关系？

先不启用 bound。定义衰减后的旧迹

$$
\mathbf z_t^-=\gamma\lambda\mathbf z_{t-1},
\qquad T_t=(\mathbf z_t^-)^\top\mathbf x_t.
$$

一次注入阶段的步长记为 $\alpha_i$，则

$$
\boxed{z_{i,t}=z_{i,t}^-+\alpha_ix_{i,t}(1-T_t).}
$$

这里 $T_t$ 和 $1-T_t$ 都是**标量**，不是矩阵。令 $\mathbf D=\operatorname{diag}(\alpha_1,\ldots,\alpha_n)$，把它展开为向量式：

$$
\begin{aligned}
\mathbf z_t
&=\mathbf z_t^-+\mathbf D\mathbf x_t
-\mathbf D\mathbf x_t\mathbf x_t^\top\mathbf z_t^-\\
&=\mathbf D\mathbf x_t+
\gamma\lambda\bigl(\mathbf I-\mathbf D\mathbf x_t\mathbf x_t^\top\bigr)\mathbf z_{t-1}.
\end{aligned}
$$

这就接回了上一篇的传播矩阵 $\mathbf M_t$：固定标量步长时，矩阵部分为 $\mathbf I-\alpha\mathbf x_t\mathbf x_t^\top$。$1-T_t$ 是这段矩阵乘法展开后出现的标量因子，不是矩阵本身。

**“避免重复加入历史信用”只是直觉解释。** $T_t$ 是带符号的内积，不保证处于 $[0,1]$；因此 $1-T_t$ 也不保证总是一个介于零和一之间的缩放系数。

### 6.4 对整个 Dutch trace 求导

相应地，令 $\bar z_{i,t}^-=\gamma\lambda\bar z_{i,t-1}$。对

$$
z_{i,t}=z_{i,t}^-+e^{\beta_i}x_{i,t}(1-T_t)
$$

求导，有

$$
\bar z_{i,t}
\approx\bar z_{i,t}^-
+\alpha_ix_{i,t}(1-T_t)
-\alpha_ix_{i,t}\frac{\partial T_t}{\partial\beta_i}.
$$

由于 $T_t=\sum_jz_{j,t}^-x_{j,t}$，在对角近似下

$$
\frac{\partial T_t}{\partial\beta_i}
\approx x_{i,t}\bar z_{i,t}^-.
$$

于是得到

$$
\boxed{
\bar z_{i,t}\approx\bar z_{i,t}^-
+\alpha_ix_{i,t}
\left(1-T_t-x_{i,t}\bar z_{i,t}^-\right).
}
$$

新增的最后一项来自“步长改变旧资格迹，旧资格迹又改变 overlap”这条路径。固定输入和求导参数、无 bound 时，这就是 SwiftTD 附录 A.2 对 trace sensitivity 的核心推导。[2]

一维检查：取 $\gamma\lambda=0.8$、$\alpha=0.1$、两步特征都为 $1$，从零迹开始。第一步 $z_0=\bar z_0=0.1$；第二步

$$
z_1=0.08+0.1(1-0.08)=0.172,
$$

$$
\bar z_1=0.08+0.1(1-0.08-0.08)=0.164.
$$

**资格迹的数值和资格迹对步长的导数，不是同一个量。** 第一轮看起来相同，后续就会因为 Dutch overlap 的依赖关系而分开。

## 7. $d_t$ 与权重更新：True Online 为什么还要记录旧预测？

### 7.1 同一个状态，两个参数版本

在时刻 $t-1$ 的转移中，$S_t$ 已经被观察并用于 bootstrap；随后完成更新 $\mathbf w_{t-1}\to\mathbf w_t$。因此同一个 $S_t$ 有更新前、后的两个预测。定义

$$
\boxed{
d_t=\hat v(S_t,\mathbf w_t)-\hat v(S_t,\mathbf w_{t-1})
=\mathbf x_t^\top(\mathbf w_t-\mathbf w_{t-1}).
}
$$

$d_t$ 不是状态之间的价值差，也不是新奖励，而是**上一次学习对当前状态预测造成的变化**。 对分幕任务的首步，需要使用初始化约定，而不是实际访问一个不存在的 $\mathbf w_{-1}$。

### 7.2 标准误差与旧预测基准下的误差

令本轮 target 为 $Y_t=R_{t+1}+\gamma\hat v(S_{t+1},\mathbf w_t)$，则

$$
\begin{aligned}
\delta_t&=Y_t-\hat v(S_t,\mathbf w_t),\\
\tilde\delta_t&=Y_t-\hat v(S_t,\mathbf w_{t-1}),\\
\tilde\delta_t-\delta_t&=d_t.
\end{aligned}
$$

所以

$$
\boxed{\tilde\delta_t=\delta_t+d_t.}
$$

论文缓存中的 $\delta'$ 对应这里的 $\tilde\delta_t$，$v^\delta$ 对应 $d_t$。由此也就能理解第 5 节的元更新为何使用 $\delta'-v^\delta$。

### 7.3 为什么更新里既加漂移、又减漂移？

前文固定步长、未吸收外部步长的 True Online 更新是

$$
\Delta\mathbf w_t
=\alpha(\delta_t+d_t)\mathbf z_t^{\mathrm{std}}
-\alpha d_t\mathbf x_t.
$$

换成第 2 节的资格迹约定，暂时无 bound、令 $z_{i,t}^{\delta}=\alpha_ix_{i,t}$，得到

$$
\boxed{
\Delta w_{i,t}=\tilde\delta_tz_{i,t}-d_tz_{i,t}^{\delta}.
}
$$

展开为

$$
\boxed{
\Delta w_{i,t}
=\underbrace{\delta_tz_{i,t}}_{\text{TD 学习}}
+\underbrace{d_t(z_{i,t}-z_{i,t}^{\delta})}_{\text{True Online 修正}}.
}
$$

这个更新结构来自 True Online 的 forward/backward 转换，不是根据“有漂移”临时猜出来的修正。[2,3] 代数上，先将误差换到旧预测基准，得到 $+d_tz_{i,t}$；再减去与当前基础注入对应的 $d_tz_{i,t}^{\delta}$。剩下的不是对所有资格一视同仁的误差放大，而是对相应历史传播结构的修正。

尤其要注意：$z^\delta$ 是**乘上 Dutch 因子之前的基础注入**，实际加进旧迹的是 $z^\delta(1-T_t)$。因此

$$
z_{i,t}-z_{i,t}^{\delta}
=z_{i,t}^- -z_{i,t}^{\delta}T_t,
$$

不等于简单的“旧迹”。这部分同时包含历史传播和 overlap 调整。

例如 $\delta_t=0.5,d_t=0.2,z_i=0.4,z_i^\delta=0.1$，两种写法分别给出

$$
0.7\times0.4-0.2\times0.1
=0.5\times0.4+0.2(0.4-0.1)=0.26.
$$

### 7.4 为什么这里又会影响 $h$？

因为现在要求的是整个更新后的权重对 $\beta_i$ 的敏感度。仅对修正项求导，就会遇到

$$
\frac{\partial(d_tz_{i,t}^{\delta})}{\partial\beta_i}
=z_{i,t}^{\delta}\frac{\partial d_t}{\partial\beta_i}
+d_t\frac{\partial z_{i,t}^{\delta}}{\partial\beta_i}.
$$

而在线性、对角近似下

$$
\boxed{
\frac{\partial d_t}{\partial\beta_i}
\approx x_{i,t}(h_{i,t}-h_{i,t-1}).
}
$$

于是相邻两版 $h$ 都必须保留。$h^{\mathrm{old}},h,h^{\mathrm{temp}}$ 是版本与部分计算结果的缓存，不是三种不同学习目标。附录 A 把这几项合并成完整的无 bound 敏感度递推。

## 8. Swift 的稳定性模块：学步长，不等于步长永远合适

### 8.1 $\tau$ 从哪里来？

先看固定监督目标 $y$ 的一次线性更新。误差 $e=y-\mathbf w^\top\mathbf x$，逐特征更新为 $\Delta w_i=\alpha_i e x_i$，则

$$
\Delta\hat v=\sum_i\Delta w_ix_i
=e\sum_i\alpha_ix_i^2.
$$

定义

$$
\boxed{\tau=\sum_i\alpha_ix_i^2,}
\qquad e_{\mathrm{new}}=(1-\tau)e.
$$

在这个固定目标例子里，$0<\tau<1$ 是向目标靠近而不越过，$\tau=1$ 恰好到达，$\tau>1$ 会越过目标。它解释了为什么仅检查单个 $\alpha_i$ 不够：**共同激活的特征越多，整体修正可能越大。**

但完整 TD 还涉及移动的 bootstrap、历史迹和 prediction drift，不能把这个单次回归等式直接升级成“所有 TD error 都必然单调下降”。

### 8.2 Bound 限制的是哪一个量？

在登记一次新特征的阶段，使用该阶段已计算出的步长 $\alpha_i$：

$$
c=\min\left(1,\frac{\eta}{\tau}\right),
\qquad
\boxed{z_i^{\delta}=c\alpha_ix_i.}
$$

当 $\tau=0$ 时可令 $c=1$，因为此时注入为零，不应执行除零。随后

$$
z_i\leftarrow z_i^-+z_i^{\delta}(1-T).
$$

因此被直接限制的是

$$
\sum_i z_i^{\delta}x_i=c\tau=\min(\tau,\eta),
$$

<span style="color:#d1242f"><strong>也就是这次基础注入沿当前预测方向的强度。</strong></span>它不是把整个历史 $\mathbf z$ 清零，不是对最后的 $\Delta\mathbf w$ 做统一裁剪，也不是只修改元梯度。 这份受限注入还会留在后续的资格迹中。[2]

### 8.3 Decay 改的是未来步长

每当本轮 $\tau>\eta$，算法还执行

$$
\beta_i\leftarrow\beta_i+x_i^2\ln\epsilon,
\qquad 0<\epsilon<1,
$$

等价于

$$
\alpha_i\leftarrow\alpha_i\epsilon^{x_i^2}.
$$

它不是等到“连续多次触发”才开始，而是在每次满足条件时对相关活跃坐标执行。这里的 $\epsilon$ 是衰减系数，**不是探索策略的** $\epsilon$。

这也意味着必须分清更新阶段：<span style="color:#d1242f"><strong>本轮 </strong>$z^\delta$<strong> 用的是计算注入时的步长；之后的 decay 改变后续使用的步长，不会追溯重算已经登记的历史注入。</strong></span>

最终 $\beta$ 有两类变化来源：误差与 $p$ 形成的元梯度，可以增大或减小步长；条件式 decay 则将其向下调。论文还使用 $\beta$ 的范围限制与部分敏感度状态重置。[2]

### 8.4 这是否是对完整程序的精确求导？

不是。前面的 $h,\bar z$ 推导采用局部历史敏感度、半梯度与对角近似。加入 bound 后，缩放因子本身也依赖步长；最终算法还包含 clipping、decay 和 reset。不能把这些递推统称为“对最终执行程序做了无近似的总导数”。

**True Online 的对应关系、步长敏感度的近似质量，以及完整学习器的稳定性，是三个需要分别检查的层次。**

[![SwiftTD 在 Pong 上的步长保护与衰减机制对比]({{ '/assets/images/swifttd-bound-decay-comparison.png' | relative_url }})]({{ '/assets/images/swifttd-bound-decay-comparison.png' | relative_url }})
<p class="figure-caption">图 2：Pong 预测实验中的三种组合：左图仅在 True Online TD($\lambda$) 上加入步长优化，中图加入保护但关闭 decay，右图为完整 SwiftTD。横轴是元步长 $\theta$，纵轴是初始步长，色阶表示 lifetime error，越低越好；斜线区域表示发散。中图没有发散，但大步长区域仍有较大误差，说明防止坏更新与调整未来步长各有作用。这是该实验参数范围内的结果。图源：Javed、Sharifnassab 与 Sutton，<a href="https://rlj.cs.umass.edu/2024/papers/RLJ_RLC_2024_111.pdf#page=10">SwiftTD（2024），Figure 5</a> [2]。由论文原图裁切，点击图片可查看大图。</p>

## 9. 把变量放回一次真实的 SwiftTD 更新

现在再读 Algorithm 1，可以先看两条主要更新，而不是先背全部临时变量：

$$
\begin{aligned}
\Delta w_i&=\delta'z_i-v^\delta z_i^{\delta},\\
\Delta\beta_i^{\mathrm{meta}}
&=\frac{\theta}{e^{\beta_i}}(\delta'-v^\delta)p_i.
\end{aligned}
$$

$z$ 与 $p$ 接收误差信号，但**更新的对象不同**：一个作用于**价值参数**，一个作用于**步长参数**。 $h$ 和 $\bar z$ 则负责让后者拥有必要的敏感度信息。

一次接收—更新循环，可以按下面的阶段理解。它是信息流导读，不替代原论文包含缓存轮换、稀疏循环和重置的完整伪代码。

| 阶段 | 做什么 | 必须保留的区别 |
|---|---|---|
| 读取反馈 | 得到 $R_{t+1},\mathbf x_{t+1}$，计算更新前的下一预测 | 下一状态已到达，不代表本轮权重已更新 |
| 形成误差 | 由保存的旧预测形成 $\delta'$，保留已有 $v^\delta$ | 这里读到的是上一轮留下的漂移 |
| 价值与步长学习 | 旧 $z$ 更新 $w$，旧 $p$ 更新 $\beta$；更新敏感度缓存 | 当前的元梯度没有使用尚未登记的新 $xh$ |
| 历史迹衰减 | 对 $z,p,\bar z$ 做 $\gamma\lambda$ 衰减 | 它们递推形式相似，含义不同 |
| 登记新预测 | 计算 $T,\tau,z^\delta$；构造新 Dutch trace、meta trace 与敏感度 | 给下一次转移准备历史信息 |
| 保存下一轮状态 | 保存更新前下一预测，以及本轮 $\Delta\mathbf w$ 在 $\mathbf x_{t+1}$ 上造成的漂移 | 供下一轮形成同一状态的新旧版本差 |

在登记阶段，bound 先约束基础注入；随后满足条件时执行 decay 和相应 reset。所有“当前”“旧”“新”，都必须相对于这个循环解释。

我的理解方式是：**先问一个量是在“处理已经到来的误差”，还是在“为下一次误差登记历史”。** 这样 $p$ 为什么先被使用、再衰减并增加新的 $xh$，就不再是难记的赋值顺序。

## 10. SwiftTD 到 Swift-Sarsa：改变学习对象，保留核心机制

### 10.1 一套状态预测变成逐动作预测

SwiftTD 不负责通过动作价值选择行为。原始 Swift-Sarsa 为有限离散动作集合 $\mathcal A=\{1,\ldots,K\}$ 维护逐动作的线性预测器：

$$
\boxed{
\hat q(S_t,a)=(\mathbf w_t^a)^\top\mathbf x_t,
\qquad\mathbf x_t=\mathbf x(S_t).
}
$$

这里 $\mathbf x_t$ 是共享的状态输入，而 $\mathbf w_t^a$ 表示动作 $a$ 的权重。$w_{i,t}^a$ 的三个索引分别表示动作、特征坐标和时间。

每个动作还维护自己的 $\mathbf z^a,\boldsymbol\beta^a,\mathbf h^a,\mathbf p^a,\bar{\mathbf z}^a,\mathbf z^{\delta,a}$ 及对应缓存。**“每个动作一套 SwiftTD-like 状态”是合适的结构理解，但不意味着各套预测器使用互不相干的数据独立训练。**[4]

### 10.2 一条实际轨迹连接所有动作

在 $S_t$ 执行 $A_t$，观察 $R_{t+1},S_{t+1}$，再由策略选择 $A_{t+1}$。标准 Sarsa 误差是

$$
\delta_t=R_{t+1}+\gamma\hat q(S_{t+1},A_{t+1},\mathbf w_t)
-\hat q(S_t,A_t,\mathbf w_t).
$$

$A_t$ 确定这次实际转移，$A_{t+1}$ 的价值进入 bootstrap。新选中的 $A_{t+1}$ 随后还要登记自己的资格迹，为下一次反馈做准备。

使用与预测部分相同的版本记号，令

$$
d_t=\hat q(S_t,A_t,\mathbf w_t)
-\hat q(S_t,A_t,\mathbf w_{t-1}),
\qquad\tilde\delta_t=\delta_t+d_t.
$$

在原始分块参数表示下，对各动作的权重更新结构为

$$
\Delta w_{i,t}^a
=\tilde\delta_tz_{i,t}^a-d_tz_{i,t}^{\delta,a}.
$$

<span style="color:#d1242f"><strong>同一个标量误差，经不同动作保留的历史迹分配更新。</strong></span> 本轮没执行的动作，可能因为此前执行过、仍有非零迹而获得更新；这不是知道了“本轮改选它会得到什么”的反事实奖励。

完成学习后，所有动作的旧迹衰减，但新基础注入只进入 $A_{t+1}$。例如，对 meta trace 可以写成

$$
p_{i,t+1}^a
=\gamma\lambda p_{i,t}^a
+\mathbf 1\{a=A_{t+1}\}\,x_{i,t+1}h_{i,t}^a,
$$

其中 $h_{i,t}^a$ 对应新预测被记录时使用的权重版本。动作指示项非常重要：**并不是所有动作**每一步都增加当前的 $xh$。 权重迹与迹敏感度同样遵循“所有历史衰减，只有选中动作新增”的动作分工。[4]

### 10.3 一个带漂移修正的局部算例

假设已运行一段时间，只有一个特征、两个动作。当前 $x_t=x_{t+1}=1$，执行 $L$ 后下一步选择 $R$。给定

$$
w_t^L=0.4,\quad w_t^R=0.8,\quad
R_{t+1}=1,\quad\gamma=0.9,
$$

以及缓存的旧当前动作预测为 $0.35$，所以 $d_t=0.05$。旧资格迹为

$$
z_t^L=0.2,\quad z_t^R=0.1,\quad
z_t^{\delta,L}=0.05,\quad z_t^{\delta,R}=0.
$$

这里只核对权重、漂移和新资格的局部计算，不声称给出了从零初始化运行完整算法的所有元状态。

首先

$$
\delta_t=1+0.9\times0.8-0.4=1.32,
\qquad\tilde\delta_t=1.37.
$$

两动作的更新为

$$
\Delta w_t^L=1.37\times0.2-0.05\times0.05=0.2715,
\qquad
\Delta w_t^R=1.37\times0.1=0.137.
$$

因此 $w_{t+1}^L=0.6715,w_{t+1}^R=0.937$。即使这次执行的是 $L$，历史 $R$ 仍得到更新。

接着取 $\gamma\lambda=0.72$，两套旧迹衰减为 $0.144,0.072$。假设此阶段用于新 $R$ 注入的步长为 $0.04$、$\eta=0.1$，则 $\tau=0.04$ 不触发 bound，$T=0.072$，所以

$$
z_{t+1}^R=0.072+0.04(1-0.072)=0.10912,
\qquad z_{t+1}^L=0.144.
$$

下一轮缓存的旧 $R$ 预测仍应是本轮更新前算出的 $0.8$；本轮已将该预测推到 $0.937$，因此保存的下一轮漂移是 $0.137$。**动作选择、权重学习、资格登记和版本保存，是相互连接但不能颠倒的步骤。**

### 10.4 每个动作独立一套参数，是实现选择，不是 TD 的唯一可能

原始分块结构也可以写成一个大向量。例如

$$
\mathbf w=\begin{bmatrix}\mathbf w^L\\\mathbf w^R\end{bmatrix},
\qquad
\mathbf x(S,L)=\begin{bmatrix}\mathbf x(S)\\\mathbf0\end{bmatrix},
\qquad
\mathbf x(S,R)=\begin{bmatrix}\mathbf0\\\mathbf x(S)\end{bmatrix}.
$$

这样又得到 $\hat q(S,a)=\mathbf w^\top\mathbf x(S,a)$。动作只是决定激活哪个参数块。

若动作数为 $K$、特征维数为 $n$，权重与各类辅助向量的稠密存储都具有 $O(Kn)$ 的规模。共享的固定线性状态—动作特征可以改变这一表示；但同时需要考虑动作间泛化与干扰，以及怎样高效选择动作。**减少参数量，不自动等于减少动作搜索开销。**

[![Swift-Sarsa 在含大量噪声特征的控制任务中的参数敏感性实验]({{ '/assets/images/swift-sarsa-control-parameter-sensitivity.png' | relative_url }})]({{ '/assets/images/swift-sarsa-control-parameter-sensitivity.png' | relative_url }})
<p class="figure-caption">图 3：Swift-Sarsa 在 operant conditioning benchmark 上的控制实验，输入中只有少量相关信号，其余为噪声。两幅图的 $n=60{,}000$ 与 $n=30{,}000$ 是输入维数；横轴是元步长，纵轴是初始步长，色阶表示 lifetime reward，即全程平均奖励，越高越好。它补充的是控制效果的实验证据，适用范围是论文评测的这一基准。图源：Khurram Javed、Richard S. Sutton，<a href="https://arxiv.org/pdf/2507.19539v1#page=4">Swift-Sarsa（2025），Figure 1</a> [4]，<a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>。由论文原图裁切，点击图片可查看大图。</p>

## 11. 到这里，怎样把整个体系组织起来？

### 11.1 不是独立插件，而是存在接口依赖的模块

从功能看，可以区分表示、目标与误差、时间信用、步长适应、稳定性和控制。但从实际公式看，它们并不互相独立。

例如，改变资格迹定义，就改变了 $w$ 的更新方向；改变步长作用的位置，就可能改变 $z$ 对 $\beta$ 的依赖；改变权重更新中的旧预测修正，就会改变 $h$ 必须保存的时间版本。加入动作选择之后，学习器又会通过行为改变以后收到的数据。

因此，**模块化的价值是看清职责与接口，而不是宣称所有模块都能原样替换。** 我会用下面三条关系检查一个组合是否真的成立：

$$
(\text{误差},\ \text{资格迹},\ \text{版本修正})
\longrightarrow\Delta\mathbf w,
$$

$$
(\text{记录的预测敏感度},\ \text{后续误差})
\longrightarrow\Delta\boldsymbol\beta,
$$

$$
\boldsymbol\beta\longrightarrow\boldsymbol\alpha
\longrightarrow\text{注入规则与未来学习过程}.
$$

这里的箭头表示信息依赖；不是说 $z$ 或 $p$ 生成了 TD error。TD error 来自奖励、bootstrap 与预测，两种迹决定这个信号怎样参与各自的更新。

### 11.2 用退化情况检查是否真正理解

当 $\lambda=0$ 时，当前完整迹等于基础注入，$z_i=z_i^\delta$，所以权重中的 $d_t(z_i-z_i^\delta)$ 消失。但 $h$ 仍可能包含先前学习的敏感度，不能说“整个算法失去所有记忆”。

当 $\theta=0$ 时，基于 $p$ 的元梯度关闭；若 decay 仍启用，步长仍可能下降。要比较真正的固定步长，需要把相关的步长变化来源分别关掉。

当暂时没有 $p$ 而只用当前 $xh$ 时，改变的是元目标的时间信用机制；不能把它称为完全相同算法的省内存实现，也不能自动等同于原始 TIDBD，因为底层更新可能仍然不同。

当更换可训练的特征网络时，$\mathbf x$ 本身也会变化。前面的固定特征与对角敏感度假设，就不能不加检查地照搬。

## 12. 总复盘：从表格 Sarsa 到 Swift-Sarsa，整条路线如何展开？

到这里再回头看前面三篇文章，我会发现：我们最开始只是在问“怎样估计采取一个动作后的回报”，后来却不得不处理延迟反馈、函数泛化、在线更新的一致性、学习率本身的优化，最后再把这些机制放回控制任务。

**这不是一条所有算法都严格“前后替代”的历史顺序。** 更准确地说，它是两条相互关联的研究主线：一条研究**误差如何形成并沿时间传播**，另一条研究**参数应该以多快的速度学习**。SwiftTD 把两条主线接到同一个在线预测器中，Swift-Sarsa 再把这个预测器用于动作价值控制。

### 12.1 表格 Sarsa：先把“学习什么”定义清楚

最开始的表格 Sarsa 学习动作价值 $Q(S_t,A_t)$。它使用在 $S_{t+1}$ **由当前策略实际选出的** $A_{t+1}$ 构造目标：

$$
\delta_t=R_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t),
$$

$$
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\delta_t.
$$

这里首先解决的是 **on-policy control 的学习信号**：既评价当前执行的动作，也用后续实际选择的动作估计未来价值。

但它留下了两个明显限制：一步更新很难迅速处理长时延反馈；表格也无法很好地扩展到巨大的状态—动作空间。

### 12.2 从一步到多步：$n$-step、$\lambda$-return 与 Sarsa($\lambda$)

为了让较晚出现的奖励更快影响较早的决策，我们先看 $n$-step return。在未终止的转移上，可以写成：

$$
G_t^{(n)}=\sum_{k=1}^{n}\gamma^{k-1}R_{t+k}
+\gamma^nQ(S_{t+n},A_{t+n}).
$$

$n=1$ 对应一步 bootstrap；更大的 $n$ 纳入更多实际奖励。$\lambda$-return 则对不同步数的 return 加权，提供一条从短视 bootstrap 到较长回报的连续过渡。

问题随之变成：**如果 forward view 的目标依赖未来，怎样做到每步交互后立刻在线更新？** 资格迹给出 backward-view 的实现思路。以表格 accumulating trace 为例：

$$
z_t(s,a)=\gamma\lambda z_{t-1}(s,a)
+\mathbf1\{S_t=s,A_t=a\},
$$

$$
Q(s,a)\leftarrow Q(s,a)+\alpha\delta_t z_t(s,a).
$$

当前 $\delta_t$ 不再只影响一个表项，还会传给仍有非零 trace 的历史状态—动作对。

**新增的核心能力是 temporal credit assignment。** 这里必须区分：$\lambda$-return 是一种多步学习目标的 forward-view 表述，而 $z$ 是在线分配历史信用的计算状态；两者有关联，却不是同一个对象。

### 12.3 从表格到线性函数近似：让经验可以跨状态泛化

接下来要解决的是表示问题。表格中每个 $(s,a)$ 都有独立的数值，而线性 Sarsa 用特征表示动作价值：

$$
\hat q(S_t,A_t,\mathbf w_t)
=\mathbf w_t^\top\mathbf x(S_t,A_t).
$$

共享特征使不同状态—动作对能够共享参数和学习经验；代价是一次更新可能同时改变多个状态—动作对的预测。

半梯度方法仍使用 Sarsa TD error，但对当前价值参数求导时，把 bootstrap target 暂时视为固定。加入资格迹后，基本更新方向由当前特征转为历史特征迹：

$$
\mathbf z_t=\gamma\lambda\mathbf z_{t-1}+\mathbf x(S_t,A_t),
\qquad
\Delta\mathbf w_t=\alpha\delta_t\mathbf z_t.
$$

**这一步改变的是价值的表示和泛化方式，而不是把 Sarsa target 换成另一种 target。** 同时，在线参数共享也为 True Online 要解决的问题埋下了伏笔。

### 12.4 True Online Sarsa($\lambda$)：处理“参数一直在变”带来的差异

有了函数近似与资格迹，我们希望 backward view 真正对应在线不断更新参数时的 forward view。但是，**传统 accumulating trace 并不会自动保证这种一致性**。

True Online Sarsa($\lambda$) 进一步引入两种相互配合的机制：

- **Dutch trace**：修正已有 trace 与当前特征重叠时的增量。
- **旧预测** / $Q_{\mathrm{old}}$ correction：处理同一个 state-action pair 在不同参数版本下产生的预测变化。

因此，True Online 并非简单地“保留更多历史”，而是改变了在线资格迹和权重更新的配合方式。标准线性、固定步长设定下，它有明确的 online forward/backward 对应关系；后面加入动态步长与保护机制时，不能不加条件地继承这些性质。

到此，第一条主线已经具备：**on-policy target、temporal credit、函数近似与在线版本修正**。

### 12.5 另一条主线：从 IDBD 到 TIDBD，开始学习“如何学习”

此前的 Sarsa 系列主要回答如何根据误差更新价值参数，但通常需要人为给定学习率。IDBD 另辟一条主线，为每个特征维护

$$
\alpha_i=e^{\beta_i},
\qquad
h_i\approx\frac{\partial w_i}{\partial\beta_i}.
$$

这里 $\beta_i$ 决定第 $i$ 个特征的学习速度，$h_i$ 追踪学习过程对这个选择有多敏感。

TIDBD 将这类逐特征步长适应带入 TD：不仅有 bootstrap TD error，还有历史资格迹参与权重更新。因此，更新方向里的历史迹 $z$ 与当前预测导数里的特征 $x$ 会同时影响 $h$ 的递推。

一个值得保留的边界是：原始 TIDBD 的**直接元更新**使用 $\delta xh$；它**并没有** SwiftTD 后来单独维护的 $p$。 因此，IDBD $\rightarrow$ TIDBD 与 Sarsa $\rightarrow$ True Online Sarsa 是两种不同的改进维度，而不是前者沿着后者的同一条直线出现。

### 12.6 两条主线汇合：True Online step-size optimization 与 SwiftTD

现在我们尝试把 **True Online 的在线价值学习** 与 **逐特征的 step-size optimization** 放到一起，问题就不再只是“写一个 $\alpha_i$”。因为改变步长会同时影响权重和资格迹，在线预测又涉及新旧两个参数版本。

这时，本篇详细推导过的几个变量终于有了共同位置：

- $h_i$：保存 $\beta_i$ 对权重的**敏感度**，用来计算当前预测敏感度 $x_i h_i$。
- $p_i$：把历次预测敏感度沿时间累积，和后续误差信号一起更新 $\beta_i$；它是 meta-level temporal credit。
- $\bar z_i$：保存资格迹对 $\beta_i$ 的**敏感度**，避免求 $h$ 时漏掉“步长 $\rightarrow$ trace $\rightarrow$ 权重”的路径。
- $d_t$ 与不同版本的 $h$：处理同一预测在连续权重版本之间的变化及其导数。

在这个基础上，SwiftTD 再加入 **overshoot bound** 和 **step-size decay**：前者限制本次基础 trace 注入，后者在触发条件下减小未来的步长。

因此，可以把 SwiftTD 看成：

$$
\boxed{
\text{True Online TD}(\lambda)
+\text{per-feature step-size optimization}
+\text{stability mechanisms}
}
$$

这里写的是 **TD prediction**，不是 Sarsa 的动作选择。它说明了为什么 SwiftTD 和 True Online Sarsa 可以共享重要机制，却不能直接把两者当成同一个任务。

### 12.7 Swift-Sarsa：将预测机制接回 on-policy control

最后一步，把预测的对象从状态价值扩展为动作价值。原始 Swift-Sarsa 对有限离散动作 $a$ 维护：

$$
\hat q(S_t,a)=(\mathbf w^a)^\top\mathbf x(S_t),
$$

并为每个动作保存相应的 $\mathbf z^a,\boldsymbol\beta^a,\mathbf h^a,\mathbf p^a,\bar{\mathbf z}^a$ 等状态。

关键不是“把 SwiftTD 复制 $K$ 次就结束”，而是让这些状态进入**同一个控制闭环**：策略选 $A_t$，环境给出 $R_{t+1},S_{t+1}$，策略再选 $A_{t+1}$，由此形成共同的 Sarsa TD error；历史非零资格迹决定哪些动作的参数可能收到这次更新。

所以这一阶段增加的是 **action value + policy + 跨动作的时间信用分配**。它仍然采用线性、有限离散动作的实现；大规模共享参数或 LLM/Agent 动作空间属于进一步的研究问题。

### 12.8 最后压缩成一张路线图

| 阶段 | 新增或改变的核心机制 | 主要解决的问题 |
|---|---|---|
| 表格 Sarsa | $Q(S,A)$ 与真实下一动作的 on-policy TD target | 学习什么动作价值 |
| $n$-step / $\lambda$-return | 多步 bootstrap 与 return 加权 | 如何使用更长时间范围的反馈 |
| Sarsa($\lambda$) | 资格迹 $z$ | 延迟反馈如何在线追责过去 |
| 线性半梯度 Sarsa($\lambda$) | $\mathbf w^\top\mathbf x$，共享参数 | 如何从表格过渡到泛化 |
| True Online Sarsa($\lambda$) | Dutch trace + 旧预测修正 | 在线更新中怎样处理参数版本差异 |
| IDBD → TIDBD | $\beta,\alpha,h$；从监督增量学习进入 TD | 每个特征应该学多快 |
| True Online step-size optimization | $p,\bar z$ 与多版本 $h$ | 怎样对完整在线更新追踪步长敏感度 |
| SwiftTD | $\tau$、bounded injection、条件式 decay | 怎样保护自适应步长引发的更新 |
| Swift-Sarsa | 逐动作 Swift 状态 + Sarsa target + policy | 怎样把预测机制用于动作控制 |

这张表只归纳本系列的**问题与模块关系**，并不表示每一行都严格以紧邻的上一行为唯一数学或历史起点。特别是，**IDBD/TIDBD 是步长学习分支，SwiftTD 是预测算法，Swift-Sarsa 才重新接回 control**。

如果把整条路线压缩成两个问题，就是：

> **第一条线：**预测误差怎样沿实际交互历史传播，且在线更新时保持正确的版本关系？
>
> **第二条线：**每个参数应以什么速度学习，学习过程对这个速度有多敏感，速度过大时怎样保护？

**SwiftTD 在预测层面组合了这两条线；Swift-Sarsa 则让它们共同服务于动作价值与策略选择。** 这也是我理解整个系列时最重要的模块化视角。

## 13. 我的理解：先追踪影响，再谈改进

这条路线中，最容易让我混淆的是：好多变量都看起来“与过去有关”，于是被统称为资格迹。现在更有用的区分是：

| 量 | 保存或表示什么 | 在更新中的落点 |
|---|---|---|
| $w_i$ | 价值预测参数 | 生成预测，进而形成误差 |
| $z_i$ | 带有步长与历史传播结构的资格 | 直接进入 $\Delta w_i$ |
| $h_i$ | 学习历史对权重的近似步长敏感度 | 形成预测敏感度，并参与自己的后续递推 |
| $x_i h_i$ | 指定参数版本下，预测对 $\beta_i$ 的近似敏感度 | 作为新信息进入 $p_i$ |
| $p_i$ | 衰减累积的历史预测敏感度 | 与误差信号一起进入 $\Delta\beta_i$ |
| $\bar z_i$ | 资格迹对 $\beta_i$ 的近似敏感度 | 进入 $h_i$ 的递推 |
| $d_t$ | 同一预测在相邻参数版本下的变化 | 权重修正与相应敏感度计算 |
| $T_t$ | 衰减后历史迹与当前特征的带符号内积 | Dutch trace 与其敏感度递推 |
| $\tau_t$ | 当前步长组合的基础注入强度 | 触发 bound 和条件式 decay |
| $z_i^\delta$ | bounded 基础注入，尚未乘 $1-T$ | 同时用于资格递推和权重修正 |

对我来说，这比记住每个变量的英文名称更重要。读一个新公式时，我会先问：它在描述数值本身，还是数值对某个参数的导数？它已经包含步长了吗？它记录的是当前版本还是旧版本？它最后改的是价值参数还是学习速度？

**一个变量真正被理解，不是因为能说出定义，而是因为能指出：没有它，哪一条更新会漏掉什么信息。**

## 14. 最后收束：两条学习回路，一个控制闭环

普通学习利用当前误差与历史资格改变价值参数：

$$
(\delta_t,\mathbf z_t,\text{True Online correction})
\longrightarrow\mathbf w_{t+1}.
$$

元学习利用误差与历史预测敏感度改变步长参数：

$$
(\delta_t,\mathbf p_t)
\longrightarrow\boldsymbol\beta
\longrightarrow\boldsymbol\alpha.
$$

$h$ 和 $\bar z$ 使第二条回路能够近似追踪第一条回路对步长的依赖；bound 与 decay 对注入和步长施加额外约束。Swift-Sarsa 再把学习到的动作价值连接到策略与真实环境，形成

$$
\text{动作价值}\longrightarrow\text{行为}\longrightarrow\text{经验}
\longrightarrow\text{价值与步长更新}.
$$

我现在对整个体系的理解是：在线强化学习不仅需要记住预测了多少，还需要记住**历史更新怎样影响当前预测**，以及**这种学习过程本身应当怎样调整**。 SwiftTD 把这些依赖放进了一个在线预测器；Swift-Sarsa 则让这个预测器参与控制。

---

## 附录 A：把 $h$ 的各条敏感度路径完整合起来

这一段用于核对变量如何落到更新中。假设无 bound、无 clipping/decay，并采用前文的固定特征、目标半梯度和对角近似。由

$$
w_{i,t+1}=w_{i,t}+\tilde\delta_tz_{i,t}-d_tz_{i,t}^{\delta}
$$

求导：

$$
\begin{aligned}
h_{i,t+1}\approx{}&h_{i,t}
+z_{i,t}\frac{\partial\tilde\delta_t}{\partial\beta_i}
+\tilde\delta_t\bar z_{i,t}\\
&-z_{i,t}^{\delta}\frac{\partial d_t}{\partial\beta_i}
-d_t\frac{\partial z_{i,t}^{\delta}}{\partial\beta_i}.
\end{aligned}
$$

各导数的来源分别是

$$
\begin{aligned}
\frac{\partial\tilde\delta_t}{\partial\beta_i}
&\approx-x_{i,t}h_{i,t-1},\\
\frac{\partial d_t}{\partial\beta_i}
&\approx x_{i,t}(h_{i,t}-h_{i,t-1}),\\
\frac{\partial z_{i,t}^{\delta}}{\partial\beta_i}
&=z_{i,t}^{\delta}\quad\text{（无 bound 的 }e^{\beta_i}x_{i,t}\text{ 注入）}.
\end{aligned}
$$

第一行之所以用 $h_{i,t-1}$，是因为 $\tilde\delta_t$ 减去的是旧版本预测；bootstrap target 的导数在这里停止。

代回并整理：

$$
\begin{aligned}
h_{i,t+1}\approx{}&h_{i,t}
-x_{i,t}h_{i,t-1}(z_{i,t}-z_{i,t}^{\delta})\\
&-x_{i,t}h_{i,t}z_{i,t}^{\delta}
+\tilde\delta_t\bar z_{i,t}
-d_tz_{i,t}^{\delta}.
\end{aligned}
$$

这就解释了为什么需要 $\bar z$、新旧 $h$ 和 $d_t$。它对应 SwiftTD 附录 A.2 的无保护机制敏感度结构，**不是对最终 bound/clipping/reset 程序的精确自动微分结果**。[2]

## 附录 B：阅读 Algorithm 1 时的符号映射

| 本文 | 论文/伪代码中常见写法 | 核对要点 |
|---|---|---|
| $\mathbf x$ | $\boldsymbol\phi$ | 特征向量，不改变算法含义 |
| $\tilde\delta_t$ | $\delta'$ | 使用缓存的旧预测基准 |
| $d_t$ | $v^\delta$ | 本轮读取时，是上一轮为当前状态保存的漂移 |
| $\delta_t$ | $\delta'-v^\delta$ | 在本文明确的版本对应下为标准 TD error |
| $z_{i,t}^{\delta}$ | `z_delta[i]` | 基础注入，不是完整迹，也不含乘上的 $1-T$ |
| $h_{i,t}$ 与相邻版本 | `h`、`h_old`、`h_temp` | 程序变量会被覆盖，必须结合执行顺序 |
| $p_{i,t}$ | `p[i]` | 使用旧迹调整步长，再登记新预测敏感度 |
| $T$ | SwiftTD 的 $T$、Swift-Sarsa 的 $b$ | 都需在相应的衰减与选中动作阶段解释 |

本篇保留了“概念推导”和“实现版本”之间的区别。单靠变量名把不同时刻的式子拼在一起，恰恰是最容易出现错误的地方。

## 参考文献

[1] Alex Kearney, Vivek Veeriah, Jaden B. Travnik, Richard S. Sutton, Patrick M. Pilarski. [*TIDBD: Adapting Temporal-difference Step-sizes Through Stochastic Meta-descent*](https://arxiv.org/abs/1804.03334), 2018. 重点参照 Algorithm 1 与第 4 节。

[2] Khurram Javed, Arsalan Sharifnassab, Richard S. Sutton. [*SwiftTD: A Fast and Robust Algorithm for Temporal Difference Learning*](https://rlj.cs.umass.edu/2024/papers/Paper111.html). Reinforcement Learning Journal, 2024. 重点参照第 5 节、Algorithm 1、附录 A.1–A.2 与 B。

[3] Harm van Seijen, A. Rupam Mahmood, Patrick M. Pilarski, Marlos C. Machado, Richard S. Sutton. [*True Online Temporal-Difference Learning*](https://arxiv.org/abs/1512.04087). Journal of Machine Learning Research, 2016.

[4] Khurram Javed, Richard S. Sutton. [*Swift-Sarsa: Fast and Robust Linear Control*](https://arxiv.org/abs/2507.19539), 2025. 重点参照第 2 节与 Algorithm 1；实验与结论边界参照第 4 节。

[5] Richard S. Sutton, Andrew G. Barto. *Reinforcement Learning: An Introduction*, 2nd edition, 2018. 基础 TD、资格迹与 Sarsa 记号沿用本系列前文。
