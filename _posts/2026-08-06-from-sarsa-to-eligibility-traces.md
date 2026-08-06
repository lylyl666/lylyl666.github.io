---
layout: post
title: "从 Sarsa 到资格迹：我的强化学习基础总结"
date: 2026-08-06 12:00:00 +0800
categories: [Reinforcement Learning]
tags: [Sarsa, TD, n-step, lambda-return, eligibility-trace]
series: "强化学习基础"
math: true
---


> 这篇文章记录我从回报、价值函数和 TD，逐步学习到表格 Sarsa、n-step Sarsa、$\lambda$-return 与资格迹的过程。重点不只是罗列公式，而是解释每个概念为什么会自然地引出下一个概念。

![强化学习学习主线总览]({{ '/assets/images/rl-three-lessons-overview.png' | relative_url }})

## 1. 这条学习主线在强化学习中的位置

这一阶段学习的是：

- **model-free**：不显式使用环境转移模型进行规划，而是直接从经验学习价值；
- **on-policy**：用当前执行的策略产生经验，同时评价和改进这个策略；
- **value-based control**：学习动作价值 $Q(s,a)$，再用它比较和选择动作。

主线可以概括为：

```text
一条轨迹上的回报 G_t
    ↓
状态价值 vπ(s) 与动作价值 qπ(s,a)
    ↓
Bellman 关系与 TD
    ↓
表格 Sarsa
    ↓
n-step Sarsa
    ↓
λ-return
    ↓
资格迹与 Sarsa(λ)
```

这条链路一直在解决同一个问题：**怎样根据一条条真实轨迹，更准确、更高效地估计当前状态或动作的长期价值。**

---

## 2. 从一条轨迹的回报 $G_t$ 开始

从时间 $t$ 开始，未来累计折扣回报为：

$$
G_t=R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots
$$

它也可以递归写成：

$$
G_t=R_{t+1}+\gamma G_{t+1}
$$

其中，$\gamma$ 控制未来奖励随时间距离衰减的速度。$\gamma$ 越小，越重视近期奖励；$\gamma$ 越接近 1，远期奖励衰减得越慢。

> 我的理解：$G_t$ 表示站在当前时间点，沿着这一条具体轨迹继续走，最终实际获得了多少累计奖励。但它只是一条轨迹的结果，同一个状态出发可能得到不同的 $G_t$。

未来尚未发生时，$G_t$ 是一个随机变量；轨迹发生后，它才成为可以计算的确定结果。算法中的 $V$ 和 $Q$ 才是价值估计。

还需要区分：$\gamma$ 的幂表示**时间折扣**，并不是 bootstrap。Bootstrap 是用一个已有价值估计去更新另一个价值估计，例如使用 $V(S_{t+1})$ 更新 $V(S_t)$。

---

## 3. 状态价值与动作价值

状态价值定义为：

$$
v_\pi(s)=\mathbb E_\pi[G_t\mid S_t=s]
$$

它表示当前处于状态 $s$，当前及后续动作都按照策略 $\pi$ 选择时，未来回报的期望。

动作价值定义为：

$$
q_\pi(s,a)=\mathbb E_\pi[G_t\mid S_t=s,A_t=a]
$$

它表示当前处于状态 $s$，先固定执行动作 $a$，之后再按照策略 $\pi$ 行动时，未来回报的期望。

> 我的理解：$v_\pi(s)$ 和 $q_\pi(s,a)$ 最核心的区别，是当前动作是否被固定。状态价值中当前动作也由策略决定；动作价值中当前动作先被指定，之后再按策略行动。

这也解释了为什么控制算法通常学习 $Q(s,a)$。为了选择动作，需要比较同一个状态下不同动作的长期价值；仅知道 $V(s)$，只能知道这个状态整体有多好，不能直接比较动作。

在表格 Sarsa 中，算法为每个离散的状态—动作对保存一个数：

$$
Q[s,a]
$$

它通常不保存全部历史轨迹，而是把历史经验逐步压缩进 Q 表。

---

## 4. Bellman 关系与 TD

Bellman 方程不是一个具体算法，而是价值函数应满足的递归关系。由 $G_t=R_{t+1}+\gamma G_{t+1}$ 可以得到：

$$
v_\pi(s)=\mathbb E_\pi\left[R_{t+1}+\gamma v_\pi(S_{t+1})\mid S_t=s\right]
$$

它表达的是：

$$
\text{当前价值}=\text{下一步奖励}+\text{折扣后的下一状态价值}
$$

TD 的全称是 **Temporal-Difference Learning，时序差分学习**。在不知道环境模型时，TD 用一次真实采样近似 Bellman 方程中的期望：

$$
\delta_t=R_{t+1}+\gamma V(S_{t+1})-V(S_t)
$$

然后更新：

$$
V(S_t)\leftarrow V(S_t)+\alpha\delta_t
$$

其中，$R_{t+1}+\gamma V(S_{t+1})$ 是新目标，$\delta_t$ 是新目标与旧估计之间的差。

---

## 5. 表格 Sarsa 更新哪个 Q

Sarsa 的 TD error 为：

$$
\delta_t=R_{t+1}+\gamma Q(S_{t+1},A_{t+1})-Q(S_t,A_t)
$$

更新公式为：

$$
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\delta_t
$$

公式中虽然出现了两个 Q，但作用不同：

- 真正被更新的是刚刚执行的 $Q(S_t,A_t)$；
- $Q(S_{t+1},A_{t+1})$ 用来估计剩余未来，构造当前 target。

> **我的记忆方式：** 更新刚刚执行的动作，参考下一步准备执行的动作。

Sarsa 使用当前策略实际选择的 $A_{t+1}$，所以它属于 on-policy 方法。

---

## 6. 为什么需要 n-step Sarsa

一步 Sarsa 只使用一个真实奖励：

$$
G_{t:t+1}=R_{t+1}+\gamma Q(S_{t+1},A_{t+1})
$$

n-step Sarsa 使用前 $n$ 步真实奖励，再在第 $n$ 步 bootstrap：

$$
G_{t:t+n}=\sum_{k=1}^{n}\gamma^{k-1}R_{t+k}+\gamma^nQ(S_{t+n},A_{t+n})
$$

一次 n-step return 来自**同一条实际轨迹中的连续片段**。同一条已经发生的轨迹上，return 是确定的；所谓随机性更大，是指从相同的状态—动作对重复运行时，环境随机性或探索动作可能产生不同轨迹，使 target 在不同采样之间波动更大。

- 小 $n$：等待短，但更依赖当前 Q 是否准确；
- 大 $n$：使用更多真实奖励，bootstrap 更少，但等待更久，也通常暴露于更多轨迹随机性。

这里“方差通常更大”不是绝对定理。如果环境和策略都是确定性的，相同起点总产生相同轨迹，那么采样方差也可以为 0。

---

## 7. $\lambda$-return：不固定选择唯一的 n

固定选择某一个 $n$ 不够灵活，因此 $\lambda$-return 把不同长度的 n-step return 加权混合：

$$
G_t^\lambda=(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}G_{t:t+n}
$$

展开为：

$$
G_t^\lambda=(1-\lambda)G_{t:t+1}+(1-\lambda)\lambda G_{t:t+2}+(1-\lambda)\lambda^2G_{t:t+3}+\cdots
$$

> 我的理解：不同 n-step return 都在估计同一个长期价值，只是使用的真实奖励长度不同。$\lambda$-return 不选唯一的 $n$，而是把短期、中期和长期的估计综合起来。

$G_t^\lambda$ 中的上标 $\lambda$ 不是指数运算，而是标签，表示这个学习目标由参数 $\lambda$ 混合多个 n-step return 得到。

$\gamma$ 和 $\lambda$ 的作用不同：

- $\gamma$：控制同一个 return 内，未来奖励的时间折扣；
- $\lambda$：控制不同长度 n-step return 的混合比例。

在 episodic 任务中，有限形式会把剩余权重放在最终的完整 return 上。因此：

- $\lambda=0$：得到一步 return；
- $\lambda=1$：得到完整 Monte Carlo return。

$G_t^\lambda$ 是一个综合学习目标，通常不保证等于这条轨迹的完整 $G_t$，因为较短的 n-step return 中仍包含当前可能不准确的 Q 估计。

---

## 8. 资格迹为什么出现

假设奖励只在轨迹末尾出现：

```text
x0 → x1 → x2 → 终点奖励
```

初始所有 Q 都为 0 时，第一次 episode 的终点奖励通常只能先更新 $Q(x_2)$。之后再次经历轨迹，$Q(x_1)$ 才能通过 $Q(x_2)$ 学到价值，随后 $Q(x_0)$ 再从 $Q(x_1)$ 学到价值。

```text
Q(x2) → Q(x1) → Q(x0)
```

> 我的理解：一步 Sarsa 的价值信息像接力一样逐步向前传播。面对延迟奖励时，较早动作可能需要多次访问才能获得明显更新。

资格迹为每个过去访问过的状态—动作对维护一个资格值：

$$
Z_t(s,a)=\gamma\lambda Z_{t-1}(s,a)+\mathbb I(S_t=s,A_t=a)
$$

它表示这个状态—动作目前还有多少资格接收新的 TD error。

当前状态—动作被访问时，资格增加；之后每经过一步，旧资格乘 $\gamma\lambda$ 衰减。因此，$\lambda$ 越大，过去状态—动作保留资格越久；$\lambda=0$ 时，过去资格立即消失，算法退化为一步 Sarsa。

![资格迹的后向视图]({{ '/assets/images/eligibility-trace-backward-view.png' | relative_url }})

Sarsa($\lambda$) 使用当前 TD error 更新所有仍有资格的状态—动作：

$$
Q(s,a)\leftarrow Q(s,a)+\alpha\delta_tZ_t(s,a)
$$

也就是：

$$
\Delta Q(s,a)=\alpha\times\delta_t\times Z_t(s,a)
$$

其中：

- $\alpha$：总体学习速度；
- $\delta_t$：当前新信息与旧预测相差多少；
- $Z_t(s,a)$：这个过去的状态—动作应该接收多少误差。

---

## 9. 资格迹传播的到底是什么

资格迹不是把后面的 Q 直接复制给前面的 Q，也不是把当前奖励平均分给过去。它真正传播的是当前产生的 TD error。

前向 $\lambda$-return 和后向资格迹之间的核心联系是：

$$
G_t^\lambda-Q_t
=
\delta_t+\gamma\lambda\delta_{t+1}+(\gamma\lambda)^2\delta_{t+2}+\cdots
$$

从两个方向看：

- **前向视图**：站在过去的状态—动作看，它会逐渐接收未来 TD errors；
- **后向视图**：站在当前 TD error 看，它会按照资格迹同时更新过去多个状态—动作。

资格迹把原本需要等待未来的前向计算，改写成可以逐步在线执行的后向更新。

需要注意，资格迹主要按照时间距离分配信用，并不能证明距离奖励最近的动作具有最大的真实因果贡献。

---

## 10. 总结

我目前对整条链路的理解是：

1. $G_t$ 描述一条具体轨迹从当前开始实际获得的累计折扣回报；
2. $v_\pi(s)$ 和 $q_\pi(s,a)$ 是多种可能轨迹回报的期望，区别在于当前动作是否固定；
3. 控制问题需要比较动作，因此 Sarsa 学习 $Q(s,a)$；
4. Sarsa 更新刚刚执行的 $Q(S_t,A_t)$，下一状态—动作价值只用于构造 target；
5. n-step Sarsa 使用多步真实奖励；$\lambda$-return 把不同长度的 n-step return 综合起来；
6. 资格迹记录过去状态—动作仍有多少更新资格，使当前 TD error 可以在同一 episode 中更新多个过去的 Q；
7. $\lambda$ 控制这种信用能够保留和传播多远，但不等于真正的因果识别。

> **一句话概括：** 这篇文章讲的是如何从一条轨迹的回报出发，逐步构造动作价值的学习目标，并通过 n-step、$\lambda$-return 和资格迹，让延迟的价值信息更高效地传播到较早的状态—动作。
