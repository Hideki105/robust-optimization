# はじめに

分布的ロバスト最適化の応用例について考えてみたので共有させてください。最小二乗法について触れた後、重み付き最小二乗法について触れ、最後に重みの不確かさを考慮した最小二乗法について提案する。

---
# 最小二乗法とは
係数行列$A$および測定値$b$が与えられたとき、$x$を推定する最小二乗法は式(1)で与えられる。

$$
\min_{x} \|Ax-b\|^2 \tag{1}
$$

式変形すると、
$$
\min_{x} (Ax-b)^T(Ax-b)
$$

式変形すると、

$$
\min_{x} \sum_{i=1}^{n}\left(\sum_{j=1}^{m} a_{ij}x_j-b_i\right)^2
$$

---
# 重み付き最小二乗法とは

重み付き最小二乗法は式(2)で与えられる。

$$
\min_{x} (Ax-b)^T\Sigma(Ax-b) \tag{2}
$$

ただし、$\sigma_i$は測定値$b_i$における計測値の標準偏差であり、
$$
\Sigma = \mathrm{diag}(1/\sigma_1^2,1/\sigma_2^2,\cdots,1/\sigma_n^2)
$$

式変形すると、

$$
\min_{x} \sum_{i=1}^{n}\dfrac{1}{\sigma_i^2}\left(\sum_{j=1}^{m} a_{ij}x_j-b_i\right)^2
$$

---

# 重みの不確かさを考慮した最小二乗法

$$
f_i(x) := \left(\sum_{j=1}^{m} a_{ij}x_j-b_i\right)^2
$$

$$
\begin{aligned}
\hat{\Sigma}  := &\dfrac{1}{\mathrm{trace}(\Sigma)}\Sigma\\
               = &\mathrm{diag}(1/\hat{\sigma}_1^2,1/\hat{\sigma}_2^2,\cdots,1/\hat{\sigma}_n^2)
\end{aligned}
$$

ここで、
$$
\mathrm{trace}(\hat{\Sigma}) = \dfrac{\mathrm{trace}(\Sigma)}{\mathrm{trace}(\Sigma)}=1
$$

よって、
$$
1/\hat{\sigma}_1^2+1/\hat{\sigma}_2^2+\cdots+1/\hat{\sigma}_n^2=1
$$

ここで、正規化した重み$p_i$を下式で定義する。
$$
p_i := \dfrac{1}{\hat{\sigma}_i^2}
$$

重みの不確かさを考慮した最小二乗法を定式化する。

$$
\min_{x}\max_{\mathbb{Q}\in\mathcal{Q}} \mathbb{E_Q}[f]
$$
を解けばよい。ただし、$\max_{\mathbb{Q}\in\mathcal{Q}} \mathbb{E_Q}[f]$は下式で与えられる。

$$
\max_{\mathbb{Q}\in\mathcal{Q}} \mathbb{E_Q}[f] = 
\left|
\begin{aligned}
\min_{q} \quad & f^Tq\\
\textrm{s.t.} \quad & \sum_{i=1}^{n}q_i\ln\dfrac{q_i}{p_i}\leq\epsilon\\
                    & 1^Tq=1    \\
                    & q_i>0, i=1,2,\cdots,n
\end{aligned}
\right.
$$

双対性を使うと、

$$
\max_{\mathbb{Q}\in\mathcal{Q}}\mathbb{E}_{\mathbb{Q}}[f] =
\min_{\lambda>0,\eta}
\{
\epsilon\lambda+\eta
+\lambda\sum_{i=1}^{n}p_i\exp\left(\dfrac{f_i-\eta}{\lambda}-1\right)
\}
$$

よって、
$$
\min_{\lambda>0,\eta,x}
\{
\epsilon\lambda+\eta
+\lambda\sum_{i=1}^{n}p_i\exp\left(\dfrac{\left(\sum_{j=1}^{m} a_{ij}x_j-b_i\right)^2-\eta}{\lambda}-1\right)
\}
$$

重みの不確かさを考慮した最小二乗法は、非線形最適化問題に帰着した。

これをとけば、重みの不確かさを考慮した最小二乗法を実現することができる。今回は制約なし最適化を扱ったが、制約あり最適化にも適用することができる。

# シミュレーション

