## Run Hello_RL_World
```
python _1_ppo.py --env-id CartPole-v1 --total-timesteps 50000

```

## Metrics 

Ok, mình sẽ làm bảng mapping thật chi tiết giữa **metrics trong TensorBoard của CleanRL PPO** ↔ **khái niệm & công thức RL** mà mình và bạn đã bàn trong phần lý thuyết.

---

## 📊 **Bảng mapping: CleanRL PPO metrics ↔ RL theory**

| Nhóm                   | Metric (TensorBoard)        | Công thức / Khái niệm RL                                                                                                      | Ý nghĩa & Cách đọc                                                                                                                   |
|------------------------|-----------------------------|-------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| **Hiệu suất train**    | `charts/SPS`                | —                                                                                                                             | Số step environment xử lý/giây. Chỉ phản ánh tốc độ, không liên quan trực tiếp tới chất lượng.                                       |
|                        | `charts/episodic_length`    | $\text{Length} = \frac{1}{N} \sum_{i=1}^N T_i$                                                                                | Trung bình số bước/episode. Với CartPole, dài hơn nghĩa là agent giữ cột lâu hơn.                                                    |
|                        | `charts/episodic_return`    | $G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k+1}$                                                                                   | Reward trung bình mỗi episode. Ở CartPole, gần như bằng `episodic_length` vì reward = 1 mỗi step.                                    |
|                        | `charts/learning_rate`      | Hyperparam PPO                                                                                                                | Learning rate của optimizer (Adam), ở PPO mặc định giảm tuyến tính để giảm biến động giai đoạn cuối.                                 |
| **PPO-specific**       | `losses/approx_kl`          | $D_{KL}(\pi_{\theta_{\text{old}}} \parallel \pi_\theta)$                                                                      | Đo mức thay đổi giữa policy mới và cũ. PPO muốn giữ KL nhỏ (ổn định update).                                                         |
|                        | `losses/clipfrac`           | $\frac{\#\{\text{samples bị clip}\}}{\text{total samples}}$                                                                   | Tỉ lệ sample mà ratio $\frac{\pi_\theta}{\pi_{\theta_{\text{old}}}}$ bị cắt bởi clip range $\epsilon$. Cao → update bị giới hạn nhiều.|
| **Exploration**        | `losses/entropy`            | $H(\pi(s)) = -\sum_{a} \pi(a\mid s)\,\log \pi(a\mid s)$                                  | Độ ngẫu nhiên của policy. Cao → nhiều exploration; thấp → policy chắc chắn hơn. |
| **Critic performance** | `losses/explained_variance` | $1 - \frac{\mathrm{Var}(V_t - R_t)}{\mathrm{Var}(R_t)}$                                                                       | Xem critic (value function) fit return tốt không. Gần 1 → critic mạnh, gần 0 → yếu.                                                  |
| **Debug**              | `losses/old_approx_kl`      | —                                                                                                                             | KL tính lại từ policy cũ, để so sánh với `approx_kl`.                                                                                |
| **Losses**             | `losses/policy_loss`        | $L^{CLIP}(\theta) = \mathbb{E}\left[\min(r_t(\theta)\hat{A}_t, \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)\right]$ | Mục tiêu tối ưu actor. Giá trị âm → actor tăng reward.                                                                               |
|                        | `losses/value_loss`         | $L^{VF} = (V_\theta(s_t) - R_t)^2$                                                                                            | Mục tiêu tối ưu critic (Value Function). Cao → critic dự đoán kém.                                                                   |

---

## 🔍 **Cách sử dụng bảng này khi đọc kết quả**

1. **Tổng quan chất lượng** → nhìn `episodic_return` và `episodic_length`.
2. **Ổn định PPO** → xem `approx_kl` (ổn định), `clipfrac` (không quá cao).
3. **Exploration** → entropy giảm từ từ, không tụt quá sớm.
4. **Critic** → explained\_variance cao thì value function ổn, ngược lại thì cần cải thiện critic.
5. **Loss patterns**:

   * `policy_loss` không dao động cực mạnh → update actor ổn định.
   * `value_loss` giảm hoặc dao động nhỏ → critic hội tụ.

---

📌 Nếu bạn muốn, mình có thể **chèn luôn công thức PPO đầy đủ kèm highlight từng thành phần tương ứng với metric này** → kiểu một sơ đồ “PPO Objective ↔ TensorBoard log” để nhìn vào loss chart là biết nó tương ứng với phần nào của công thức.
Bạn có muốn mình vẽ sơ đồ đó không?
Mình nghĩ nó sẽ giúp bạn vừa học lý thuyết, vừa đọc log CleanRL rất nhanh.


-------------------

Ok, mình sẽ giải thích kỹ về **$\pi\_\theta$** trong bối cảnh PPO và CleanRL, cả **ý nghĩa lý thuyết** lẫn **format dữ liệu khi chạy code**.

---

## 1️⃣ Ý nghĩa lý thuyết của $\pi\_\theta$

* **$\pi$ (policy)**: hàm ánh xạ từ **state** $s$ sang **phân phối xác suất** trên các **action** $a$.
* **$\theta$**: vector tham số của policy (thường là trọng số mạng neural).
* Với **discrete action space**:

  $$
  \pi_\theta(a|s) \in [0, 1], \quad \sum_{a'} \pi_\theta(a'|s) = 1
  $$

  → Đây là xác suất chọn action $a$ khi ở state $s$.
* Với **continuous action space**:

  * $\pi\_\theta$ thường là **Gaussian policy**: xuất ra **mean** $\mu\_\theta(s)$ và **log-std** $\log \sigma\_\theta(s)$.
  * Sampling: $a \sim \mathcal{N}(\mu\_\theta(s), \sigma\_\theta(s))$.

---

## 2️⃣ Trong code CleanRL PPO

Trong file `ppo.py` của CleanRL, policy thường được implement như một **mạng neural** trả về hai thứ:

```python
# Pseudo-code trong CleanRL PPO
logits, value = policy(obs)
```

* **Discrete env** (`CartPole-v1`, `Atari`, ...):

  * `logits`: tensor shape `(batch_size, n_actions)`
  * Softmax(logits) → $\pi\_\theta(a|s)$
  * Sampling: `Categorical(logits=logits).sample()`
  * Log-prob: `Categorical(logits=logits).log_prob(action)`

* **Continuous env** (`MuJoCo`, `Pendulum`...):

  * Output: `(mean, log_std)` → shape `(batch_size, action_dim)`
  * Sampling: `Normal(mean, std).sample()`
  * Log-prob: `Normal(mean, std).log_prob(action).sum(axis=-1)`

---

## 3️⃣ Format dữ liệu cụ thể

### **Discrete**

```python
# Example: CartPole-v1
obs.shape      # (batch_size, obs_dim)
logits.shape   # (batch_size, n_actions)
probs = torch.softmax(logits, dim=-1)  # π_θ(a|s)
probs[i]       # array size (n_actions), sum = 1
```

Ví dụ:

```
obs[0]       = [0.02, 0.01, 0.05, -0.03]
probs[0]     = [0.1, 0.9]   # π_θ(a=0|s)=0.1, π_θ(a=1|s)=0.9
```

### **Continuous**

```python
# Example: Pendulum-v1
mean.shape     # (batch_size, action_dim)
log_std.shape  # (batch_size, action_dim)
```

Ví dụ:

```
mean[0]        = [0.25]
log_std[0]     = [-0.5]   # std = exp(-0.5) ~ 0.61
a ~ Normal(mean, std)
```

---

## 4️⃣ Vai trò của $\pi\_\theta$ trong PPO Objective

Hàm mục tiêu của PPO:

$$
L^{CLIP}(\theta) =
\mathbb{E}\left[
\min\left(
r_t(\theta)\,\hat{A}_t,
\mathrm{clip}\big(r_t(\theta), 1-\epsilon, 1+\epsilon\big)\,\hat{A}_t
\right)
\right]
$$

Trong đó:

$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

* **$\pi\_\theta(a\_t|s\_t)$**: xác suất của action thực hiện ở state đó **theo policy hiện tại**.
* **$\pi\_{\theta\_{\text{old}}}(a\_t|s\_t)$**: xác suất **theo policy cũ** (lúc rollout).

CleanRL lưu cả log-prob của $\pi\_{\theta\_{\text{old}}}$ để tính ratio này nhanh.

---

Nếu bạn muốn, mình có thể **trace cụ thể trong code CleanRL PPO** xem $\pi\_\theta(a|s)$ được tạo, lưu và dùng ở những dòng nào — từ bước forward network đến tính loss.
Như vậy bạn sẽ thấy rõ **flow dữ liệu của $\pi\_\theta$ trong toàn bộ vòng PPO**.

Bạn có muốn mình trace luôn không?
Mình nghĩ bạn sẽ hiểu cực sâu từ lý thuyết ↔ code.
