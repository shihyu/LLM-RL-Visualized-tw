# 第7部分：策略最佳化架構及衍生演算法

## 【策略最佳化架構演算法及其衍生】Actor-Critic架構
- **Actor-Critic**架構（AC架構），即演員-評委架構，是一種應用極為**廣泛**的強化學習架構。知名的PPO、DPG、DDPG、TD3等演算法均基於Actor-Critic架構。
- （1）**Actor（演員）**：對應於**策略模型**π，負責選擇動作，直接輸出策略π(a | s)，即在給定狀態s下選擇動作a的機率分佈。
- （2）**Critic（評委）**：對應於**價值模型**Q，評估Actor執行的動作的好壞，這可以協助Actor逐步最佳化策略模型的引數。
- 該架構有效**融合**了基於策略（Policy-Based）和基於價值（Value-Based）的方法，結合了兩種方法的優勢，透過同時學習策略和價值函式來提升學習效率與穩定性，是一種混合型方法。
[![【策略最佳化架構演算法及其衍生】Actor-Critic架構](images/%E3%80%90%E7%AD%96%E7%95%A5%E4%BC%98%E5%8C%96%E6%9E%B6%E6%9E%84%E7%AE%97%E6%B3%95%E5%8F%8A%E5%85%B6%E8%A1%8D%E7%94%9F%E3%80%91Actor-Critic%E6%9E%B6%E6%9E%84.png)](images/%E3%80%90%E7%AD%96%E7%95%A5%E4%BC%98%E5%8C%96%E6%9E%B6%E6%9E%84%E7%AE%97%E6%B3%95%E5%8F%8A%E5%85%B6%E8%A1%8D%E7%94%9F%E3%80%91Actor-Critic%E6%9E%B6%E6%9E%84.png)

## 【策略最佳化架構演算法及其衍生】引入基線與優勢（Advantage）函式A的作用
- 基礎版本的Actor-Critic架構存在**高方差**等問題。為了解決這些問題，A2C（Advantage Actor-Critic）方法在Actor-Critic的基礎上引入**基線**（Baseline），並進一步構造了**優勢函式**（Advantage Function）。
- （1）引入**基線**：基線通常採用狀態價值函式V(s)，即在狀態s下的預期回報。將基線設為V(s)，相當於在每個狀態s下設定了一個**平均水平**。
- （2）構造**優勢函式**：優勢函式衡量在給定狀態下，採取某一動作相對於“平均水平”的優劣，即某個動作a相對於特定狀態下其他動作的**相對**優勢。優勢函式關注的是執行某個動作的相對優勢，而非動作的絕對價值。
[![【策略最佳化架構演算法及其衍生】引入基線與優勢函式A的作用](images/%E3%80%90%E7%AD%96%E7%95%A5%E4%BC%98%E5%8C%96%E6%9E%B6%E6%9E%84%E7%AE%97%E6%B3%95%E5%8F%8A%E5%85%B6%E8%A1%8D%E7%94%9F%E3%80%91%E5%BC%95%E5%85%A5%E5%9F%BA%E7%BA%BF%E4%B8%8E%E4%BC%98%E5%8A%BF%E5%87%BD%E6%95%B0A%E7%9A%84%E4%BD%9C%E7%94%A8.png)](images/%E3%80%90%E7%AD%96%E7%95%A5%E4%BC%98%E5%8C%96%E6%9E%B6%E6%9E%84%E7%AE%97%E6%B3%95%E5%8F%8A%E5%85%B6%E8%A1%8D%E7%94%9F%E3%80%91%E5%BC%95%E5%85%A5%E5%9F%BA%E7%BA%BF%E4%B8%8E%E4%BC%98%E5%8A%BF%E5%87%BD%E6%95%B0A%E7%9A%84%E4%BD%9C%E7%94%A8.png)

## 【策略最佳化架構演算法及其衍生】GAE（廣義優勢估計,Generalized Advantage Estimation）演算法
- **GAE**（Generalized Advantage Estimation, **廣義優勢估計**）演算法由**RL巨佬John Schulman**等發表。該演算法是**PPO**等演算法的關鍵組成部分。
- GAE演算法**借鑑**了TD(λ)演算法的思路，（TD(λ)如《大型模型演算法》5.3.3節所述）。TD(λ)演算法透過調節λ因子在偏差和方差之間取得平衡，GAE演算法也使用λ因子以達到類似的目的。
- GAE演算法計算時，通常採用**遞迴**求解的方式進行。
- GAE演算法的虛擬碼如下。演算法 6.1 GAE 演算法核心實現

```python
import numpy as np

def compute_gae(rewards, values, gamma=0.99, lambda_=0.95):
    """
    引數:
        rewards (list 或 np.ndarray): 每個時間步收集到的獎勵 r，形狀為 (T,)
        values  (list 或 np.ndarray): 每個狀態的價值估計 V，形狀為 (T+1,)
        gamma   (float): 折扣因子 γ
        lambda_ (float): GAE 的衰減引數 λ
    返回:
        np.ndarray: 優勢估計 A，形狀為 (T,)。例如對於 T=5，A=[A0, A1, A2, A3, A4]
    """
    T = len(rewards)            # 時間步數 T，終止時間步為 t=T-1
    advantages = np.zeros(T)    # 優勢估計陣列，例如 [A0, A1, A2, A3, A4]
    gae = 0                     # 初始化 GAE 累計值為 0

    # 反向從時間步 t=T-1 到 t=0 進行迭代計算，總共迭代 T 次
    for t in reversed(range(T)):
        # δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        # A_t = δ_t + γ * λ * A_{t+1}
        gae = delta + gamma * lambda_ * gae
        advantages[t] = gae

    return advantages
```

[![【策略最佳化架構演算法及其衍生】GAE（廣義優勢估計）演算法](images/%E3%80%90%E7%AD%96%E7%95%A5%E4%BC%98%E5%8C%96%E6%9E%B6%E6%9E%84%E7%AE%97%E6%B3%95%E5%8F%8A%E5%85%B6%E8%A1%8D%E7%94%9F%E3%80%91GAE%EF%BC%88%E5%B9%BF%E4%B9%89%E4%BC%98%E5%8A%BF%E4%BC%B0%E8%AE%A1%EF%BC%89%E7%AE%97%E6%B3%95.png)](images/%E3%80%90%E7%AD%96%E7%95%A5%E4%BC%98%E5%8C%96%E6%9E%B6%E6%9E%84%E7%AE%97%E6%B3%95%E5%8F%8A%E5%85%B6%E8%A1%8D%E7%94%9F%E3%80%91GAE%EF%BC%88%E5%B9%BF%E4%B9%89%E4%BC%98%E5%8A%BF%E4%BC%B0%E8%AE%A1%EF%BC%89%E7%AE%97%E6%B3%95.png)

## 【策略最佳化架構演算法及其衍生】PPO（Proximal Policy Optimization）演算法的演進
- **PPO**（Proximal Policy Optimization，近端策略最佳化），有多種**變體**，例如PPO-Penalty和PPO-Clip，這些演算法繼承了部分**TRPO**演算法的思想。
- **PPO-Clip**因其更優的效果而獲得了更多關注和應用，因此**通常**所說的PPO即指**PPO-Clip**。
[![【策略最佳化架構演算法及其衍生】PPO演算法的演進](images/%E3%80%90%E7%AD%96%E7%95%A5%E4%BC%98%E5%8C%96%E6%9E%B6%E6%9E%84%E7%AE%97%E6%B3%95%E5%8F%8A%E5%85%B6%E8%A1%8D%E7%94%9F%E3%80%91PPO%E7%AE%97%E6%B3%95%E7%9A%84%E6%BC%94%E8%BF%9B.png)](images/%E3%80%90%E7%AD%96%E7%95%A5%E4%BC%98%E5%8C%96%E6%9E%B6%E6%9E%84%E7%AE%97%E6%B3%95%E5%8F%8A%E5%85%B6%E8%A1%8D%E7%94%9F%E3%80%91PPO%E7%AE%97%E6%B3%95%E7%9A%84%E6%BC%94%E8%BF%9B.png)

## 【策略最佳化架構演算法及其衍生】TRPO（Trust Region Policy Optimization）及其置信域
- **TRPO**（Trust Region Policy Optimization，置信域策略最佳化），可以說是**PPO的前身**。
- 該演算法是對策略梯度演算法的改進，基於**兩個核心**概念：置信域和重要性取樣（Importance Sampling）。雖然這兩個概念並非在TRPO中首次提出，但TRPO將它們與策略梯度演算法相結合，顯著提升了演算法的效果。
- TRPO的核心**思想**是在最大化目標函式J(θ)的同時，限制新舊策略之間的差異。
[![【策略最佳化架構演算法及其衍生】TRPO及其置信域](images/%E3%80%90%E7%AD%96%E7%95%A5%E4%BC%98%E5%8C%96%E6%9E%B6%E6%9E%84%E7%AE%97%E6%B3%95%E5%8F%8A%E5%85%B6%E8%A1%8D%E7%94%9F%E3%80%91TRPO%E5%8F%8A%E5%85%B6%E7%BD%AE%E4%BF%A1%E5%9F%9F.png)](images/%E3%80%90%E7%AD%96%E7%95%A5%E4%BC%98%E5%8C%96%E6%9E%B6%E6%9E%84%E7%AE%97%E6%B3%95%E5%8F%8A%E5%85%B6%E8%A1%8D%E7%94%9F%E3%80%91TRPO%E5%8F%8A%E5%85%B6%E7%BD%AE%E4%BF%A1%E5%9F%9F.png)

## 【策略最佳化架構演算法及其衍生】重要性取樣（Importance sampling）
- 重要性取樣在諸如TRPO和PPO等強化學習演算法中具有關鍵作用，其主要功能是**修正**新舊策略之間的分佈差異，使得可以利用舊策略所採集的資料來最佳化新策略。
- **重要性取樣**（Importance Sampling）是一種基於**蒙特卡洛**取樣思想的方法，經常被用於估計期望值和積分。該方法基於輔助分佈進行取樣，並透過重要性權重對估計值進行修正，從而提高取樣和估計的效率。
- 重要性取樣需滿足該**條件**：如果在某個x上p(x)的機率值大於0，那麼p'(x)在同一個位置的機率也必須大於0（p'(x)不能為零，意味著該位置在p'(x)中也有可能性）。這是**因為**如果在p(x)非零的地方p'(x)= 0，那麼在這些x上的貢獻就無法透過p'(x)來取樣計算，一些x的取值會在積分中丟失。
[![【策略最佳化架構演算法及其衍生】重要性取樣](images/%E3%80%90%E7%AD%96%E7%95%A5%E4%BC%98%E5%8C%96%E6%9E%B6%E6%9E%84%E7%AE%97%E6%B3%95%E5%8F%8A%E5%85%B6%E8%A1%8D%E7%94%9F%E3%80%91%E9%87%8D%E8%A6%81%E6%80%A7%E9%87%87%E6%A0%B7.png)](images/%E3%80%90%E7%AD%96%E7%95%A5%E4%BC%98%E5%8C%96%E6%9E%B6%E6%9E%84%E7%AE%97%E6%B3%95%E5%8F%8A%E5%85%B6%E8%A1%8D%E7%94%9F%E3%80%91%E9%87%8D%E8%A6%81%E6%80%A7%E9%87%87%E6%A0%B7.png)

## 【策略最佳化架構演算法及其衍生】PPO-Clip
- 通常所說的PPO即指PPO-Clip（近端策略最佳化-剪裁）。
- PO-Clip的**目標**是最大化未來回報的期望，具體來說，透過最大化目標函式J(θ)來最佳化策略，PPO-Clip的目標函式如下圖。
- clip與min操作的**意義**：意義如下圖的曲線圖，為清晰起見，對式中的部分項用兩種縮放係數進行替換，可以分別稱為“線性”縮放係數和“剪裁”縮放**係數**。

[![【策略最佳化架構演算法及其衍生】PPO-Clip](images/%E3%80%90%E7%AD%96%E7%95%A5%E4%BC%98%E5%8C%96%E6%9E%B6%E6%9E%84%E7%AE%97%E6%B3%95%E5%8F%8A%E5%85%B6%E8%A1%8D%E7%94%9F%E3%80%91PPO-Clip.png)](images/%E3%80%90%E7%AD%96%E7%95%A5%E4%BC%98%E5%8C%96%E6%9E%B6%E6%9E%84%E7%AE%97%E6%B3%95%E5%8F%8A%E5%85%B6%E8%A1%8D%E7%94%9F%E3%80%91PPO-Clip.png)

## 【策略最佳化架構演算法及其衍生】PPO訓練中策略模型的更新過程
- 如圖簡要展示了通用場景下PPO的訓練流程及策略模型（Actor）引數θ的更新過程，其中價值模型（Critic）未在圖中展示。PPO的訓練流程主要包括以下**兩個階段**：
- （1）**樣本收集**：基於舊策略收集樣本，生成多條軌跡（經驗），並存入**回放緩衝區**，供後續訓練使用。
- （2）**多輪PPO訓練**：將回放緩衝區中的所有樣本隨機打散，並劃分為多個**小批次**（mini-batches），以便進行小批次訓練。針對每個小批次（圖中的“批次 1”、“批次 2”），分別進行一次訓練與引數更新，總計完成mini_batch次訓練。如果設定的ppo_epochs > 1，則重複利用回放緩衝區中的所有樣本，再次隨機打散並切分為小批次，重複上述訓練過程ppo_epochs輪（圖中的“第 1 輪”、“第 2 輪”）。透過對這一大批次樣本的多輪訓練，顯著提升了樣本的重複利用率。在第二階段中，總計完成ppo_epochs×mini_batch次訓練與引數更新。
- 以上兩個階段不斷**迴圈**執行，每一次迴圈稱為一個迭代（iteration）。經過多次迭代，直至完成所有PPO訓練任務。
[![【策略最佳化架構演算法及其衍生】PPO訓練中策略模型的更新過程](images/%E3%80%90%E7%AD%96%E7%95%A5%E4%BC%98%E5%8C%96%E6%9E%B6%E6%9E%84%E7%AE%97%E6%B3%95%E5%8F%8A%E5%85%B6%E8%A1%8D%E7%94%9F%E3%80%91PPO%E8%AE%AD%E7%BB%83%E4%B8%AD%E7%AD%96%E7%95%A5%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%9B%B4%E6%96%B0%E8%BF%87%E7%A8%8B.png)](images/%E3%80%90%E7%AD%96%E7%95%A5%E4%BC%98%E5%8C%96%E6%9E%B6%E6%9E%84%E7%AE%97%E6%B3%95%E5%8F%8A%E5%85%B6%E8%A1%8D%E7%94%9F%E3%80%91PPO%E8%AE%AD%E7%BB%83%E4%B8%AD%E7%AD%96%E7%95%A5%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%9B%B4%E6%96%B0%E8%BF%87%E7%A8%8B.png)

## 【策略最佳化架構演算法及其衍生】PPO的虛擬碼
```python
### 簡寫 R:rewards, V:values, Adv:advantages, J:objective, P:probability
for iteration in range(num_iterations):  # 進行num_iterations個訓練迭代
    #【1/2】收集樣本(prompt, response_old, logP_old, Adv, V_target)
    prompt_batch, response_old_batch = [], []
    logP_old_batch, Adv_batch, V_target_batch = [], [], []
    for _ in range(num_examples):
        logP_old, response_old  = actor_model(prompt)
        V_old    = critic_model(prompt, response_old)
        R        = reward_model(prompt, response_old)[-1]
        logP_ref = ref_model(prompt, response_old)
        
        # KL距離懲罰。注意：上面的R只取了最後一個token對應的獎勵分數
        KL = logP_old - logP_ref
        R_with_KL = R - scale_factor * KL

        # 透過GAE演算法計算優勢Adv
        Adv = GAE_Advantage(R_with_KL, V_old, gamma, λ)
        V_target = Adv + V_old

        prompt_batch        += prompt
        response_old_batch  += response_old
        logP_old_batch      += logP_old
        Adv_batch           += Adv
        V_target_batch      += V_target

    # 【2/2】多輪PPO訓練，多次引數更新
    for _ in range(ppo_epochs):
        mini_batches = shuffle_split( (prompt_batch, response_old_batch, 
            logP_old_batch, Adv_batch, V_target_batch), mini_batch_size )
        
        for prompt, response_old, logP_old, Adv, V_target in mini_batches:
            logits, logP_new = actor_model(prompt, response_old)
            V_new            = critic_model(prompt, response_old)

            # 策略機率比: ratio(θ) = π_θ(a|s) / π_θ_old (a|s)
            ratios = exp(logP_new - logP_old)

            # 計算策略模型Loss
            L_clip = -mean( min( ratios * Adv,
                                clip(ratios, 1 - ε, 1 + ε) * Adv ) )
            
            S_entropy = mean( compute_entropy(logits) )  # 計算策略的熵

            Loss_V = mean((V_new - V_target) ** 2)   # 計算價值模型Loss

            Loss = L_clip + C1 * Loss_V - C2 * S_entropy # 總損失
            backward_update(Loss, L_clip, Loss_V) # 反向傳播; 更新模型引數
```

## 【策略最佳化架構演算法及其衍生】PPO與GRPO（Group Relative Policy Optimization） <sup>[<a href="references.md">72</a>]</sup>
- **GRPO**（Group Relative Policy Optimization, 群體相對策略最佳化）是一種基於策略的強化學習演算法，由**DeepSeek**團隊提出，並已在DeepSeek、Qwen等模型的訓練中得到應用。
- 傳統的PPO方法除了訓練策略模型外，還需額外構建一個規模相近的價值網路，這會顯著增加計算和視訊記憶體的**開銷**。
- 如圖所示，GRPO**摒棄**了單獨的價值網路，並透過多項改進，在保留PPO核心思想的基礎上，顯著降低了訓練所需資源，同時確保了策略更新的高效性和穩定性。
- GRPO的**核心思想**在於利用群體相對優勢估計來取代傳統的價值模型。具體來說，GRPO透過取樣一組候選輸出，並將這些輸出的平均獎勵作為基線，來計算各個輸出的優勢值。這種方法不僅避免了對額外價值模型的依賴，同時也充分發揮了獎勵模型的比較特性，從而提高了訓練的效率和穩定性。
[![【策略最佳化架構演算法及其衍生】GRPO&PPO](images/%E3%80%90%E7%AD%96%E7%95%A5%E4%BC%98%E5%8C%96%E6%9E%B6%E6%9E%84%E7%AE%97%E6%B3%95%E5%8F%8A%E5%85%B6%E8%A1%8D%E7%94%9F%E3%80%91GRPO%26PPO.png)](images/%E3%80%90%E7%AD%96%E7%95%A5%E4%BC%98%E5%8C%96%E6%9E%B6%E6%9E%84%E7%AE%97%E6%B3%95%E5%8F%8A%E5%85%B6%E8%A1%8D%E7%94%9F%E3%80%91GRPO%26PPO.png)

## 【策略最佳化架構演算法及其衍生】確定性策略vs隨機性策略（Deterministic policy vs. Stochastic policy）
- 強化學習的策略型別可以**分為**：確定性策略（Deterministic Policy）和隨機性策略（Stochastic Policy）。
- 確定性策略：在每個狀態下，策略都會輸出一個**確定**的動作。
- 隨機性策略：在每個狀態下，策略輸出的是動作的**機率分佈**。
[![【策略最佳化架構演算法及其衍生】確定性策略vs隨機性策略](images/%E3%80%90%E7%AD%96%E7%95%A5%E4%BC%98%E5%8C%96%E6%9E%B6%E6%9E%84%E7%AE%97%E6%B3%95%E5%8F%8A%E5%85%B6%E8%A1%8D%E7%94%9F%E3%80%91%E7%A1%AE%E5%AE%9A%E6%80%A7%E7%AD%96%E7%95%A5vs%E9%9A%8F%E6%9C%BA%E6%80%A7%E7%AD%96%E7%95%A5.png)](images/%E3%80%90%E7%AD%96%E7%95%A5%E4%BC%98%E5%8C%96%E6%9E%B6%E6%9E%84%E7%AE%97%E6%B3%95%E5%8F%8A%E5%85%B6%E8%A1%8D%E7%94%9F%E3%80%91%E7%A1%AE%E5%AE%9A%E6%80%A7%E7%AD%96%E7%95%A5vs%E9%9A%8F%E6%9C%BA%E6%80%A7%E7%AD%96%E7%95%A5.png)

## 【策略最佳化架構演算法及其衍生】確定性策略梯度（DPG）
- **確定性策略梯度**（Deterministic Policy Gradient, **DPG**）演算法由**RL大佬David Silver**在DeepMind期間與其他研究者於2014年發表的論文中系統性闡述，該演算法旨在解決連續動作空間中的強化學習問題。
- DPG採用確定性策略，將狀態直接對映到具體的動作。
- DPG採用了演員-評委（Actor-Critic）架構。
[![【策略最佳化架構演算法及其衍生】確定性策略梯度（DPG）](images/%E3%80%90%E7%AD%96%E7%95%A5%E4%BC%98%E5%8C%96%E6%9E%B6%E6%9E%84%E7%AE%97%E6%B3%95%E5%8F%8A%E5%85%B6%E8%A1%8D%E7%94%9F%E3%80%91%E7%A1%AE%E5%AE%9A%E6%80%A7%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6%EF%BC%88DPG%EF%BC%89.png)](images/%E3%80%90%E7%AD%96%E7%95%A5%E4%BC%98%E5%8C%96%E6%9E%B6%E6%9E%84%E7%AE%97%E6%B3%95%E5%8F%8A%E5%85%B6%E8%A1%8D%E7%94%9F%E3%80%91%E7%A1%AE%E5%AE%9A%E6%80%A7%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6%EF%BC%88DPG%EF%BC%89.png)

## 【策略最佳化架構演算法及其衍生】DDPG（Deep Deterministic Policy Gradient）
- **深度確定性策略梯度**（Deep Deterministic Policy Gradient, **DDPG**）是確定性策略梯度（DPG）演算法的改進版本，**結合**了深度Q網路（DQN）的思想，由DeepMind的研究團隊發表。
- DDPG在訓練過程中需要載入**4個模型**——包括2個策略網路（Actor）和2個評估網路（Critic）。
- 知名的**TD3** ：雙延遲深度確定性策略梯度（Twin Delayed Deep Deterministic Policy Gradient, TD3）演算法是對DDPG（Deep Deterministic Policy Gradient）的顯著改進。在TD3的訓練過程中，需要載入**6個模型**——包括2個策略網路（Actor）和4個評估網路（Critic）。這些網路分工明確、相互配合，透過延遲更新目標網路等技巧，有效提升了訓練的穩定性和效能。
[![【策略最佳化架構演算法及其衍生】DDPG](images/%E3%80%90%E7%AD%96%E7%95%A5%E4%BC%98%E5%8C%96%E6%9E%B6%E6%9E%84%E7%AE%97%E6%B3%95%E5%8F%8A%E5%85%B6%E8%A1%8D%E7%94%9F%E3%80%91DDPG.png)](images/%E3%80%90%E7%AD%96%E7%95%A5%E4%BC%98%E5%8C%96%E6%9E%B6%E6%9E%84%E7%AE%97%E6%B3%95%E5%8F%8A%E5%85%B6%E8%A1%8D%E7%94%9F%E3%80%91DDPG.png)
