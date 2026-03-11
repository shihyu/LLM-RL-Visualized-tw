# 第4部分：DPO（直接偏好最佳化）

## 【DPO】RLHF與DPO的訓練架構對比
不同於RLHF，DPO以監督學習的訓練方式，大幅簡化了對齊訓練：
- **流程簡潔**：DPO直接對策略模型進行最佳化，不需要預先訓練Reward模型（獎勵函式）。DPO只需要基於預先給定的偏好資料進行訓練，**無需**中途取樣。
- **穩定性**：DPO是一種監督學習方法，擺脫了強化學習訓練的不穩定性。
- **低開銷**：DPO在訓練過程中只需要載入**一個模型**（只需載入策略模型，而對於參考模型，可以將參考模型的輸出結果預先錄製好，然後在訓練時就不需要載入），算力開銷更低，更易於落地實踐。
[![【DPO】RLHF與DPO的訓練架構對比](images/%E3%80%90DPO%E3%80%91RLHF%E4%B8%8EDPO%E7%9A%84%E8%AE%AD%E7%BB%83%E6%9E%B6%E6%9E%84%E5%AF%B9%E6%AF%94.png)](images/%E3%80%90DPO%E3%80%91RLHF%E4%B8%8EDPO%E7%9A%84%E8%AE%AD%E7%BB%83%E6%9E%B6%E6%9E%84%E5%AF%B9%E6%AF%94.png)

## 【DPO】Prompt的收集
[![【DPO】Prompt的收集](images/%E3%80%90DPO%E3%80%91Prompt%E7%9A%84%E6%94%B6%E9%9B%86.png)](images/%E3%80%90DPO%E3%80%91Prompt%E7%9A%84%E6%94%B6%E9%9B%86.png)

## 【DPO】DPO（Direct Preference Optimization）
- **DPO**（Direct Preference Optimization，直接偏好最佳化）是由斯坦福大學等研究團隊於2023年提出的一種偏好最佳化演算法，可用於LLM、VLM與MLLM的對齊訓練。
- 該演算法在基於PPO的RLHF基礎上進行了大幅**簡化**。
- DPO演算法跳過了訓練獎勵模型這一中間過程，直接（Direct）最佳化策略模型 ——這正是DPO命名中“D（Direct）”的**含義**所在。
DPO的訓練涉及2個模型——策略模型和參考模型。它們的**初始化**方法如下：
- **策略模型**：直接複製SFT模型作為初始化。
- **參考模型**：通常也從SFT模型複製。但在某些情況下，可能會選擇從一個比SFT模型更強的模型進行復制。此時，需特別關注參考模型與策略模型的匹配性，主要涉及兩者的KL距離及訓練資料分佈等方面。
[![【DPO】DPO（Direct Preference Optimization）](images/%E3%80%90DPO%E3%80%91DPO%EF%BC%88Direct%20Preference%20Optimization%EF%BC%89.png)](images/%E3%80%90DPO%E3%80%91DPO%EF%BC%88Direct%20Preference%20Optimization%EF%BC%89.png)

## 【DPO】DPO訓練全景圖
- DPO訓練時，可以選擇**載入**2個模型（策略模型 和參考模型 ），也可以只加載1個模型（策略模型 ）。本文將優先分析載入2個模型的情況。
- DPO的整體訓練流程如圖所示（**藍色**色塊代表偏好資料對中的“優質回答” 及其對應的中間計算結果；**粉色**色塊代表偏好資料對中的“劣質回答” 及其對應的中間計算結果）。
[![【DPO】DPO訓練全景圖](images/%E3%80%90DPO%E3%80%91DPO%E8%AE%AD%E7%BB%83%E5%85%A8%E6%99%AF%E5%9B%BE.png)](images/%E3%80%90DPO%E3%80%91DPO%E8%AE%AD%E7%BB%83%E5%85%A8%E6%99%AF%E5%9B%BE.png)

## 【DPO】β引數對DPO的影響
- 在DPO中， 引數的作用**類似**於其在RLHF中的作用。
[![【DPO】β引數對DPO的影響](images/%E3%80%90DPO%E3%80%91%CE%B2%E5%8F%82%E6%95%B0%E5%AF%B9DPO%E7%9A%84%E5%BD%B1%E5%93%8D.png)](images/%E3%80%90DPO%E3%80%91%CE%B2%E5%8F%82%E6%95%B0%E5%AF%B9DPO%E7%9A%84%E5%BD%B1%E5%93%8D.png)

## 【DPO】隱式獎勵差異對引數更新幅度的影響
- DPO的梯度更新旨在**增加**優質回答的機率，同時**減少**劣質回答的機率。
- 更重要的是，梯度中包含一個**動態係數**——優質和劣質回答的隱式獎勵差異。換言之，這個動態係數反映了隱式“獎勵模型”在對偏好順序的判斷上有多大誤差。
[![【DPO】隱式獎勵差異對引數更新幅度的影響](images/%E3%80%90DPO%E3%80%91%E9%9A%90%E5%BC%8F%E5%A5%96%E5%8A%B1%E5%B7%AE%E5%BC%82%E5%AF%B9%E5%8F%82%E6%95%B0%E6%9B%B4%E6%96%B0%E5%B9%85%E5%BA%A6%E7%9A%84%E5%BD%B1%E5%93%8D.png)](images/%E3%80%90DPO%E3%80%91%E9%9A%90%E5%BC%8F%E5%A5%96%E5%8A%B1%E5%B7%AE%E5%BC%82%E5%AF%B9%E5%8F%82%E6%95%B0%E6%9B%B4%E6%96%B0%E5%B9%85%E5%BA%A6%E7%9A%84%E5%BD%B1%E5%93%8D.png)
