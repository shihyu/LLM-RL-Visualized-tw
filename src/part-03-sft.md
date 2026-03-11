# 第3部分：SFT（有監督微調）

## 【SFT】微調（Fine-Tuning）技術分類
- 可用於SFT的微調技術種類多樣，如下圖的分類圖所示：前兩種方法僅需基於預訓練模型主體進行微調，開發成本較低；而並聯低秩微調和Adapter Tuning則需要**引入額外**的新模組，實施過程相對複雜一些。這些方法均是針對模型引數進行微調，而基於Prompt的微調則另闢蹊徑，從**模型的輸入**著手進行微調。
[![【SFT】微調技術分類](images/%E3%80%90SFT%E3%80%91%E5%BE%AE%E8%B0%83%E6%8A%80%E6%9C%AF%E5%88%86%E7%B1%BB.png)](images/%E3%80%90SFT%E3%80%91%E5%BE%AE%E8%B0%83%E6%8A%80%E6%9C%AF%E5%88%86%E7%B1%BB.png)

## 【SFT】LoRA（1 of 2）
- **LoRA**（Low-Rank Adaptation, 低秩適配微調）由微軟的研究團隊於2021年發表。由於其高效的微調方式和良好的效果，在各類模型中得到廣泛的應用。LoRA的**核心思想**在於——微調前後模型的引數差異∆W具有低秩性。
- 一個矩陣具有**低秩性**，意味著它包含較多的冗餘資訊，將其分解為多個低維矩陣後，不會損失過多有用資訊。例如，如圖所示，一個1024×1024的矩陣可以被近似分解為一個1024×2矩陣與一個2×1024矩陣的乘積，從而將引數量約減少至原來的0.4%。

[![【SFT】LoRA（1 of 2）](images/%E3%80%90SFT%E3%80%91LoRA%EF%BC%881%20of%202%EF%BC%89.png)](images/%E3%80%90SFT%E3%80%91LoRA%EF%BC%881%20of%202%EF%BC%89.png)

## 【SFT】LoRA（2 of 2）
- A和B的**初始化**方式：
- （1）矩陣A使用隨機數初始化，如kaiming初始化；
- （2）矩陣B使用全0初始化，或使用極小的隨機數初始化。
- **目的**在於——確保在訓練初期，插入的LoRA模組不會對模型整體輸出造成過大的擾動。
[![【SFT】LoRA（2 of 2）](images/%E3%80%90SFT%E3%80%91LoRA%EF%BC%882%20of%202%EF%BC%89.png)](images/%E3%80%90SFT%E3%80%91LoRA%EF%BC%882%20of%202%EF%BC%89.png)

## 【SFT】Prefix-Tuning
- **Prefix-Tuning**由斯坦福大學的研究團隊提出，可以用於對語言模型進行輕量級微調。
- 如圖所示，該方法在輸入的起始位置插入一段連續的、可訓練的向量（Embedding），稱為字首（Prefix）。在處理後續的Token時，Transformer將這些向量視為 **虛擬Token**，並參與Attention計算。

[![【SFT】Prefix-Tuning](images/%E3%80%90SFT%E3%80%91Prefix-Tuning.png)](images/%E3%80%90SFT%E3%80%91Prefix-Tuning.png)

## 【SFT】TokenID與詞元的對映關係
- 以SFT資料（經過ChatML格式預處理）為例，該資料經過Tokenizer處理後，被轉換為33個Token，對應33個序列位置，如圖所示。
- 每個Token ID與相應的詞元**一一對應**。
[![【SFT】TokenID與詞元的對映關係](images/%E3%80%90SFT%E3%80%91TokenID%E4%B8%8E%E8%AF%8D%E5%85%83%E7%9A%84%E6%98%A0%E5%B0%84%E5%85%B3%E7%B3%BB.png)](images/%E3%80%90SFT%E3%80%91TokenID%E4%B8%8E%E8%AF%8D%E5%85%83%E7%9A%84%E6%98%A0%E5%B0%84%E5%85%B3%E7%B3%BB.png)

## 【SFT】SFT的Loss（交叉熵）
- 與預訓練階段類似，SFT的Loss也是基於交叉熵（CE，Cross Entropy）
[![【SFT】SFT的Loss（交叉熵）](images/%E3%80%90SFT%E3%80%91SFT%E7%9A%84Loss%EF%BC%88%E4%BA%A4%E5%8F%89%E7%86%B5%EF%BC%89.png)](images/%E3%80%90SFT%E3%80%91SFT%E7%9A%84Loss%EF%BC%88%E4%BA%A4%E5%8F%89%E7%86%B5%EF%BC%89.png)

## 【SFT】指令資料的來源
- **指令資料**（Instructions）是指為模型提供的一組引導性輸入及期望輸出的描述，通常包括問題（或任務提示）及其對應的答案，常用於對模型進行微調訓練。
[![【SFT】指令資料的來源](images/%E3%80%90SFT%E3%80%91%E6%8C%87%E4%BB%A4%E6%95%B0%E6%8D%AE%E7%9A%84%E6%9D%A5%E6%BA%90.png)](images/%E3%80%90SFT%E3%80%91%E6%8C%87%E4%BB%A4%E6%95%B0%E6%8D%AE%E7%9A%84%E6%9D%A5%E6%BA%90.png)

## 【SFT】多個數據的拼接（Packing）
- 模型訓練時通常使用**固定長度**的輸入。當輸入的資料長度不一致時，對於短序列，會在其末尾填充（padding）以匹配最大序列長度，這會導致計算資源的**浪費**。
- 因此，正如圖所示，常見的做法是將多條資料拼接（Packing）在一起，填充到一個固定長度的輸入序列中。
- 為了確保計算過程中不同資料之間**互不干擾**，通常需要**重設**位置編號（Reset position ID）和注意力掩碼（Reset attention mask），以在計算Attention時保持各條資料的語義獨立性。
[![【SFT】多個數據的拼接（Packing）](images/%E3%80%90SFT%E3%80%91%E5%A4%9A%E4%B8%AA%E6%95%B0%E6%8D%AE%E7%9A%84%E6%8B%BC%E6%8E%A5%EF%BC%88Packing%EF%BC%89.png)](images/%E3%80%90SFT%E3%80%91%E5%A4%9A%E4%B8%AA%E6%95%B0%E6%8D%AE%E7%9A%84%E6%8B%BC%E6%8E%A5%EF%BC%88Packing%EF%BC%89.png)
