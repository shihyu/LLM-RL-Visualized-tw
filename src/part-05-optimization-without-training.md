# 第5部分：免訓練的大型模型最佳化技術

## 【免訓練的最佳化技術】CoT（Chain of Thought）與傳統問答的對比
- **CoT**（Chain of Thought，**思維鏈**）由Jason Wei等研究者於2022年在Google期間提出，是大型模型領域的一項重要技術創新，並迅速被廣泛應用於學術研究和實際場景。
- 其**核心思想**是透過顯式地分步推理，提升模型在複雜推理任務中的表現。
[![【免訓練的最佳化技術】CoT與傳統問答的對比](images/%E3%80%90%E5%85%8D%E8%AE%AD%E7%BB%83%E7%9A%84%E4%BC%98%E5%8C%96%E6%8A%80%E6%9C%AF%E3%80%91CoT%E4%B8%8E%E4%BC%A0%E7%BB%9F%E9%97%AE%E7%AD%94%E7%9A%84%E5%AF%B9%E6%AF%94.png)](images/%E3%80%90%E5%85%8D%E8%AE%AD%E7%BB%83%E7%9A%84%E4%BC%98%E5%8C%96%E6%8A%80%E6%9C%AF%E3%80%91CoT%E4%B8%8E%E4%BC%A0%E7%BB%9F%E9%97%AE%E7%AD%94%E7%9A%84%E5%AF%B9%E6%AF%94.png)

## 【免訓練的最佳化技術】CoT、Self-consistency CoT、ToT、GoT <sup>[<a href="references.md">87</a>]</sup>
- 在CoT展現其潛力後，迅速**衍生**出多種相關技術，例如ToT、GoT、Self-consistency CoT、Zero-shot-CoT、Auto-CoT、MoT、XoT等
[![【免訓練的最佳化技術】CoT、Self-consistencyCoT、ToT、GoT](images/%E3%80%90%E5%85%8D%E8%AE%AD%E7%BB%83%E7%9A%84%E4%BC%98%E5%8C%96%E6%8A%80%E6%9C%AF%E3%80%91CoT%E3%80%81Self-consistencyCoT%E3%80%81ToT%E3%80%81GoT.png)](images/%E3%80%90%E5%85%8D%E8%AE%AD%E7%BB%83%E7%9A%84%E4%BC%98%E5%8C%96%E6%8A%80%E6%9C%AF%E3%80%91CoT%E3%80%81Self-consistencyCoT%E3%80%81ToT%E3%80%81GoT.png)

## 【免訓練的最佳化技術】窮舉搜尋（Exhaustive Search）
- Token的生成過程可以形象地表示為一個以詞表大小 V=105 為基數的**多叉樹**結構。理論上，採用窮舉搜尋（Exhaustive Search）的方法能夠獲得全域性最優解。然而，窮舉搜尋的計算代價極高。
[![【免訓練的最佳化技術】窮舉搜尋（Exhaustive Search）](images/%E3%80%90%E5%85%8D%E8%AE%AD%E7%BB%83%E7%9A%84%E4%BC%98%E5%8C%96%E6%8A%80%E6%9C%AF%E3%80%91%E7%A9%B7%E4%B8%BE%E6%90%9C%E7%B4%A2%EF%BC%88Exhaustive%20Search%EF%BC%89.png)](images/%E3%80%90%E5%85%8D%E8%AE%AD%E7%BB%83%E7%9A%84%E4%BC%98%E5%8C%96%E6%8A%80%E6%9C%AF%E3%80%91%E7%A9%B7%E4%B8%BE%E6%90%9C%E7%B4%A2%EF%BC%88Exhaustive%20Search%EF%BC%89.png)

## 【免訓練的最佳化技術】貪婪搜尋（Greedy Search）
- **貪婪搜尋**（Greedy Search）在生成下一個Token（詞元）時，每次都會選擇當前**機率最高**的一個Token，不考慮生成序列的全域性最優性或多樣性，然後繼續對下一個Token位置執行相同的操作。儘管這種方法簡單快速，但生成的內容可能過早陷入區域性最優，**缺乏多樣性**。
[![【免訓練的最佳化技術】貪婪搜尋（Greedy Search）](images/%E3%80%90%E5%85%8D%E8%AE%AD%E7%BB%83%E7%9A%84%E4%BC%98%E5%8C%96%E6%8A%80%E6%9C%AF%E3%80%91%E8%B4%AA%E5%A9%AA%E6%90%9C%E7%B4%A2%EF%BC%88Greedy%20Search%EF%BC%89.png)](images/%E3%80%90%E5%85%8D%E8%AE%AD%E7%BB%83%E7%9A%84%E4%BC%98%E5%8C%96%E6%8A%80%E6%9C%AF%E3%80%91%E8%B4%AA%E5%A9%AA%E6%90%9C%E7%B4%A2%EF%BC%88Greedy%20Search%EF%BC%89.png)

## 【免訓練的最佳化技術】波束搜尋（Beam Search）
- **Beam Search**（波束搜尋）在每一步生成時，不僅僅選擇一個最優Token（詞元），而是保留**多個候選**序列（稱為Beam，即波束），其餘的路徑則被剪枝。這些候選序列會在後續步驟中繼續擴充套件，直到生成結束。
- 最終，從所有候選序列中選擇得分**最高**的一條作為最終輸出。Beam的數量（num_beams引數）越大，搜尋空間越廣，生成結果越**接近全域性最優**，但計算成本也隨之增加。
[![【免訓練的最佳化技術】波束搜尋（Beam Search）](images/%E3%80%90%E5%85%8D%E8%AE%AD%E7%BB%83%E7%9A%84%E4%BC%98%E5%8C%96%E6%8A%80%E6%9C%AF%E3%80%91%E6%B3%A2%E6%9D%9F%E6%90%9C%E7%B4%A2%EF%BC%88Beam%20Search%EF%BC%89.png)](images/%E3%80%90%E5%85%8D%E8%AE%AD%E7%BB%83%E7%9A%84%E4%BC%98%E5%8C%96%E6%8A%80%E6%9C%AF%E3%80%91%E6%B3%A2%E6%9D%9F%E6%90%9C%E7%B4%A2%EF%BC%88Beam%20Search%EF%BC%89.png)

## 【免訓練的最佳化技術】多項式取樣（Multinomial Sampling）
- **多項式取樣**（Multinomial Sampling）是生成式模型中一種常見的**隨機取樣**方法，生成下一個Token時，以模型預測的機率分佈為依據，在機率分佈中“按機率大小”隨機抽取Token（而非等機率隨機抽樣）。
- 包含Top-K、Top-P等
[![【免訓練的最佳化技術】多項式取樣（Multinomial Sampling）](images/%E3%80%90%E5%85%8D%E8%AE%AD%E7%BB%83%E7%9A%84%E4%BC%98%E5%8C%96%E6%8A%80%E6%9C%AF%E3%80%91%E5%A4%9A%E9%A1%B9%E5%BC%8F%E9%87%87%E6%A0%B7%EF%BC%88Multinomial%20Sampling%EF%BC%89.png)](images/%E3%80%90%E5%85%8D%E8%AE%AD%E7%BB%83%E7%9A%84%E4%BC%98%E5%8C%96%E6%8A%80%E6%9C%AF%E3%80%91%E5%A4%9A%E9%A1%B9%E5%BC%8F%E9%87%87%E6%A0%B7%EF%BC%88Multinomial%20Sampling%EF%BC%89.png)

## 【免訓練的最佳化技術】Top-K取樣（Top-K Sampling）
- **Top-K取樣**（Top-K Sampling）是一種在生成任務中常用的策略，類似於**多項式取樣**，但其取樣候選池經過限制。使用Top-K取樣時，每一步生成Token時僅保留模型預測機率最高的**前K個**詞，並從中按機率分佈進行隨機抽樣。
[![【免訓練的最佳化技術】Top-K取樣（Top-K Sampling）](images/%E3%80%90%E5%85%8D%E8%AE%AD%E7%BB%83%E7%9A%84%E4%BC%98%E5%8C%96%E6%8A%80%E6%9C%AF%E3%80%91Top-K%E9%87%87%E6%A0%B7%EF%BC%88Top-K%20Sampling%EF%BC%89.png)](images/%E3%80%90%E5%85%8D%E8%AE%AD%E7%BB%83%E7%9A%84%E4%BC%98%E5%8C%96%E6%8A%80%E6%9C%AF%E3%80%91Top-K%E9%87%87%E6%A0%B7%EF%BC%88Top-K%20Sampling%EF%BC%89.png)

## 【免訓練的最佳化技術】Top-P取樣（Top-P Sampling）
- **Top-P取樣**（Top-P Sampling），又稱**核取樣**（Nucleus Sampling），該方法透過動態選擇一個最小候選集合，使得候選詞的機率和達到設定的機率閾值P，然後，在該候選集合中隨機取樣。與Top-K取樣相比，Top-P取樣能夠根據機率累積**動態調整**候選集的大小。
[![【免訓練的最佳化技術】Top-P取樣（Top-P Sampling）](images/%E3%80%90%E5%85%8D%E8%AE%AD%E7%BB%83%E7%9A%84%E4%BC%98%E5%8C%96%E6%8A%80%E6%9C%AF%E3%80%91Top-P%E9%87%87%E6%A0%B7%EF%BC%88Top-P%20Sampling%EF%BC%89.png)](images/%E3%80%90%E5%85%8D%E8%AE%AD%E7%BB%83%E7%9A%84%E4%BC%98%E5%8C%96%E6%8A%80%E6%9C%AF%E3%80%91Top-P%E9%87%87%E6%A0%B7%EF%BC%88Top-P%20Sampling%EF%BC%89.png)

## 【免訓練的最佳化技術】RAG（檢索增強生成,Retrieval-Augmented Generation）
- **RAG**（Retrieval-Augmented Generation，檢索增強生成）是一種結合資訊檢索與模型生成的技術，透過引入外部知識庫或檢索系統，增強生成式模型的知識範圍和回答準確性。
- Meta（前身為Facebook AI Research）等研究團隊於2020年在其發表的工作中提出了RAG，並顯著提升了知識密集型NLP任務的效能。
- RAG的原理如圖所示，整體可分為兩部分——離線構建環節和線上服務環節。
[![【免訓練的最佳化技術】RAG（檢索增強生成）](images/%E3%80%90%E5%85%8D%E8%AE%AD%E7%BB%83%E7%9A%84%E4%BC%98%E5%8C%96%E6%8A%80%E6%9C%AF%E3%80%91RAG%EF%BC%88%E6%A3%80%E7%B4%A2%E5%A2%9E%E5%BC%BA%E7%94%9F%E6%88%90%EF%BC%89.png)](images/%E3%80%90%E5%85%8D%E8%AE%AD%E7%BB%83%E7%9A%84%E4%BC%98%E5%8C%96%E6%8A%80%E6%9C%AF%E3%80%91RAG%EF%BC%88%E6%A3%80%E7%B4%A2%E5%A2%9E%E5%BC%BA%E7%94%9F%E6%88%90%EF%BC%89.png)

## 【免訓練的最佳化技術】功能呼叫（Function Calling）
- **功能呼叫**（Function Calling），也稱工具呼叫（Tool Use），是指在基於大型模型完成任務的過程中，Agent透過特定機制呼叫外部物件，獲取返回結果後將其與原始Prompt一起輸入到大型模型，由大型模型進一步推理並完成特定任務。
- 被呼叫的物件可以是遠端API、資料庫查詢介面、本地函式或工具外掛（Plugin）等。
- 下圖展示了功能與工具呼叫技術的執行流程。其中，Agent是一個本地執行的軟體系統，大型模型是其子模組之一。Agent還包括使用者請求解析模組、引數處理模組、工具呼叫模組、呼叫結果解析模組及與大型模型互動的元件等。

[![【免訓練的最佳化技術】功能呼叫（Function Calling）](images/%E3%80%90%E5%85%8D%E8%AE%AD%E7%BB%83%E7%9A%84%E4%BC%98%E5%8C%96%E6%8A%80%E6%9C%AF%E3%80%91%E5%8A%9F%E8%83%BD%E8%B0%83%E7%94%A8%EF%BC%88Function%20Calling%EF%BC%89.png)](images/%E3%80%90%E5%85%8D%E8%AE%AD%E7%BB%83%E7%9A%84%E4%BC%98%E5%8C%96%E6%8A%80%E6%9C%AF%E3%80%91%E5%8A%9F%E8%83%BD%E8%B0%83%E7%94%A8%EF%BC%88Function%20Calling%EF%BC%89.png)
