# 第9部分：邏輯推理（Reasoning）能力最佳化

## 【邏輯推理能力最佳化】基於CoT的知識蒸餾（Knowledge Distillation）
- **知識蒸餾**（Knowledge Distillation, KD）是一種模型**壓縮**技術，由**深度學習之父**Geoffrey Hinton等人於2015年提出。
- 強大型模型（例如OpenAI的o系列模型、DeepSeek、Qwen等）生成CoT推理鏈和回答結果，並將其作為訓練資料用於對**較弱**模型進行**蒸餾訓練**。
- 需要注意的是，此處的方法與傳統知識蒸餾略**有區別**。
[![【邏輯推理能力最佳化】基於CoT的知識蒸餾](images/%E3%80%90%E9%80%BB%E8%BE%91%E6%8E%A8%E7%90%86%E8%83%BD%E5%8A%9B%E4%BC%98%E5%8C%96%E3%80%91%E5%9F%BA%E4%BA%8ECoT%E7%9A%84%E7%9F%A5%E8%AF%86%E8%92%B8%E9%A6%8F.png)](images/%E3%80%90%E9%80%BB%E8%BE%91%E6%8E%A8%E7%90%86%E8%83%BD%E5%8A%9B%E4%BC%98%E5%8C%96%E3%80%91%E5%9F%BA%E4%BA%8ECoT%E7%9A%84%E7%9F%A5%E8%AF%86%E8%92%B8%E9%A6%8F.png)

## 【邏輯推理能力最佳化】基於DeepSeek進行蒸餾
- 為了降低模型體積和部署開銷，可以採用蒸餾技術，將效能較強的模型（例如DeepSeek）的部分能力**遷移**到體積更小的模型中。
[![【邏輯推理能力最佳化】基於DeepSeek進行蒸餾](images/%E3%80%90%E9%80%BB%E8%BE%91%E6%8E%A8%E7%90%86%E8%83%BD%E5%8A%9B%E4%BC%98%E5%8C%96%E3%80%91%E5%9F%BA%E4%BA%8EDeepSeek%E8%BF%9B%E8%A1%8C%E8%92%B8%E9%A6%8F.png)](images/%E3%80%90%E9%80%BB%E8%BE%91%E6%8E%A8%E7%90%86%E8%83%BD%E5%8A%9B%E4%BC%98%E5%8C%96%E3%80%91%E5%9F%BA%E4%BA%8EDeepSeek%E8%BF%9B%E8%A1%8C%E8%92%B8%E9%A6%8F.png)

## 【邏輯推理能力最佳化】ORM和PRM（結果獎勵模型和過程獎勵模型）
- **結果獎勵模型**（Outcome Reward Model, **ORM**）：對於給定的Prompt及模型生成的回答y，結果獎勵模型僅對**最終**輸出的結果進行整體驗證和獎勵評分。
- **過程獎勵模型**（Process Reward Model, **PRM**）：過程獎勵模型對推理過程中的每個中間**步驟**逐一進行驗證和獎勵評分，從而提供更**精細**的反饋，能夠有效指導模型關注推理過程的質量。
[![【邏輯推理能力最佳化】ORM和PRM](images/%E3%80%90%E9%80%BB%E8%BE%91%E6%8E%A8%E7%90%86%E8%83%BD%E5%8A%9B%E4%BC%98%E5%8C%96%E3%80%91ORM%E5%92%8CPRM.png)](images/%E3%80%90%E9%80%BB%E8%BE%91%E6%8E%A8%E7%90%86%E8%83%BD%E5%8A%9B%E4%BC%98%E5%8C%96%E3%80%91ORM%E5%92%8CPRM.png)

## 【邏輯推理能力最佳化】MCTS每次迭代的四個關鍵步驟
- **MCTS**每次迭代的**四個關鍵步驟**： MCTS演算法透過構建決策樹逐步探索問題的解空間，其核心思想是從當前狀態出發，透過**模擬**或推演評估狀態的價值，並利用評估結果指導後續的搜尋方向。
[![【邏輯推理能力最佳化】MCTS每次迭代的四個關鍵步驟](images/%E3%80%90%E9%80%BB%E8%BE%91%E6%8E%A8%E7%90%86%E8%83%BD%E5%8A%9B%E4%BC%98%E5%8C%96%E3%80%91MCTS%E6%AF%8F%E6%AC%A1%E8%BF%AD%E4%BB%A3%E7%9A%84%E5%9B%9B%E4%B8%AA%E5%85%B3%E9%94%AE%E6%AD%A5%E9%AA%A4.png)](images/%E3%80%90%E9%80%BB%E8%BE%91%E6%8E%A8%E7%90%86%E8%83%BD%E5%8A%9B%E4%BC%98%E5%8C%96%E3%80%91MCTS%E6%AF%8F%E6%AC%A1%E8%BF%AD%E4%BB%A3%E7%9A%84%E5%9B%9B%E4%B8%AA%E5%85%B3%E9%94%AE%E6%AD%A5%E9%AA%A4.png)

## 【邏輯推理能力最佳化】MCTS的執行過程
- MCTS透過**重複迭代**上述四個步驟，不斷擴充套件和完善搜尋樹，進而逐步提升對各狀態的評估精度。隨著推演迭代次數的增加，演算法**逐漸**傾向於選擇更優的路徑，從而不斷最佳化決策質量。
[![【邏輯推理能力最佳化】MCTS的執行過程](images/%E3%80%90%E9%80%BB%E8%BE%91%E6%8E%A8%E7%90%86%E8%83%BD%E5%8A%9B%E4%BC%98%E5%8C%96%E3%80%91MCTS%E7%9A%84%E8%BF%90%E8%A1%8C%E8%BF%87%E7%A8%8B.png)](images/%E3%80%90%E9%80%BB%E8%BE%91%E6%8E%A8%E7%90%86%E8%83%BD%E5%8A%9B%E4%BC%98%E5%8C%96%E3%80%91MCTS%E7%9A%84%E8%BF%90%E8%A1%8C%E8%BF%87%E7%A8%8B.png)

## 【邏輯推理能力最佳化】語言場景下的搜尋樹示例
- 搜尋樹中的每個節點代表句子或段落級別的內容（一個推理步驟）。
[![【邏輯推理能力最佳化】語言場景下的搜尋樹示例](images/%E3%80%90%E9%80%BB%E8%BE%91%E6%8E%A8%E7%90%86%E8%83%BD%E5%8A%9B%E4%BC%98%E5%8C%96%E3%80%91%E8%AF%AD%E8%A8%80%E5%9C%BA%E6%99%AF%E4%B8%8B%E7%9A%84%E6%90%9C%E7%B4%A2%E6%A0%91%E7%A4%BA%E4%BE%8B.png)](images/%E3%80%90%E9%80%BB%E8%BE%91%E6%8E%A8%E7%90%86%E8%83%BD%E5%8A%9B%E4%BC%98%E5%8C%96%E3%80%91%E8%AF%AD%E8%A8%80%E5%9C%BA%E6%99%AF%E4%B8%8B%E7%9A%84%E6%90%9C%E7%B4%A2%E6%A0%91%E7%A4%BA%E4%BE%8B.png)

## 【邏輯推理能力最佳化】BoN（Best-of-N）取樣
- 由模型（通常為參考模型）生成多個候選輸出（例如N個），然後依據某種評價標準，從中**挑選**出質量**最優**的一個作為最終輸出。
[![【邏輯推理能力最佳化】BoN（Best-of-N）取樣](images/%E3%80%90%E9%80%BB%E8%BE%91%E6%8E%A8%E7%90%86%E8%83%BD%E5%8A%9B%E4%BC%98%E5%8C%96%E3%80%91BoN%EF%BC%88Best-of-N%EF%BC%89%E9%87%87%E6%A0%B7.png)](images/%E3%80%90%E9%80%BB%E8%BE%91%E6%8E%A8%E7%90%86%E8%83%BD%E5%8A%9B%E4%BC%98%E5%8C%96%E3%80%91BoN%EF%BC%88Best-of-N%EF%BC%89%E9%87%87%E6%A0%B7.png)

## 【邏輯推理能力最佳化】多數投票（Majority Vote）方法
- **多數投票**（Majority Vote）的核心思想是：針對一個給定的Prompt，模型進行多次推理，生成多條不同的推理路徑，每條路徑可能產生一個最終答案，透過統計最終答案的出現頻率，將**頻率最高**的答案作為最終輸出。
[![【邏輯推理能力最佳化】多數投票（Majority Vote）方法](images/%E3%80%90%E9%80%BB%E8%BE%91%E6%8E%A8%E7%90%86%E8%83%BD%E5%8A%9B%E4%BC%98%E5%8C%96%E3%80%91%E5%A4%9A%E6%95%B0%E6%8A%95%E7%A5%A8%EF%BC%88Majority%20Vote%EF%BC%89%E6%96%B9%E6%B3%95.png)](images/%E3%80%90%E9%80%BB%E8%BE%91%E6%8E%A8%E7%90%86%E8%83%BD%E5%8A%9B%E4%BC%98%E5%8C%96%E3%80%91%E5%A4%9A%E6%95%B0%E6%8A%95%E7%A5%A8%EF%BC%88Majority%20Vote%EF%BC%89%E6%96%B9%E6%B3%95.png)

## 【邏輯推理能力最佳化】AlphaGoZero在訓練時的效能增長趨勢 <sup>[<a href="references.md">179</a>]</sup>
- **AlphaGo** Zero展現出最強的效能，Elo評分高達5185。在推理過程中，如**不使用MCTS**進行預搜尋，其Elo評分僅為3055。這一對比直觀地揭示了：**推理時搜尋**與計算能夠顯著提升模型的效能表現。
[![【邏輯推理能力最佳化】AlphaGoZero在訓練時的效能增長趨勢](images/%E3%80%90%E9%80%BB%E8%BE%91%E6%8E%A8%E7%90%86%E8%83%BD%E5%8A%9B%E4%BC%98%E5%8C%96%E3%80%91AlphaGoZero%E5%9C%A8%E8%AE%AD%E7%BB%83%E6%97%B6%E7%9A%84%E6%80%A7%E8%83%BD%E5%A2%9E%E9%95%BF%E8%B6%8B%E5%8A%BF.png)](images/%E3%80%90%E9%80%BB%E8%BE%91%E6%8E%A8%E7%90%86%E8%83%BD%E5%8A%9B%E4%BC%98%E5%8C%96%E3%80%91AlphaGoZero%E5%9C%A8%E8%AE%AD%E7%BB%83%E6%97%B6%E7%9A%84%E6%80%A7%E8%83%BD%E5%A2%9E%E9%95%BF%E8%B6%8B%E5%8A%BF.png)
