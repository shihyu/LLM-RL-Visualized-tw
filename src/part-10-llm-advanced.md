# 第10部分：大型模型基礎拓展

## 【LLM基礎拓展】大型模型效能最佳化技術圖譜
- 如圖所示，在訓練和推理階段，大型模型的最佳化技術可大致分為**五個層次**：服務層、模型層、框架層、系統編譯層和硬體通訊層。
- 圖中各層內的細分技術為最佳化提供了明確的方向和實踐路徑。
[![【LLM基礎拓展】大型模型效能最佳化技術圖譜](images/%E3%80%90LLM%E5%9F%BA%E7%A1%80%E6%8B%93%E5%B1%95%E3%80%91%E5%A4%A7%E6%A8%A1%E5%9E%8B%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96%E6%8A%80%E6%9C%AF%E5%9B%BE%E8%B0%B1.png)](images/%E3%80%90LLM%E5%9F%BA%E7%A1%80%E6%8B%93%E5%B1%95%E3%80%91%E5%A4%A7%E6%A8%A1%E5%9E%8B%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96%E6%8A%80%E6%9C%AF%E5%9B%BE%E8%B0%B1.png)

## 【LLM基礎拓展】ALiBi位置編碼
- RoPE已成主流，ALiBi逐漸被拋棄
[![【LLM基礎拓展】ALiBi位置編碼](images/%E3%80%90LLM%E5%9F%BA%E7%A1%80%E6%8B%93%E5%B1%95%E3%80%91ALiBi%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81.png)](images/%E3%80%90LLM%E5%9F%BA%E7%A1%80%E6%8B%93%E5%B1%95%E3%80%91ALiBi%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81.png)

## 【LLM基礎拓展】傳統的知識蒸餾
- **知識蒸餾**（Knowledge Distillation）：透過將訓練好的**教師模型**輸出的**軟標籤**（soft labels）轉移給更小的**學生模型**進行訓練，使學生模型學習教師的輸出分佈資訊，從而在**模型壓縮**和**推理效率**上獲得顯著提升。 
- 最早由**深度學習之父**Geoffrey Hinton等人在論文《Distilling the Knowledge in a Neural Network》中提出。

[![【LLM基礎拓展】傳統的知識蒸餾](images/%E3%80%90LLM%E5%9F%BA%E7%A1%80%E6%8B%93%E5%B1%95%E3%80%91%E4%BC%A0%E7%BB%9F%E7%9A%84%E7%9F%A5%E8%AF%86%E8%92%B8%E9%A6%8F.png)](images/%E3%80%90LLM%E5%9F%BA%E7%A1%80%E6%8B%93%E5%B1%95%E3%80%91%E4%BC%A0%E7%BB%9F%E7%9A%84%E7%9F%A5%E8%AF%86%E8%92%B8%E9%A6%8F.png)

## 【LLM基礎拓展】數值表示、量化（Quantization）
- AI中常用的數值型別如圖，它們的數值實現如下表。

| 型別   | 總位數 | 符號位 | 指數位 | 尾數位 / 整數位   |
| ------ | ------ | ------ | ------ | ----------------- |
| FP64   | 64     | 1      | 11     | 52 (尾數)         |
| FP32   | 32     | 1      | 8      | 23 (尾數)         |
| TF32   | 19     | 1      | 8      | 10 (尾數)         |
| BF16   | 16     | 1      | 8      | 7 (尾數)          |
| FP16   | 16     | 1      | 5      | 10 (尾數)         |
| INT64  | 64     | 1      | –      | 63 (整數值)       |
| INT32  | 32     | 1      | –      | 31 (整數值)       |
| INT8   | 8      | 1      | –      | 7 (整數值)        |
| UINT8  | 8      | 0      | –      | 8 (整數值)        |
| INT4   | 4      | 1      | –      | 3 (整數值)        |

[![【LLM基礎拓展】數值表示、量化](images/%E3%80%90LLM%E5%9F%BA%E7%A1%80%E6%8B%93%E5%B1%95%E3%80%91%E6%95%B0%E5%80%BC%E8%A1%A8%E7%A4%BA%E3%80%81%E9%87%8F%E5%8C%96.png)](images/%E3%80%90LLM%E5%9F%BA%E7%A1%80%E6%8B%93%E5%B1%95%E3%80%91%E6%95%B0%E5%80%BC%E8%A1%A8%E7%A4%BA%E3%80%81%E9%87%8F%E5%8C%96.png)

## 【LLM基礎拓展】常規訓練時前向和反向過程
這張圖展示了神經網路在常規訓練中的前向傳播（Forward）和反向傳播（Backward）流程：
- 上半部分是**前向傳播**：輸入依次透過各層（Layer 1~4），每層的**啟用值**（Activation）會被快取，用於後續計算梯度。
- 下半部分是**反向傳播**：從最終的損失（Loss）開始，逐層計算梯度（Gradient 4 → 1），並利用前向快取的啟用值進行**鏈式求導**。
[![【LLM基礎拓展】常規訓練時前向和反向過程](images/%E3%80%90LLM%E5%9F%BA%E7%A1%80%E6%8B%93%E5%B1%95%E3%80%91%E5%B8%B8%E8%A7%84%E8%AE%AD%E7%BB%83%E6%97%B6%E5%89%8D%E5%90%91%E5%92%8C%E5%8F%8D%E5%90%91%E8%BF%87%E7%A8%8B.png)](images/%E3%80%90LLM%E5%9F%BA%E7%A1%80%E6%8B%93%E5%B1%95%E3%80%91%E5%B8%B8%E8%A7%84%E8%AE%AD%E7%BB%83%E6%97%B6%E5%89%8D%E5%90%91%E5%92%8C%E5%8F%8D%E5%90%91%E8%BF%87%E7%A8%8B.png)

## 【LLM基礎拓展】梯度累積（Gradient Accumulation）訓練
這張圖對比了常規訓練與梯度累積（Gradient Accumulation）訓練的差異：
- **常規訓練**（上方）：每個 batch（如 Batch 1）執行一次前向 & 反向傳播，立刻計算梯度並更新模型引數，更新**頻繁**，對視訊記憶體要求較低，但 batch size 不能太大。
- **梯度累積**訓練（下方）：多個 batch（如 Batch 1~3）分別計算梯度，不立刻更新，而是將**多個梯度累加後統一更新一次**，等效於更大的 batch size，適合視訊記憶體受限的情況。
[![【LLM基礎拓展】梯度累積（Gradient Accumulation）訓練](images/%E3%80%90LLM%E5%9F%BA%E7%A1%80%E6%8B%93%E5%B1%95%E3%80%91%E6%A2%AF%E5%BA%A6%E7%B4%AF%E7%A7%AF%EF%BC%88Gradient%20Accumulation%EF%BC%89%E8%AE%AD%E7%BB%83.png)](images/%E3%80%90LLM%E5%9F%BA%E7%A1%80%E6%8B%93%E5%B1%95%E3%80%91%E6%A2%AF%E5%BA%A6%E7%B4%AF%E7%A7%AF%EF%BC%88Gradient%20Accumulation%EF%BC%89%E8%AE%AD%E7%BB%83.png)

## 【LLM基礎拓展】Gradient Checkpoint（梯度重計算）
這張圖對比了常規訓練與梯度重計算（Gradient Checkpoint）訓練的不同：
- **常規訓練**（上方）：每層的啟用值（Activation）都**完整儲存**，便於反向傳播時直接使用，但視訊記憶體佔用大，不適合大型模型。
- **梯度重計算**（下方）：只儲存**關鍵層**的啟用值（如 Activation 1 和 3），在反向傳播時，對中間未儲存的啟用值（如 2 和 4）進行重新前向計算，以此節省視訊記憶體。
- 總結：梯度重計算（Gradient Checkpointing）透過以**多算換空間**的方式降低視訊記憶體使用，適合訓練大型模型時節省資源，但代價是訓練時間略有增加。
[![【LLM基礎拓展】Gradient Checkpoint（梯度重計算）](images/%E3%80%90LLM%E5%9F%BA%E7%A1%80%E6%8B%93%E5%B1%95%E3%80%91Gradient%20Checkpoint%EF%BC%88%E6%A2%AF%E5%BA%A6%E9%87%8D%E8%AE%A1%E7%AE%97%EF%BC%89.png)](images/%E3%80%90LLM%E5%9F%BA%E7%A1%80%E6%8B%93%E5%B1%95%E3%80%91Gradient%20Checkpoint%EF%BC%88%E6%A2%AF%E5%BA%A6%E9%87%8D%E8%AE%A1%E7%AE%97%EF%BC%89.png)

## 【LLM基礎拓展】Full recomputation(完全重計算)
- **完全重計算**（Full Recomputation），即在反向傳播階段**不保留任何**啟用值，而是每次需要時從輸入開始重新執行前向計算獲取中間啟用。
- 這種方式極大地節省了視訊記憶體，非常適合在記憶體受限場景下訓練大型模型，但會**顯著增加**計算量和訓練時間。
[![【LLM基礎拓展】Full recomputation(完全重計算)](images/%E3%80%90LLM%E5%9F%BA%E7%A1%80%E6%8B%93%E5%B1%95%E3%80%91Full%20recomputation%28%E5%AE%8C%E5%85%A8%E9%87%8D%E8%AE%A1%E7%AE%97%29.png)](images/%E3%80%90LLM%E5%9F%BA%E7%A1%80%E6%8B%93%E5%B1%95%E3%80%91Full%20recomputation%28%E5%AE%8C%E5%85%A8%E9%87%8D%E8%AE%A1%E7%AE%97%29.png)

## 【LLM基礎拓展】LLM Benchmark
- LLM的評分（Benchmark）有多種，例如MMLU、C-eval等等，其實現原理都較為類似，如下圖所示。
[![【LLM基礎拓展】LLM Benchmark](images/%E3%80%90LLM%E5%9F%BA%E7%A1%80%E6%8B%93%E5%B1%95%E3%80%91LLM%20Benchmark.png)](images/%E3%80%90LLM%E5%9F%BA%E7%A1%80%E6%8B%93%E5%B1%95%E3%80%91LLM%20Benchmark.png)

## 【LLM基礎拓展】MHA、GQA、MQA、MLA
[![【LLM基礎拓展】MHA、GQA、MQA、MLA](images/%E3%80%90LLM%E5%9F%BA%E7%A1%80%E6%8B%93%E5%B1%95%E3%80%91MHA%E3%80%81GQA%E3%80%81MQA%E3%80%81MLA.png)](images/%E3%80%90LLM%E5%9F%BA%E7%A1%80%E6%8B%93%E5%B1%95%E3%80%91MHA%E3%80%81GQA%E3%80%81MQA%E3%80%81MLA.png)

## 【LLM基礎拓展】RNN（Recurrent Neural Network）
- **RNN**（Recurrent Neural Network）：一種專門處理**序列資料**的神經網路，透過**迴圈連線**在每個時間步保留並更新前一時刻的**隱藏狀態**，使網路具有記憶能力。 
- 優點：結構**簡單**、能夠捕捉**短期依賴**；缺點：易出現**梯度消失/爆炸**，難以學習**長距離依賴**。
[![【LLM基礎拓展】RNN](images/%E3%80%90LLM%E5%9F%BA%E7%A1%80%E6%8B%93%E5%B1%95%E3%80%91RNN.png)](images/%E3%80%90LLM%E5%9F%BA%E7%A1%80%E6%8B%93%E5%B1%95%E3%80%91RNN.png)

## 【LLM基礎拓展】Pre-norm和Post-norm
- **Pre-norm**：在**子層**（如自注意力或前饋網路）**輸入前**先執行**LayerNorm**，然後經過子層並與原始輸入進行**殘差連線**，可提升**梯度流動**和深層模型的**訓練穩定性**  
- **Post-norm**：在子層輸出後與原始輸入進行**殘差連線**，再執行**LayerNorm**，是**Transformer**的經典結構，但在深層網路中可能導致**梯度衰減**和訓練不穩定  

[![【LLM基礎拓展】Pre-norm和Post-norm](images/%E3%80%90LLM%E5%9F%BA%E7%A1%80%E6%8B%93%E5%B1%95%E3%80%91Pre-norm%E5%92%8CPost-norm.png)](images/%E3%80%90LLM%E5%9F%BA%E7%A1%80%E6%8B%93%E5%B1%95%E3%80%91Pre-norm%E5%92%8CPost-norm.png)

## 【LLM基礎拓展】BatchNorm和LayerNorm
- **BatchNorm**（批次歸一化）：對每個**小批次**中的每個**通道**進行**標準化**（減去均值、除以標準差），再加上可學習的**縮放**和**偏移**引數。
- **LayerNorm**（層歸一化）：對每個**樣本**的所有**特徵維度**進行**標準化**（減去均值、除以標準差），同樣使用可學習的**縮放**和**偏移**引數。

[![【LLM基礎拓展】BatchNorm和LayerNorm](images/%E3%80%90LLM%E5%9F%BA%E7%A1%80%E6%8B%93%E5%B1%95%E3%80%91BatchNorm%E5%92%8CLayerNorm.png)](images/%E3%80%90LLM%E5%9F%BA%E7%A1%80%E6%8B%93%E5%B1%95%E3%80%91BatchNorm%E5%92%8CLayerNorm.png)

## 【LLM基礎拓展】RMSNorm
- RMSNorm（Root Mean Square Layer Normalization）是一種歸一化方法，僅基於輸入特徵的均方根值（RMS）進行規範化，**不計算均值偏移**。  
- 對每個樣本的特徵向量 \(x\)，先計算 \(\mathrm{RMS}(x)=\sqrt{\tfrac{1}{d}\sum_i x_i^2}\)，再做縮放：\(\hat{x}=x / \mathrm{RMS}(x)\times \gamma + \beta\)。  
- 相較於 LayerNorm，它省略了減均值步驟，計算更高效且在多種 Transformer 架構中表現相近或更優。
[![【LLM基礎拓展】RMSNorm](images/%E3%80%90LLM%E5%9F%BA%E7%A1%80%E6%8B%93%E5%B1%95%E3%80%91RMSNorm.png)](images/%E3%80%90LLM%E5%9F%BA%E7%A1%80%E6%8B%93%E5%B1%95%E3%80%91RMSNorm.png)

## 【LLM基礎拓展】Prune（剪枝）
- 模型剪枝（Model Pruning）是一種壓縮深度神經網路的方法，透過移除冗餘或不重要的權重、神經元或通道，減少模型引數量和計算量。
- 可以在儘量保持原始精度的前提下，提升模型的推理速度、降低記憶體佔用。
- 典型流程包括：評估重要性 → 標記待剪引數 → 重訓練（Fine-tuning）→ 驗證效能。  
[![【LLM基礎拓展】Prune（剪枝）](images/%E3%80%90LLM%E5%9F%BA%E7%A1%80%E6%8B%93%E5%B1%95%E3%80%91Prune%EF%BC%88%E5%89%AA%E6%9E%9D%EF%BC%89.png)](images/%E3%80%90LLM%E5%9F%BA%E7%A1%80%E6%8B%93%E5%B1%95%E3%80%91Prune%EF%BC%88%E5%89%AA%E6%9E%9D%EF%BC%89.png)

## 【LLM基礎拓展】溫度係數的作用
- 在生成階段，溫度係數 \(T\) 透過放縮模型輸出的 logits 分佈來控制取樣**多樣性**。  
- 當 \(T < 1\) 時，分佈“尖銳”，模型更傾向於選擇機率**最高**的 token，回答更固定。
- 當 \(T > 1\) 時，分佈“平坦”，取樣結果更具**隨機性**。  
- 適當調整溫度可以在準確性和創造性之間取得平衡——低溫度提升確定性，高溫度增強多樣性。
[![【LLM基礎拓展】溫度係數的作用](images/%E3%80%90LLM%E5%9F%BA%E7%A1%80%E6%8B%93%E5%B1%95%E3%80%91%E6%B8%A9%E5%BA%A6%E7%B3%BB%E6%95%B0%E7%9A%84%E4%BD%9C%E7%94%A8.png)](images/%E3%80%90LLM%E5%9F%BA%E7%A1%80%E6%8B%93%E5%B1%95%E3%80%91%E6%B8%A9%E5%BA%A6%E7%B3%BB%E6%95%B0%E7%9A%84%E4%BD%9C%E7%94%A8.png)

## 【LLM基礎拓展】SwiGLU
- 基本結構：SwiGLU 是 Gated Linear Unit（GLU）的變體，將輸入向量透過**兩條**線性對映，一路直接輸出，一路經過 SiLU（Swish）啟用後作為門控訊號，兩者逐元素相乘得到最終輸出。
- 啟用**優勢**：用 SiLU（Swish-1）替換原 GLU 中的 sigmoid，使門控更平滑、梯度更穩定，有助於加速訓練收斂並提升下游任務表現。
[![【LLM基礎拓展】SwiGLU](images/%E3%80%90LLM%E5%9F%BA%E7%A1%80%E6%8B%93%E5%B1%95%E3%80%91SwiGLU.png)](images/%E3%80%90LLM%E5%9F%BA%E7%A1%80%E6%8B%93%E5%B1%95%E3%80%91SwiGLU.png)

## 【LLM基礎拓展】AUC、PR、F1、Precision、Recall
被面試者、工程師經常混淆的演算法常用指標，我畫了一張圖來對比總結，區別一目瞭然。
- **AUC**（Area Under the ROC Curve）：衡量二分類模型在所有可能閾值下區分正負樣本的能力，ROC（Receiver Operating Characteristic）曲線以假正例率（FPR）為橫軸、真正例率（TPR）為縱軸，AUC 值越接近 1 越好；若 AUC≈0.5 則相當於隨機猜測。
- **PR Curve**（Precision–Recall Curve）：模型在不同召回率（Recall）下的精確率（Precision）變化趨勢。可透過曲線下面積來綜合評估模型對正類的檢出能力。
- **Precision（精確率）**：在所有被預測為正例的樣本中，實際為正例的比例，衡量誤報率高低，適用於誤報成本高的場景，個人認為“查準率”的翻譯更貼切。Precision = TP / (TP + FP)
- **Recall （召回率）**：在所有真實為正例的樣本中，被正確預測為正例的比例，衡量漏報率高低，適用於漏報成本高的場景，個人認為“查全率”的翻譯更貼切。Recall = TP / (TP + FN)
- **F1 Score** （F1 值）：Precision 與 Recall 的調和平均，當兩者平衡時取得最高分。F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
- **Accuracy （準確率）**：在所有樣本中，預測正確的比例，適用於類別分佈較為均衡時的整體效能評估。Accuracy = (TP + TN) / (TP + TN + FP + FN)
- **BLEU**（Bilingual Evaluation Understudy；機器翻譯評估指標）：基於 n-gram 重疊率評估翻譯結果與參考的相似程度。
- **ROUGE**（Recall-Oriented Understudy for Gisting Evaluation；摘要評估指標）：生成摘要與參考在 n-gram、最長公共子序列或詞重疊方面的匹配情況，關注召回效能。

[![【LLM基礎拓展】AUC、PR、F1、Precision、Recall](images/%E3%80%90LLM%E5%9F%BA%E7%A1%80%E6%8B%93%E5%B1%95%E3%80%91AUC%E3%80%81PR%E3%80%81F1%E3%80%81Precision%E3%80%81Recall.png)](images/%E3%80%90LLM%E5%9F%BA%E7%A1%80%E6%8B%93%E5%B1%95%E3%80%91AUC%E3%80%81PR%E3%80%81F1%E3%80%81Precision%E3%80%81Recall.png)

## 【LLM基礎拓展】RoPE位置編碼
- RoPE，Rotary Position Embedding，**旋轉**位置編碼，由蘇劍林提出
- 原理：對 Transformer 中的 Query 和 Key 隱藏向量應用旋轉變換（基於正弦、餘弦函式），將位置資訊編碼為向量的**相位差**。
- 相對位置：旋轉操作天然地保留了向量之間的相對角度差異，使模型能夠捕捉**相對位置**資訊而不依賴絕對位置索引。
- 優點：無需額外引數、計算**開銷低**；在長序列場景下相較於傳統絕對位置編碼具有更好的泛化和效能。
[![【LLM基礎拓展】RoPE位置編碼](images/%E3%80%90LLM%E5%9F%BA%E7%A1%80%E6%8B%93%E5%B1%95%E3%80%91RoPE%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81.png)](images/%E3%80%90LLM%E5%9F%BA%E7%A1%80%E6%8B%93%E5%B1%95%E3%80%91RoPE%E4%BD%8D%E7%BD%AE%E7%BC%96%E7%A0%81.png)

## 【LLM基礎拓展】RoPE對各序列位置、各維度的作用
- RoPE原理、Base與θ值、作用機制詳見：[RoPE-theta-base.xlsx](files/RoPE-theta-base.xlsx) 
- 維度成對旋轉：將嵌入向量按奇偶索引**兩兩分組**，對每一對子向量應用一個二維旋轉矩陣，旋轉角度隨位置和維度頻率共同決定。
**高頻**維度：對應下圖左半區、角速度大的那些維度。旋轉角度隨位置變化快，對相鄰 token 的位置微小差異非常敏感，擅長捕捉**短距離**的區域性相對位置資訊。
**低頻**維度：對應下圖右半區、角速度小的那些維度。旋轉角度隨位置變化慢，能夠區分較**遠距離**的相對位置資訊，擅長建模全域性或長距離依賴。
[![【LLM基礎拓展】RoPE對各序列位置、各維度的作用](images/%E3%80%90LLM%E5%9F%BA%E7%A1%80%E6%8B%93%E5%B1%95%E3%80%91RoPE%E5%AF%B9%E5%90%84%E5%BA%8F%E5%88%97%E4%BD%8D%E7%BD%AE%E3%80%81%E5%90%84%E7%BB%B4%E5%BA%A6%E7%9A%84%E4%BD%9C%E7%94%A8.png)](images/%E3%80%90LLM%E5%9F%BA%E7%A1%80%E6%8B%93%E5%B1%95%E3%80%91RoPE%E5%AF%B9%E5%90%84%E5%BA%8F%E5%88%97%E4%BD%8D%E7%BD%AE%E3%80%81%E5%90%84%E7%BB%B4%E5%BA%A6%E7%9A%84%E4%BD%9C%E7%94%A8.png)

---
