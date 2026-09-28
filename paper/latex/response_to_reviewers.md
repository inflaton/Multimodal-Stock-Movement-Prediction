# Email to the SENTIRE Program Chairs

**To:** sentire@sentic.net  
**Subject:** DM791 camera-ready paper and response to reviewer comments

Dear SENTIRE Program Chairs,

Please find attached the camera-ready version of our paper DM791, “Does Sentiment Fusion Help? A Selection-Bias-Free Re-Evaluation of Multimodal Stock Movement Prediction.”

We thank Reviewer 1 for the careful and constructive assessment. We revised the manuscript to address every point in the report while preserving the deliberately bounded scope of our conclusions. The code and processed benchmark are available at:

https://github.com/inflaton/Multimodal-Stock-Movement-Prediction

The revisions are summarized below.

## 1. Recent evidence-acquisition literature

We added Deng et al., “Multi-Agent SEA: Step-Wise Evidence Acquisition for Multimodal Financial Reasoning” (Information Fusion, 2026), and a corresponding paragraph in Section II. The revision distinguishes SEA’s evidence-grounded multimodal question-answering setting from our out-of-sample stock-movement prediction setting. We also connect its task-conditioned evidence acquisition to a concrete future direction for our study: replacing daily mean-pooled sentiment with event- or query-conditioned textual representations. Section VII cites this direction explicitly.

## 2. Scope of the empirical claim

Section VII now states more directly that the negative result is based on five large-cap U.S. equities and one test year and may not transfer to other market regimes, small-cap equities, or other markets. The abstract and conclusion retain the qualifier “in this setting.”

## 3. Sentiment representation

<!-- [CA-PRECISION] Extend the response to cover the qualified claims in Sections V and VI. -->
Section VII now identifies the single daily FinBERT-Tone scalar as a specific limitation and lists document embeddings, event tags, intraday text, and step-wise evidence acquisition as alternatives. Sections V-B, V-C, and VI also qualify the negative finding as applying to the evaluated daily mean-pooled FinBERT-Tone representation. We do not make the stronger claim that sentiment is universally uninformative.

## 4. Foundation-model comparison

Section VII now combines both qualifications raised by the reviewer: the foundation-model experiments use one seed, and the released FinCast artifact accepts only price inputs. The manuscript therefore characterizes this comparison as directional rather than conclusive. Table VII also identifies the FinCast input constraint.

## 5. Learned-fusion scope

Section VII now states that the learned method tests a one-parameter convex weighting and that its failure does not rule out richer cross-modal fusion architectures.

## 6. Comparison with prior headline results

We softened the abstract and Section V-A. The manuscript now describes the Sharpe-ratio change from 1.40 to 2.29 as a controlled reconstruction showing how much inflation test-set selection can create in our fixed pipeline. It no longer implies direct causal attribution to any specific prior paper, and Section VII repeats this boundary.

<!-- [CA-PRECISION][CA-SPY][CA-VAL] Add a consolidated account of the coauthor's follow-up points. -->
## 7. Follow-up clarifications and robustness numbers

<!-- [CA-PRECISION][CA-VAL] Explain the narrow definition and remaining validation-overfitting risk. -->
We define “selection-bias-free” in the abstract and introduction as the absence of test-set information in model-family, horizon, and fusion-weight selection. Section VII now acknowledges that reusing the 2023 validation window for hyperparameter tuning and these selections can still lead to validation overfitting; nested temporal validation or additional rolling validation windows would help assess this sensitivity.

<!-- [CA-SPY] Report verified numbers, the rounding convention, and the reproducible evidence source. -->
Section VII now reports the actual SPY-excluded results. Across 420 saved runs per feature configuration, technical-only has mean Sharpe 1.05, compared with 0.39 for fixed 70:30 fusion (a 0.65 gap calculated before rounding) and 0.58 for the strongest sentiment configuration by mean Sharpe, per-stock learned fusion. Technical-only also has the highest mean AUC, 0.543 versus 0.540 for news-only, the strongest sentiment configuration by AUC. This is an evaluation-only exclusion: models and fusion weights, including the globally fitted weight, are unchanged. The accompanying `scripts/summarize_spy_exclusion.py` reproduces the summary from saved results and records input hashes in `results/spy_excluded_robustness.json`.

<!-- [CA-TONE][CA-PRECISION][CA-PERCENT] Explain wording, model-comparison, and percentage corrections. -->
We replace “honestly” and “honest” with descriptions of validation-only selection or validation-locked evaluation. Section V-D now states the observed model-family averages: Logistic Regression leads on AUC, Sharpe, and win rate; the LSTM has near-chance AUC; and the LSTM and SVM have slightly negative average Sharpe ratios. We describe Logistic Regression as a regularized linear classifier without asserting a cross-family ranking of regularization strength or model capacity. We also correct the abstract's percentage wording: the gap from 1.40 to 2.29 is 39% of the test-selected value, not a 39% increase over 1.40.

<!-- [CA-PRECISION] Renumber the existing preparation section after inserting the follow-up response. -->
## 8. Camera-ready preparation

The camera-ready manuscript includes the author names and affiliations, uses the IEEE conference format, and remains within the eight-page limit including references and the appendix. We also checked the final PDF for embedded fonts, letter-size pages, security settings, attachments, active links, and visual layout.

Thank you for including our paper in the SENTIRE program.

Kind regards,

Donghao Huang  
Zheng Kai Heng  
Zhaoxia Wang
