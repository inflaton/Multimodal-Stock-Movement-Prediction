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

Section VII now identifies the single daily FinBERT-Tone scalar as a specific limitation and lists document embeddings, event tags, intraday text, and step-wise evidence acquisition as alternatives. We do not make the stronger claim that sentiment is universally uninformative.

## 4. Foundation-model comparison

Section VII now combines both qualifications raised by the reviewer: the foundation-model experiments use one seed, and the released FinCast artifact accepts only price inputs. The manuscript therefore characterizes this comparison as directional rather than conclusive. Table VII also identifies the FinCast input constraint.

## 5. Learned-fusion scope

Section VII now states that the learned method tests a one-parameter convex weighting and that its failure does not rule out richer cross-modal fusion architectures.

## 6. Comparison with prior headline results

We softened the abstract and Section V-A. The manuscript now describes the Sharpe-ratio change from 1.40 to 2.29 as a controlled reconstruction showing how much inflation test-set selection can create in our fixed pipeline. It no longer implies direct causal attribution to any specific prior paper, and Section VII repeats this boundary.

## 7. Camera-ready preparation

The camera-ready manuscript includes the author names and affiliations, uses the IEEE conference format, and remains within the eight-page limit including references and the appendix. We also checked the final PDF for embedded fonts, letter-size pages, security settings, attachments, active links, and visual layout.

Thank you for including our paper in the SENTIRE program.

Kind regards,

Donghao Huang  
Zheng Kai Heng  
Zhaoxia Wang
