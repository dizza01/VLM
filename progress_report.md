# Explainable Deep Learning for Colon Polyp Segmentation: A Progression Report

## Dataset and Medical Domain

This research focuses on automated polyp detection and segmentation in colonoscopy images using the CVC-ClinicDB dataset. Colorectal polyps are precancerous lesions that require accurate identification during colonoscopy procedures for early cancer prevention. The dataset comprises 612 colonoscopy frames with corresponding ground truth segmentation masks, split into training (489 images) and validation (123 images) subsets. All images were standardised to 256×256 pixels for model training. The CVC-ClinicDB dataset presents realistic clinical challenges, including variations in polyp appearance, lighting conditions, and tissue texture, making it representative of real-world endoscopic imaging scenarios.

## Experiments and Preliminary Work

### End-to-End Pipeline Architecture

An end-to-end deep learning pipeline was developed incorporating three primary components: model training, explainability methods, and quantitative evaluation metrics. The pipeline enables automated polyp segmentation whilst providing interpretable explanations for model predictions, addressing the critical need for transparency in medical AI systems.

The segmentation models employed include a baseline U-Net architecture and a Segmentation Transformer (SegTransformer) variant. The U-Net achieved a Dice coefficient of 0.7879 after 50 training epochs, demonstrating robust performance on the validation set. The SegTransformer, incorporating vision transformer-based patch embeddings with multi-head self-attention mechanisms, achieved a Dice coefficient of 0.6808. For baseline comparison, Otsu thresholding was implemented, yielding substantially lower performance (Dice: 0.2821, IoU: 0.1642), confirming the necessity of deep learning approaches for this task.

### Explainability Integration

Three complementary explainability methods were integrated to provide pixel-level attribution maps: Integrated Gradients, Guided Backpropagation, and Gradient-weighted Class Activation Mapping (Grad-CAM). These methods enable clinicians to understand which image regions influenced the model's predictions. Integrated Gradients computes cumulative gradients along a baseline-to-input path, providing theoretically grounded attributions. Guided Backpropagation selectively backpropagates positive gradients to highlight discriminative features. Grad-CAM generates coarse localization maps by leveraging final convolutional layer activations.

### Quantitative Evaluation Framework

A novel contribution of this work is the evaluation of explanation quality using the Quantus library. The Sparseness metric was implemented to assess explanation focus, measuring whether attributions concentrate on relevant polyp regions or diffuse across the entire image. High sparseness values (>0.9) indicate focused, clinically interpretable explanations that highlight specific polyp boundaries. This quantitative assessment framework enables objective comparison of explainability methods, moving beyond subjective visual inspection.

### Uncertainty Quantification

Preliminary uncertainty quantification methods were implemented to provide confidence estimates for model predictions. Monte Carlo Dropout enables stochastic inference by maintaining dropout during test time, generating prediction distributions from multiple forward passes. Entropy-based uncertainty measures quantify prediction confidence at pixel-level resolution. These uncertainty estimates are critical for clinical deployment, enabling the system to flag ambiguous cases requiring expert review.

### Interactive Deployment Platform

A Streamlit-based web application was developed to demonstrate the complete pipeline in an accessible interface. The platform enables real-time inference, visualisation of multiple explainability methods, and presentation of quantitative metrics. This deployment validates the practical feasibility of the approach and facilitates stakeholder engagement during the development process.

## Next Steps

### Advanced Uncertainty Quantification

Implementation of more sophisticated uncertainty estimation techniques will be explored such as Monte Carlo Markov Chain (MCMC) and Bayesian uncertainty estimates. The aim is that these methods will improve the system's ability to identify edge cases where human expert review is essential.

### Fine-Tuning Transformer Architectures

The SegTransformer architecture requires further optimisation to match or exceed U-Net performance. In this first iteration of experiments, a baseline SegTransformer was applied without any tuning. As such, hyperparameter tuning, including learning rate schedules, patch sizes, and attention head configurations, will be conducted. Transfer learning from pre-trained vision transformers (e.g., ViT, Swin Transformer) may accelerate convergence and improve feature representations.

### Expanded Explainability Metrics

Additional Quantus metrics will be integrated to comprehensively evaluate explanation quality. Max Sensitivity assesses explanation stability under input perturbations, ensuring consistent attributions across imaging variations. Faithfulness metrics verify that highlighted regions genuinely influence model predictions by measuring performance degradation when salient features are occluded. These metrics will validate that explanations accurately reflect model reasoning rather than producing spurious correlations.

### Dataset Enhancement

Integration of datasets with expert-annotated explanations will enable supervised evaluation of attribution maps. Human-in-the-loop validation will compare model explanations against clinician attention patterns, ensuring clinical relevance of generated attributions. Expanding to larger datasets (e.g., Kvasir-SEG, ETIS-Larib) will assess model generalisation across diverse imaging conditions and polyp morphologies.

### Clinical Validation Study

A prospective validation study with gastroenterologists is planned to assess the clinical utility of the explainability framework. Metrics will include diagnostic accuracy improvement, decision confidence, and time-to-diagnosis when clinicians use AI-assisted explanations compared to unassisted interpretation. This evaluation will bridge the gap between technical performance metrics and real-world clinical impact.

---

## Vision-Language Model (VLM) Integration for Medical VQA

### Overview and Research Direction

Building on the explainability foundation established in segmentation, recent work has begun investigating vision-language models (VLMs) for medical Visual Question Answering (VQA) with multimodal explainability. This direction addresses the research question of whether VLM explanations reflect causal reasoning or constitute post-hoc rationalisations. The work frames explainability as a faithfulness and causality problem rather than solely a user interface consideration.

The research uses the MediaEval-Medico-2025 Kvasir-VQA dataset as a preliminary testbed, comprising gastrointestinal (GI) endoscopy images paired with clinical questions and answers. This medical domain provides conditions for evaluating VLM explanation reliability.

### Subtask 1: GI Image VQA with PaliGemma

An initial fine-tuning pipeline has been implemented for google/paligemma-3b-pt-224 using the ms-swift framework, configured for resource-constrained environments (Google Colab T4 GPU with 16GB memory). Key technical components include:

**Model Architecture and Training:**
- 4-bit quantisation (bitsandbytes nf4 with double quantisation) for memory efficiency
- LoRA (Low-Rank Adaptation) with rank 16, alpha 32 for parameter-efficient fine-tuning
- Frozen vision encoder with gradient checkpointing to reduce memory requirements
- Training configuration: batch size 4, gradient accumulation 4 (effective batch 16), learning rate 2e-5
- Validation during training with model selection based on token accuracy
- Automated deployment to Hugging Face Hub for reproducibility

**Data Pipeline:**
- Cached 2,000+ unique GI endoscopy images from Kvasir-VQA dataset
- Generated VLM-compatible JSONL format with image tokens and message structure
- Training set: full dataset; validation set: 1,000 randomly sampled examples
- Data structure adapted for ms-swift's multimodal training requirements

**Preliminary Results:**
- Trained model deployed at: dizza01/Kvasir-VQA-x1-lora_260116-2028
- Initial inference validation demonstrates question-answering capability on GI pathology queries
- Model deployment implemented via Swift's PtEngine with adapter loading
- Experiment tracking integrated with Weights & Biases

### Subtask 2: Multimodal Visual Explanations (Preliminary Work)

A PaliGemmaExplainer class has been developed to extract attention-based explanations from the fine-tuned VLM. This implementation provides infrastructure for subsequent causal analysis of model decision-making.

**Attention Extraction Methods:**
- Direct extraction of vision encoder attention weights from transformer layers
- CLS token attention aggregation across heads to identify salient image regions
- Gradient-based saliency as alternative when attention outputs are unavailable
- Dynamic patch grid size detection (handles 14×14 or other resolutions)
- Attention map normalisation and upsampling to original image dimensions

**Grad-CAM Implementation:**
- Backward gradients through vision tower to compute class activation maps
- ReLU activation to highlight positive contributions
- Heatmap overlay generation for visualisation

**Multi-Panel Explanation Framework:**
The system generates 4-panel visualisations for clinical review:
1. Original GI Image (baseline reference)
2. Vision Encoder Attention (attention weights showing model focus regions)
3. Grad-CAM Saliency (gradient-weighted discriminative feature localisation)
4. Clinical Summary (structured explanation including question, model-generated answer, confidence score, detected medical terms, and confidence level interpretation)

**Medical Domain Integration:**
- Medical terminology extraction from free-text answers using curated clinical vocabulary
- Confidence calibration: average token probability across generated answer
- Clinical relevance scoring based on detected pathology terms
- Visual attention validation against expected anatomical regions

### Evaluation Framework for Explanation Faithfulness

**Batch Evaluation Pipeline:**
- Automated generation of visual explanations for test set samples
- Confidence distribution analysis across predictions
- Medical term coverage statistics (percentage of answers containing clinical entities)
- Per-sample storage of predictions, ground truth, confidence scores, and medical terms
- JSON export of evaluation metrics for reproducibility

**Quantitative Metrics Implemented:**
- Mean Confidence: Average token probability across test set (calibration indicator)
- Confidence Distribution: Histogram and box plots showing High (>0.8), Moderate (0.5-0.8), Low (<0.5) categories
- Medical Term Coverage: Percentage of predictions containing domain-specific vocabulary
- Attention Concentration: Measured via attention map entropy and sparseness

**Visualisation Outputs:**
All explanation artifacts saved to output_Kvasir-VQA-x1/subtask2_explanations/:
- Individual 4-panel explanation PNGs per test sample
- evaluation_results.json with aggregate metrics
- confidence_analysis.png showing distribution statistics

### Planned Experiments for Publication

The current implementation establishes infrastructure for rigorous causal evaluation of VLM explanations. Planned experiments include:

**Axis 1: Visual Faithfulness (Causal Relevance)**
- Visual Deletion Test: Progressive removal of top-k salient regions identified by attention maps, measuring answer degradation (accuracy drop, confidence change, entropy increase)
- Hypothesis: Faithful explanations should cause faster degradation under targeted deletion versus random or background deletion

**Axis 2: Language Faithfulness (Rationale Sensitivity)**
- Explanation-Answer Dependency Test: Replace generated explanations with contradictory, generic, or shuffled alternatives; measure answer stability
- Tests whether explanations are post-hoc (answer unchanged) or integrated into reasoning

**Axis 3: Cross-Modal Alignment Consistency**
- Token-Region Grounding: Measure overlap between text tokens (e.g., "polyp", "ulcer") and corresponding visual attention regions
- Extends beyond IoU to link language semantics to vision at token level using grounded attention maps

**Axis 4: Counterfactual Multimodal Reasoning**
- Counterfactual VQA: Inpaint or remove pathology regions, swap textual attributes, measure explanation adaptation
- Metrics: Explanation edit distance, semantic shift, visual attention shift under intervention

**Human-Centred Calibration Study:**
- Error Detection Task: Evaluate whether clinicians can detect incorrect model answers with and without explanations
- Over-Reliance Measurement: Assess whether plausible but incorrect explanations increase false acceptance
- Conditions: Answer-only versus Answer+Visual versus Answer+Multimodal explanations
- Statistical Analysis: Bootstrap confidence intervals combined with mixed-effects models (clinician as random effect)

### Technical Infrastructure

**Reproducibility:**
- API token management for Hugging Face and Weights & Biases
- Complete training command with documented hyperparameters in notebook
- Version-pinned dependencies: ms-swift, bitsandbytes, transformers>=4.40, scipy, scikit-learn
- Checkpoint saving strategy: save_steps=1000, save_total_limit=2

**Computational Efficiency:**
- Memory optimisation enables training on free-tier Colab T4 (16GB)
- Inference batch processing with garbage collection and CUDA cache clearing
- Dataloader workers and dataset preprocessing for I/O efficiency

**Code Organisation:**
- Modular PaliGemmaExplainer class separating model loading, adapter integration, attention extraction, visualisation generation, and batch evaluation orchestration
- Separation between training pipeline and explanation analysis

### Research Contributions

**Methodological:**
- Attention extraction from VLM internal states (not simulated heatmaps)
- Multiple explanation modalities: attention maps, Grad-CAM, confidence scores, medical term grounding
- Evaluation framework for multimodal explanation faithfulness in high-stakes domains
- Batch processing infrastructure enabling quantitative analysis

**Conceptual:**
- Frames VLM explainability as causal inference problem rather than visualisation task
- Provides testbed for investigating whether language explanations can be decoupled from visual evidence
- Framework designed to generalise beyond GI imaging to other high-stakes VLM applications

**Clinical Safety:**
- Confidence calibration reveals when model is uncertain
- Medical term extraction identifies clinically relevant predictions
- Visual explanation validation against anatomical expectations
- Foundation for detecting when explanations may mislead clinicians (over-reliance studies)

### Integration with Original Segmentation Work

The VLM-VQA direction complements the polyp segmentation work by:
- Shared explainability focus: Both investigate whether AI explanations reflect genuine reasoning
- Progressive complexity: Segmentation (pixel-level) to VQA (semantic reasoning level)
- Clinical deployment pathway: Segmentation identifies polyps; VQA answers diagnostic questions about them
- Unified evaluation framework: Quantitative faithfulness metrics span both modalities

This positions the overall research programme as an investigation of explainability in medical AI, from low-level vision to high-level multimodal reasoning.

---

*Recent work demonstrates that VLMs can be fine-tuned for specialised medical VQA tasks and that their internal representations can be extracted for causal analysis. The infrastructure developed enables systematic investigation of whether multimodal explanations genuinely reflect model reasoning or constitute post-hoc rationalisations. This question is relevant for safe deployment of AI in clinical practice, though further evaluation is required.*
