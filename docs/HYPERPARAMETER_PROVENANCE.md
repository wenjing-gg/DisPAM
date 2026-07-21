# Hyperparameter Provenance

No formal hyperparameter sweep was conducted. Each parameter used one
prespecified value without a search range. Final settings use three provenance
categories.

| Parameter group | Source | Final-setting principle |
| --- | --- | --- |
| Student architecture depth, width, and block configuration | Reference-implementation defaults | VISTA3D, PrimusV2, SegMambaV2, and U-RWKV retain the settings used by their released reference implementations. |
| PAM input size and crop protocol | Prior work | The input size of 224 and crop ratio of 1.5 follow the frozen PAM inference protocol. |
| PASE smoothing and VLM decisions | Fixed design | The smoothing scale, three repeated decisions, majority rule, and top-three fallback are fixed for every dataset and target. |
| Prompt perturbation and Core construction | Fixed design | One-pixel Dilation and Erosion, a binarization threshold of 0.5, and a confidence threshold of 0.7 are applied uniformly. |
| Training crop and sampling | Fixed design | The 96 x 96 x 96 crop, foreground-aware sampling, and Core sampling probabilities are shared by all students. |
| Dual-distillation optimization | Fixed design | AdamW, cosine annealing, 100 epochs, and the released learning-rate and weight-decay settings define one common protocol. |
| CDFT optimization and region construction | Fixed design | Ten epochs, three-of-four Consensus support, population-standard-deviation filtering, six-connectivity, and the released descriptor settings are fixed globally. |
| Sliding-window prediction | Established convention | An overlap of 0.5, four-model probability averaging, and a threshold of 0.5 are used for every target. |
| Repeated runs and reporting | Evaluation protocol | Five independent seeds are averaged per case before the mean and population standard deviation are computed across cases. |
