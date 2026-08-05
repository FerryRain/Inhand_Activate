## Meta Review of Submission416 by Area Chair u3rD
Meta Reviewby Area Chair u3rD20 Jul 2026, 13:37 (modified: 05 Aug 2026, 05:31)Senior Area Chairs, Area Chairs, Authors, Reviewers, Program ChairsRevisions
Metareview:
The reviewers agree that AURORA addresses an interesting problem and presents a clear, technically coherent system for active in-hand reconstruction. The Ray-GPIS formulation is well explained, and the real-robot experiments show improved reconstruction over non-active baselines. The main concerns centre on evaluation and positioning. Reviewers request stronger comparisons with related active reconstruction and next-best-view methods, including ActNeRF and PB-NBV, or clearer justification where direct comparison is infeasible. Several also ask for ablations of uncertainty estimation, novelty weighting, ray aggregation, and other pipeline components. It remains unclear whether the improved reconstruction yields a meaningful downstream benefit, or whether a simpler pose-conditioned reorientation strategy could perform similarly. Additional concerns include the limited object diversity, lack of robustness analysis to pose-tracking errors, relatively slow replanning, and missing runtime breakdowns. Overall, the work is viewed as promising, but the current evidence does not fully isolate or establish the practical value of its central design choices.

Pre-Rebuttal Recommendation: Proceed to rebuttal: The paper has at least one score of weak accept or above, or the AC believes the paper warrants author response.
Key Issues for Rebuttal:
The rebuttal should primarily clarify the paper's positioning relative to prior active reconstruction and next-best-view methods, including why comparisons with approaches such as ActNeRF or PB-NBV are not included or are not directly applicable. It should also justify the practical value of the proposed pipeline by explaining the downstream benefits of improved reconstruction and whether simpler pose-conditioned reorientation strategies would achieve similar performance. Finally, the authors should address concerns regarding the limited evaluation, including missing ablations, robustness to pose estimation errors, runtime, and the limited object diversity.

## Official Review of Submission416 by Reviewer BKKY
Official Reviewby Reviewer BKKY14 Jul 2026, 01:37 (modified: 05 Aug 2026, 05:29)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers, AuthorsRevisions
Summary:
This paper, AURORA, proposes an active 3D reconstruction framework that leverages in-hand manipulation for object inspection. Specifically, it 1) tracks the grasped object’s 6D pose from segmented RGB-D observations 2) transforms selected depth frames into a common object coordinate system to incrementally fuse a partial point cloud 3) uses Ray-GPIS to estimate uncertainty and novelty along object-centered candidate viewing rays 4) selects the next-best-view corresponding to the most uncertain and insufficiently observed surface region 5) maps that desired view to the closest feasible in-hand rotation primitive, executes the tactile-based rotation policy. For the experiments, six objects are tested and relatively good performance is shown during the execution.

Strengths:
This motivation of breaking policy into high-level object pose identification and low-level hand motions is very promising especially in complicate tasks such as in-hand manipulation.

Writing and visualization are very clear to show the experiments and methodology.

The code sign of the in-hand manipulation and object reconstruction is interesting, where two tasks can benefit from each others.

Weaknesses:
It would be better if author could show more details about ablation module of each design in the methodology pipeline.
It would be better if author could show the performance comparison between directly using object pose to steering the rotation policy. Currently, there is some pipeline like Hora that already shows its capability of rotating objects through any axises, So it would be better to compare AURORA to overlaying an object pose identification model to get object poses and steering low-level rotation policy. In their experiments, a certain amount of object generalization has already been shown, so it would be better to show whether the mesh reconstruction is useful in this task.
frequency of replanning seems to be too low, for during actually robot rotation, 6 seconds can happen a lot.
More experiments are how AURORA deals with errors in object 6D poses tracking will be very meaningful.
Questions For Authors:
Please refer to the weakness.

Limitations And Broader Impact:
Yes.

No.

Overall Score: 5: Weak accept. A good paper with merit that outweighs its weaknesses. The contribution is valid but may have gaps in evaluation, limited novelty over prior work, or clarity issues that the authors could address.
Confidence Score: 4: High confidence. I am knowledgeable in this area and confident in my assessment.
Ethical Concerns: None
LLM Disclosure: No

## Official Review of Submission416 by Reviewer dWbG
Official Reviewby Reviewer dWbG13 Jul 2026, 15:19 (modified: 05 Aug 2026, 05:29)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers, AuthorsRevisions
Summary:
The paper presents a system for active object reconstruction using in-hand re-orientation. It shows a closed-loop pipeline that first estimates the reconstruction uncertainty, then uses that to pick the next-best-view direction, which is in turn used to find the best orientation for the next in-hand object rotation, exposing that region to a fixed RGB-D camera. The core component of the paper is the Ray-GPIS algorithm, which casts rays from the object center and turns the GPIS uncertainty along each ray into a score for that direction, which is then used to calculate the axis for the next in-hand rotation. The approach shows better reconstruction quality and faster uncertainty reduction compared to non-active baselines.

Strengths:
The presented formulation is algorithmically sound, and Ray-GPIS is derived clearly.
The pipeline is well constructed, connecting pose tracking, keyframe filtering, incremental fusion, uncertainty estimation, and NBV-to-action mapping into a working closed loop, with implementation details and hyperparameters documented.
Weaknesses:
The general motivation for active perception that the paper starts with, and how it is used by humans to figure out unknown parts of an object, makes sense. However, using that motivation to justify this specific system is less convincing, for the following reason: the paper demonstrates that active planning and viewpoint selection improve reconstruction quality, but it remains unclear what downstream benefit this improved reconstruction quality provides for any task. As a result, for a novel system, the practical significance of the proposed pipeline is somewhat under-motivated.
The core contribution, the Ray-GPIS algorithm, is mostly independent from the in-hand manipulation setting. It operates on a fused point cloud, and its output to the controller is a rotation axis selected from a small discrete set {−x, −y, +z}, while the in-hand rotation policy from [1] is used as a plug-and-play axis-conditioned controller. The planning algorithm would in principle function identically if the requested rotation were executed by another mechanism, so the reason for using in-hand rotation here is not clear to me.
Building on the above point, since the primary contribution is the planning strategy rather than the low-level controller, it is unclear why an algorithm-level comparison is not possible, for example against ActNeRF[2], which does something similar.
[1] Yin, Zhao-Heng, et al. "Rotating without seeing: Towards in-hand dexterity through touch." arXiv preprint arXiv:2303.10880 (2023).

[2] Dasgupta, Saptarshi, et al. "Uncertainty-aware active learning of nerf-based object models for robot manipulators using visual and re-orientation actions." arXiv preprint arXiv:2404.01812 (2024).

Questions For Authors:
It would be helpful if the authors could clarify the points mentioned in the weaknesses section, specifically the intended downstream use of the reconstruction, motivation the in-hand reorientation setting, and why an algorithm-level comparison against a method like ActNeRF is not feasible.

Limitations And Broader Impact:
Limitations are adequately discussed. No ethical concerns.

Overall Score: 4: Borderline. The paper has interesting ideas but notable weaknesses — e.g., insufficient experiments, unclear contribution, or limited novelty. It is unlikely authors could address all the issues in the limited period of the rebuttal.
Confidence Score: 3: Moderate confidence. I am familiar with the area but not an expert; some aspects may be outside my expertise.
Ethical Concerns: None
LLM Disclosure: Yes — please describe below

## Official Review of Submission416 by Reviewer 4zfY
Official Reviewby Reviewer 4zfY08 Jul 2026, 01:08 (modified: 05 Aug 2026, 05:29)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers, AuthorsRevisions
Summary:
This paper introduces AURORA, an active in-hand object reconstruction framework that couples online object-centric reconstruction with uncertainty-driven reorientation. The central technical contribution is Ray-GPIS, a viewing-ray-conditioned uncertainty estimation module that estimates direction-wise view uncertainty. By actively selecting in-hand rotation actions that expose under-observed regions, AURORA achieves more complete reconstructions than open-loop rotation baselines in real-world experiments.

Strengths:
The paper studies active in-hand object reconstruction, which is a meaningful and interesting problem.
The real-world qualitative and quantitative experiments demonstrate the effectiveness of the proposed method over non-active baselines.
Weaknesses:
The paper could better discuss its relation to prior active reconstruction / NBV planning works [paper 1-3]. Although these works are not specifically about in-hand reconstruction, they also use active manipulation or view planning to improve reconstruction. A clearer discussion of the differences, or possible comparisons if feasible, would strengthen the paper.
More ablation studies would be helpful. For example, the authors could ablate key Ray-GPIS components, such as uncertainty estimation, novelty weighting, and ray-based aggregation, to better support the design choices.
While the experiments demonstrate clear improvements over the baselines, the object diversity is relatively limited. Evaluation on more challenging objects (for example, irregularly shaped objects or objects with concavities), would better demonstrate the generality of the method.
[1] PB-NBV: Efficient Projection-Based Next-Best-View Planning Framework for Reconstruction of Unknown Objects

[2] Scalable Real2Sim: Physics-Aware Asset Generation Via Robotic Pick-and-Place Setups

[3] Uncertainty-aware Active Learning of NeRF-based Object Models for Robot Manipulators using Visual and Re-orientation Actions

Questions For Authors:
Could the authors provide a runtime breakdown for each module?

Limitations And Broader Impact:
The paper discusses its limitations. I do not see major ethical concerns or potential negative societal impacts.

Overall Score: 3: Weak reject. Below the acceptance threshold. The paper has identifiable merit but significant weaknesses — e.g., missing key comparisons, unconvincing results, or incremental contribution.
Confidence Score: 3: Moderate confidence. I am familiar with the area but not an expert; some aspects may be outside my expertise.
Ethical Concerns: None
LLM Disclosure: Yes — please describe below
LLM Disclosure Details:
I used ChatGPT only for language polishing
