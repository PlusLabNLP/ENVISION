# ENVISION: Localizing Active Objects from Egocentric Vision with Symbolic World Knowledge
[![LICENSE](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.6-blue)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.5-yellow)


**[Localizing Active Objects from Egocentric Vision with Symbolic World Knowledge](https://arxiv.org/pdf/2310.15066v1.pdf)**<br>
<em>Te-Lin Wu<sup>*</sup>, Yu Zhou<sup>*</sup> (equal contribution), Nanyun Peng</em> <font color="red"> <br/>
EMNLP 2023 (Oral) <br>

## Abstract
The ability to actively ground task instructions from an egocentric view is crucial for AI agents to accomplish tasks or assist humans. One important step towards this goal is to localize and track key active objects that undergo major state change as a consequence of human actions/interactions in the environment (e.g., localizing and tracking the ‘sponge‘ in video from the instruction "Dip the sponge into the bucket.") without being told exactly what/where to ground. While existing works approach this problem from a pure vision perspective, we investigate to which extent the language modality (i.e., task instructions) and their interaction with visual modality can be beneficial. Specifically, we propose to improve phrase grounding models’ ability in localizing the active objects by: (1) learning the role of objects undergoing change and accurately extracting them from the instructions, (2) leveraging pre- and post-conditions of the objects during actions, and (3) recognizing the objects more robustly with descriptional knowledge. We leverage large language models (LLMs) to extract the aforementioned action-object knowledge, and design a per-object aggregation masking technique to effectively perform joint inference on object phrases with
symbolic knowledge. We evaluate our framework on Ego4D and Epic-Kitchens datasets. Extensive experiments demonstrate the effectiveness of our proposed framework, which leads to > 54% improvements in all standard metrics on the TREK-150-OPE-Det localization + tracking task, > 7% improvements in all standard metrics on the TREK-150-OPE tracking task, and > 3% improvements in average precision (AP) on the Ego4D SCOD task.

[![Spotlight Video](./media/spotlight.png)](https://www.youtube.com/watch?v=t_yDXUriRZo)

[Twitter](https://x.com/yu_bryan_zhou/status/1717724445715128628?s=20)<br>

## Reproduce
### Requirements

The models in this paper are runnable on a single Nvidia V-100 GPU and CUDA Version: 12.0. <br><br>
Please see [environment.yml](https://github.com/PlusLabNLP/ENVISION/blob/main/requirements.txt) for specific package requirements.<br><br>

---

### ENVISION

For data processing and LLM knowledge extraction, please refer to [data_processing](https://github.com/PlusLabNLP/ENVISION/tree/main/data_processing). For model training and inference using ENVISION, please refer to [modeling](https://github.com/PlusLabNLP/ENVISION/tree/main/modeling). For visualizing data and ENVISION inference results, please refer to [visualization](https://github.com/PlusLabNLP/ENVISION/tree/main/visualization). For model training and inference using ENVISION, please refer to [baselines](https://github.com/PlusLabNLP/ENVISION/tree/main/baselines).<br><br>

---


## BibTeX

If you find the code in this repo useful, please consider citing our paper:

```
@article{wu2023localizing,
  title={Localizing active objects from egocentric vision with symbolic world knowledge},
  author={Wu, Te-Lin and Zhou, Yu and Peng, Nanyun},
  journal={arXiv preprint arXiv:2310.15066},
  year={2023}
}
```

