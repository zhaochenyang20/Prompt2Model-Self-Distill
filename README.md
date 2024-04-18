# Prompt2Model - Generate Deployable Models from Instructions

[![PyPI version](https://badge.fury.io/py/prompt2model.svg)](https://badge.fury.io/py/prompt2model)
![Github Actions CI tests](https://github.com/neulab/prompt2model/actions/workflows/ci.yml/badge.svg)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![Discord](https://img.shields.io/discord/1144245269001678959)](https://discord.gg/UCy9csEmFc)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neulab/prompt2model/blob/main/prompt2model_demo.ipynb)

We introduce `SELF-GUIDE`, a novel methodology that enables LLMs to better execute task-specific instructions without requiring additional data or training signals. `SELF-GUIDE` operates in the few-shot setting, where we are given only a task instruction and around 3 examples of task demonstrations. `SELF-GUIDE` works by first employing the target model to generate a synthetic dataset for a given task. The model is then finetuned on this “self-generated’’ data.

<img width="360" alt="prompt2model_teaser" src="self_guide_diagram.png">

## Quick Start

### Dataset

We use tasks from [NaturalInstructions V2](https://arxiv.org/abs/2204.07705). For each task, we have task instruction and example input-output pairs according to the dataset.

According to our one param fits all parameters, you could use them to self-generate dataset and finetune on the dataset to improve its performance.

Selected tasks including (The validation and test set can be accessed from NI_dataset folder)
- Generation tasks: task121, task039, task036, task1195, task1345, task281, task1562, task1622
- Classification tasks: task190, task199, task200, task738, task937, task1385, task1386, task1516, task1529, task1612, task1615, task284, task329, task346
### Notebook

You can run our demo of `Prompt2Model` through a notebook:

- [Open Locally](./prompt2model_demo.ipynb)
- [Open in Colab](https://colab.research.google.com/github/neulab/prompt2model/blob/main/prompt2model_demo.ipynb)

### Command Line

You can also run through the command line.

```bash
pip install prompt2model
```

`Prompt2Model` supports various platforms such as OpenAI, Anthropic, Huggingface, etc. using [LiteLLM](https://github.com/BerriAI/litellm).

If you are using OpenAI models (such as the default `gpt-3.5-turbo`), please obtain an
OpenAI API key on their [website](https://platform.openai.com/) then set
the environment variable `OPENAI_API_KEY` to your API key by running
the following command in your terminal:

```bash
export OPENAI_API_KEY=<your key>
```

[List of all supported providers](https://docs.litellm.ai/docs/providers)

You can then run

```bash
python prompt2model_demo.py
```

to create a small model from a prompt, as shown in
the demo video below. This script must be run on a
device with an internet connection to access the OpenAI
API. For best results, run
this script on a device with a GPU for training
your model.

## Contribution

If you're interested in contributing to the `prompt2model` project, please

- refer to [CONTRIBUTING.md](CONTRIBUTING.md)
- open an [issue](https://github.com/neulab/prompt2model/issues) or submit a PR
- join us on [discord](https://discord.gg/UCy9csEmFc)
- or reach out to [@vijaytarian](https://twitter.com/vijaytarian)
  and [@Chenan3_Zhao](https://twitter.com/Chenan3_Zhao) on Twitter

