# AI Social Scientist: LLM-powered Interview Research

## Overview

This project explores the use of Large Language Models (LLMs) to bridge the gap between quantitative scale and qualitative depth in social science research. While traditional interviews provide rich insights but are resource-intensive, LLMs offer a potential complementary approach for gathering preliminary insights into public sentiment and policy reception.

This system is designed to serve as a preliminary tool for policymakers to:
- Gather initial insights into public reception of proposed initiatives
- Anticipate potential policy implementation challenges
- Provide a cost-effective first step in understanding public sentiment

This repository is forked and adapted from the [AI Scientist](https://github.com/SakanaAI/AI-Scientist).
## Project Structure

The workflow consists of three main stages:

### 1. Idea Generation Stage
- System is prompted as an "ambitious policymaker" focused on solving problems for society
- Generates problems and potential solutions using Semantic Scholar API
- Identifies prevalent societal issues from seed ideas

### 2. Interview Stage
The system:
- Plans relevant interview questions
- Creates target personas for interviewing
- Conducts interviews using dual GPT-4 instances:
  - One instance acts as interviewer
  - One instance acts as interviewee based on created persona
- Generates summaries focusing on key themes and insights
- Iteratively replans questions and creates new personas based on previous interviews

### 3. Writeup Stage
- Synthesizes interview findings
- Generates coherent arguments from discussion points
- Focuses on describing interviewees and their perspectives

## Technical Implementation

The interview process is implemented through `templates\interview\experiment.py`, which:
- Manages two GPT-4 instances for interviewer and interviewee roles
- Ingests predefined questions and personas
- Continues until turn limit is reached or sufficient information is gathered
- Generates summary reports of key themes and insights

**Note**: This system is intended to complement, not replace, traditional human interviews in social research.

## Requirements
The installation instructions can be found in the original [AI Scientist repo](https://github.com/SakanaAI/AI-Scientist).

## Examples
Refer to `results\interview` or `results\interview_japan` for papers written by the AI Social Scientist.

## Run AI Social Scientist Paper Generation Experiments

```bash
conda activate ai_scientist
# Run the paper generation.
python launch_scientist.py --model "gpt-4o-2024-05-13" --experiment interview --num-ideas 2
```

If you have more than 1 GPU, use the `parallel` option to parallelize ideas across multiple GPUs.


## Citing The AI Scientist

Please refer to the original work!
If you use **The AI Scientist** in your research, please cite it as follows:

```
@article{lu2024aiscientist,
  title={The {AI} {S}cientist: Towards Fully Automated Open-Ended Scientific Discovery},
  author={Lu, Chris and Lu, Cong and Lange, Robert Tjarko and Foerster, Jakob and Clune, Jeff and Ha, David},
  journal={arXiv preprint arXiv:2408.06292},
  year={2024}
}
```
