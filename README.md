# Final Project — Reinforcement Learning for LLM Code Generation

## Authors

Authors

👤 Amir Hosseinpour (Student ID: 40117393)

👤 Mani Vafapour (Student ID: 40123783)

Course

Fundamentals of Intelligent Systems — Instructor: 👨‍🏫 Dr Aliyari

---

## Project Overview

This repository contains the final project for the course. The project investigates reinforcement-learning-driven reasoning in large language models (LLMs) specifically applied to automated code generation and repair. We implement and compare two RL architectures — an **Executive** model trained to handle concrete code-generation episodes, and a **Hierarchical** model that plans and coordinates multiple subtasks using the Executive as an internal worker.

Key techniques and topics covered in the course and used throughout the project include: PCA, LDA, t-SNE, UMAP for analysis/visualization; classical ML algorithms (Decision Trees, Random Forests, Logistic Regression, SVM, K-means); genetic algorithms; and modern RL approaches. The RL methods used in this project are Double DQN and Dueling DQN with an epsilon-greedy exploration strategy and action masking.

---

## Highlights

* Two RL architectures for LLM-driven code generation and debugging: **Executive** and **Hierarchical**.
* Algorithms: **Double DQN** and **Dueling DQN**.
* Exploration: **ε-greedy** policy.
* Safety/constraints: **Action masking** to prevent invalid actions.
* Training outcomes: **Executive model — 100% success** on designed episodes; **Hierarchical model — 91% success** on end-to-end tasks.

---

## Model Responsibilities and Episodic Definitions

### Executive Model (worker)

The Executive model is trained to handle short, well-defined episodes of code generation and repair. It learns to perform sequences of actions that move from prompt → working code → tests to passing state. There are three explicit episode templates used for training and evaluation:

1. `generate code -> test pass -> stop`
2. `generate code -> test fail -> debug -> test pass -> stop`
3. `generate code -> test fail -> debug -> test fail -> search -> debug -> test pass -> stop`

The Executive model is responsible for generating code, debugging it when tests fail, and (in more complex episodes) invoking a search action before additional debugging attempts.

### Hierarchical Model (planner + executive)

The Hierarchical model contains a planning layer and uses the Executive as the execution layer. Its single episode template is:

`planning -> execution -> integration -> evaluation -> stop`

* **Planning**: the LLM receives a complex prompt and decomposes it into one or more prompts for subtasks (subtask generation / task decomposition).
* **Execution**: each subtask prompt is handled by the Executive model, which generates and (if necessary) corrects code until subtask-level tests pass. The Executive is invoked as a callable policy inside the hierarchical pipeline.
* **Integration**: the LLM collects outputs from all subtask executions and merges them into a single, cohesive codebase (one unique code file or module). This step resolves interface mismatches and composes sub-solutions into an integrated program.
* **Evaluation**: the integrated code is evaluated by calling the Executive model in an evaluation mode where the Executive is driven by the RL policy but receives the integrated code as input to the generation/evaluation pipeline (i.e., it does *not* regenerate everything from scratch). If integration tests pass, the episode stops; otherwise further corrective cycles may be run depending on the hierarchical policy.

---

## Reinforcement Learning Details

* **Algorithmic variants**: both Double DQN and Dueling DQN variants were implemented and compared to stabilize value estimates and improve policy learning.
* **Exploration**: ε-greedy schedule used during training; decays over episodes to favor exploitation once stable policies emerge.
* **Action Masking**: invalid or nonsensical actions (e.g., repeated illegal transitions) are masked out at decision time, improving learning efficiency and preventing catastrophic choices.
* **Two-model training**: the Executive is trained on the three short episode templates and then frozen/queried by the Hierarchical model during planning/execution training.

---

## Results

* **Executive model**: 100% success rate on the three designed episode types (code generation and repair tasks used during training and validation).
* **Hierarchical model**: 91% success rate on end-to-end decomposition, execution, integration, and evaluation tasks (complex prompts split across subtasks).

These metrics reflect how often the final integrated code passes the test suite for each evaluated prompt. The Executive’s perfect score highlights the effectiveness of focused RL training on canonical episodes; the Hierarchical model’s slightly lower score reflects the added difficulty of decomposition and integration.

---

## Included Notebooks and Key Files

* `RL_3Episode.ipynb` — Executive model implementation and training on the three episodes.
* `RL_Hierarchical.ipynb` — Hierarchical pipeline implementation, including planning, integration, and evaluation orchestration.
* `Combination_of_two_models.ipynb` — Scripts demonstrating combined execution, evaluation runs, and experiments that coordinate Executive and Hierarchical models.

(Please refer to the notebooks for code-level details, experiment logs, model checkpoints, and visualization cells.)

---

## Quick Start — Run the experiments

1. Clone the repository.
2. Create a virtual environment and install dependencies (see `requirements.txt` — if not present, install common ML/RL packages such as `torch`, `transformers`, `numpy`, `pandas`, `scikit-learn`, and Jupyter-related packages).

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab  # or jupyter notebook
```

3. Open the notebooks listed above and run cells in order. Training cells will log episode returns and accuracy metrics. For reproducible runs, set the provided random seeds in the top cells of each notebook.

**Hardware recommendation**: GPU is highly recommended for LLM token generation and RL training loops. Use available CUDA device if present.

---

## Configuration and Hyperparameters

Each notebook contains a configuration block near the top where hyperparameters can be adjusted:

* RL algorithm selection (Double vs Dueling DQN)
* Learning rate, batch size, replay buffer size
* ε-greedy schedule parameters
* Masking rules and action space definitions
* Model checkpoint paths

Tweak these to reproduce experiments or to run ablations.

---

## How the Pipeline Is Designed (high level)

1. **Prompt ingestion** → Planner decomposes into subtasks (if hierarchical flow used).
2. **Subtask execution** → Executive model handles each subtask episode until its subtask test passes.
3. **Integration** → Outputs are merged; interfaces resolved into a single program.
4. **Evaluation** → Executive model is used in evaluation mode (RL-driven) on the integrated code; pass/fail determines success.

This design separates responsibilities (planning vs execution) to let a focused Executive learn robust low-level behaviors while a planner learns high-level orchestration.

---

## Reproducibility & Logging

* Training and evaluation cells produce logs of episode returns, step counts, and pass/fail traces. Look for saved checkpoints and `results/` folders in the notebooks' cells.
* Random seeds are set for reproducibility; however, generation nondeterminism from LLMs and asynchronous hardware scheduling may produce slight variability.

---

## Contribution and Extensions

This repo is designed as an academic project and a starting point for further research. Possible extensions include:

* Using larger or different LLMs for the Executive and Planner roles.
* Joint end-to-end training of Planner + Executive instead of staged training.
* More sophisticated planning (hierarchical RL with learned termination conditions, subgoal discovery).
* Better integration strategies for multi-file projects and cross-module APIs.

---

## Contact & Credits

For questions about the experiments or to request clarifications about the notebooks, please reach out to the authors listed above.

---

## License

Specify your preferred license here (for example, MIT). If you do not have a license yet, consider adding one to clarify reuse terms.

---

*This README summarizes the final project: reinforcement-learning-guided reasoning for code generation in LLMs. For implementation details and to inspect training routines and checkpoints, open the included Jupyter notebooks.*
