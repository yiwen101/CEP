Here is the text-only version of the provided document.

Page 1
Enhancing LLM Reasoning via Structured Context Elaboration:

A Multi-Faceted Approach for In-Context, Retrieval, and Latent Space Methods

Page 2
Table of Contents

Motivation
Core Hypothesis & Idea
Closely Related Works
Proposed Methods
Proposed Experiment
A Brief Mention of My Other Idea: A Monte Carlo scorer optimised RLVR method
Page 3
1. Motivation

LLMs excel at known facts and simple in-context tasks. BUT performance degrades significantly when tasks require synthesizing novel, complex information. This problem exists in two key areas:

In-Context Reasoning: Failing to use dense information provided in a single prompt.
Retrieval-Augmented Generation (RAG): Failing to connect ideas across multiple retrieved documents.
Chain-of-Thought (CoT) requires demonstration and usually human prompting. Moreover, it primarily addresses the execution path of reasoning but neglects the foundational step of context assimilation.

Page 4
2. Core Assumption & Idea

I hypothesize that an LLM's ability to solve complex problems is limited by its context assimilation ability.
I hypothesize that a process of structured elaboration on contextual information applied during knowledge ingestion (aka germane load, which is mental resource spent on forming schema and making connection with existing long term memory) can:
Improve the model's use of information within a single context window.
Create a more structured, interconnected, and useful memory representation that significantly enhances future retrieval and synthesis across multiple interactions.
(The slide includes a diagram illustrating three types of cognitive load from learning theory):

Intrinsic load: Immutable by instructional intervention.
Extraneous load: Leads to ineffective learning.
Germane load: Leads to effective learning.
Page 5
2. Core Assumption & Idea

Formally, my core hypothesis is that: the probability of generating a correct output, P(S_correct | C, q), can be significantly increased by conditioning the model on an intermediate, self-generated Elaborated Context (E_c).
This E_c is a structured representation of the raw context C, generated via a Context Elaboration Prompt (CEP).
E_c can be generated in real-time within a prompt, pre-computed and stored to build a semantically rich memory, or distilled into a latent vector for efficient, non-textual steering.
Page 6
2. Core Assumption & Idea

I hypothesize that E_c enhances performance through two primary mechanisms:

Explicit State Representation: The E_c acts as an externalized "scratchpad." It transfers key entities, relationships, and synthesized facts from being implicitly tracked in the model's attention mechanism to being explicitly represented as text. This simplifies the computational task of state-tracking during final reasoning.
Conditioning for High-Probability Trajectories: The elaboration step primes the model's generative process. By first articulating the underlying structure and key principles of the context in E_c, the model is conditioned to follow a reasoning path that is logically consistent with that structure, thereby constraining the output space to more relevant and correct trajectories.
Page 7
3. Closely Related Works

Paper	Related Finding
LARGE LANGUAGE MODELS AS ANALOGICAL REASONERS	Ask AI to come up with and solve similar problems to improve QA performance.
Large Language Models are In-context Teachers for Knowledge Reasoning	Use AI generated explanation to augment demo QA pair to improve performance.
Training With “Paraphrasing the Original Text” Teaches LLM to Better Retrieve in Long-context Tasks	Adding a “paraphrasing” step when finetuning a model on information retrieval can improve eventual performance.
Page 8
3. Closely Related Works

LARGE LANGUAGE MODELS AS ANALOGICAL REASONERS

(This slide shows a diagram comparing different prompting techniques.)

0-shot: A direct question is given to the model.
0-shot CoT: The model is prompted with "Think step by step." This is described as "Generic guidance of reasoning".
Few-shot CoT: The model is given a complete example (question and reasoned answer) before the target question. This is described as "Need labeled exemplars of reasoning".
Analogical Prompting (Ours): The model is instructed to first "Recall relevant exemplars" and then "Solve the initial problem". The model then generates its own relevant example and uses it to solve the problem.
Key Idea: "Asking AI to propose and solve similar problems improves QA performance" and is "tailored to each problem".
Page 9
3. Closely Related Works

Training With “Paraphrasing the Original Text” Teaches LLM to Better Retrieve in Long-context Tasks

(This slide shows a diagram explaining a training data preparation method.)

Reference Context and Question: A long piece of text (e.g., Document-[20]) is provided along with a specific question ("Who was the first pilot to achieve 1,000 flight hours in the F-35 Joint Strike Fighter?").
Answer Design: For training, different types of answers are created from the context. The key method highlighted is "Original Text Paraphrasing", where the model is trained on a paraphrased version of the sentence containing the answer, rather than just the short answer itself. For example, instead of just "David 'Doc' Nelson", the training answer is a full sentence paraphrasing the original text from the document.
Page 10
3. Closely Related Works

Training With “Paraphrasing the Original Text” Teaches LLM to Better Retrieve in Long-context Tasks

Table 2: Performance of different training methods

Models	MultiDoc	SingleDoc	Synthetic	Summarization	FewShot	AVG
Qwen1.5-4b-Chat	65.06	77.70	14.75	19.27	57.00	46.76
Qwen1.5-4b-Chat SFT w/ ours	65.39	72.22	62.00	19.57	57.00	55.24
Qwen1.5-4b-Chat SFT w/o ours	65.28	71.86	12.75	19.84	56.50	45.25
Qwen1.5-4b-Chat SFT w/ Ziya	65.32	76.28	25.75	19.50	56.88	48.75
Qwen2-7b-Instruct	60.89	80.32	63.00	19.62	60.00	56.77
Qwen2-7b-Instruct SFT w/ ours	65.80	82.17	65.00	20.34	61.50	58.96
Qwen2-7b-Instruct SFT w/o ours	66.49	82.12	56.75	16.98	61.75	56.82
Qwen2-7b-Instruct SFT w/ Ziya	65.42	82.40	61.50	19.88	62.50	58.34
Llama3-8b-chinese-Chat	64.04	79.20	80.75	18.56	50.62	58.63
Llama3-8b-chinese-Chat SFT w/ ours	65.22	80.92	97.75	21.17	57.38	64.39
Llama3-8b-chinese-Chat SFT w/o ours	66.21	...	...	...	...	...
Llama3-8b-chinese-Chat SFT w/ Ziya	63.80	...	...	...	...	...
Key Takeaway: Adding a "paraphrasing" step (w/ ours VS w/o ours) when finetuning a model to better retrieve information improves performance.

Page 11
3. Closely Related Works

Large Language Models are In-context Teachers for Knowledge Reasoning

(This slide shows a diagram of a two-LLM system.)

A Teacher LLM is given a question-answer pair and is prompted to "Explain how to reach this answer."
The Teacher LLM generates an explanation (e.g., "Explanation: Ethambutol is a medication that is used to treat tuberculosis...").
This AI-generated explanation is then added to the in-context examples provided to a Student LLM.
The Student LLM uses these augmented examples to answer a new test question.
Key Idea: Use AI generated explanation to augment demo QA pair improves performance.

Page 12
4. Proposed Methods

Method 1: Real-Time In-Context Elaboration (ICE)

1.1 Elaborate with Intention: Given a context C and query q, a CEP is added between context and query to generate prompt [Context C] + [CEP] + [Query q], then directly query model with prompt.
1.2 Elaborate without Intention (model is unaware of query during the elaboration process):
Turn 1 (Elaboration): prompt = [Context C] + [CEP]; resp_E_c = LLM(C, CEP).
Turn 2 (Execution): query model with history (prompt, resp_E_c) and Query q.
Method 2: In-Context Elaboration as context augmentation method (ICE)

Stage 1 (Elaboration): Given a context C and query q, a CEP is used to generate a real-time E_c. E_c = LLM(C, CEP).
Stage 2 (Execution): The final answer is generated from a prompt containing the original context, the newly generated E_c, and the query.
Final_Prompt = [Context C] + [Elaborated_Context E_c] + [Query q]
Page 13
4. Proposed Methods

Method 3: Germane-Enriched Retrieval (GER) (yet to do more literature review in RAG)

This method redesigns the RAG pipeline to create a persistent, semantically rich knowledge base, moving the elaboration step to an offline ingestion phase.

Phase 1 (Offline Ingestion):
Chunk: A document is segmented into raw chunks (C_raw).
Elaborate: For each C_raw, an LLM generates a corresponding Elaborated Chunk (C_elab) using a CEP.
Store & Index: Both C_raw and C_elab are stored, and their corresponding embeddings (v_raw, v_elab) are indexed.
Phase 2 (Online Retrieval & Synthesis):
Hybrid Retrieval: A user query is used to search against both v_raw and v_elab, allowing for both literal and conceptual matching.
Prompt Construction: The retrieved, pre-digested elaborated chunks (C_elab) are used to construct the final prompt, providing the LLM with high-quality, synthesized information.
Page 14
4. Proposed Methods

Method 4: Elaboration-Enriched Vectors (EEV)

This method seeks to capture the benefits of elaboration without the token overhead of including E_c in the final prompt. It adapts the principles of In-Context Vectors (ICV) to our framework.

Phase 1 (Offline Ingestion & Vector Creation):
Elaborate: For each C_raw, generate its C_elab as in the GER method.
Concatenate: Create a combined text string: C_combined = [C_raw] + [C_elab].
Generate Vector: Perform a forward pass with C_combined through the LLM and extract the hidden state activations from a chosen layer. This activation is the Elaboration-Enriched Vector (v_ee).
Store: Store the raw chunk C_raw and its corresponding v_ee.
Phase 2 (Online Retrieval & Latent Steering):
Retrieve: A user query is used to search against the v_ee vectors, retrieving the most relevant raw chunks (C_raw).
Steer & Generate: The retrieved C_raw chunks are placed in the prompt. The corresponding v_ee vectors are then used to directly shift the LLM's latent states at specified layers during generation, steering the model's output towards a path consistent with the pre-digested elaboration, without explicitly adding the elaboration text to the prompt.
Page 15 & 16
4. Proposed Methods

Suggested Germane Questions

Category	Prompt
Understand	"Paraphrase the provided information in your own words", "summarize the given text"
Connect	"What does this information remind you of? Briefly explain the connection."
Query	"what you find to be the most surprising or interesting piece of information". "formulate two insightful questions that are raised by the text"
Application	"What can you deduce from the given information?" "formulate two insightful questions that are answered by the information given"
(A note on slide 16 adds a thought connection: "curriculum learning paper shows gradually increase complexity optimise performance")

Page 17
5. Proposed Experiment

QA Experiments

Methods:
Baseline Table: (Shows existing results for methods like COT, ToT, Analogical, Self-Refine, SPP, STEP-BACK, SimTom, MRP on various datasets)
Proposed Additions: Test the effect of adding +(meta reasoning) Structured Context Elaboration(single category/all) and + MUSR(multi-step soft reasoning) to the baseline methods.
Datasets: GSM8K, Gameof24, Trivia CW, HotpotQA, BigToM, Code, MMLU
RAG+QA Experiments

Methods:
naive, ours (GER, EEV) ... ?
Datasets:
NaturalQuestions
Multi-Doc-QA
LongBench
Page 18
6. Other Idea: Monte Carlo scorer optimised RLVR method

Motivation: Current reasoning models are (unnecessarily) verbose. Sometimes (e.g., when solving math) it likes to “trial and error" rather than doing sound reasoning.
Core Idea: Reward each step of sound reason rather than the whole reasoning chain. Reward succinct chain of thought. Score with a Monte Carlo Search like process.
Details: break one entire generation into a chain of actions (a1, a2, a3 ... an). For n times, continue generation with two adjacent subchains of action (one with a1, a2 and another with a1 a2 a3 for example) and use the difference of success rate to award/penalize the last action (e.g. a3).
Rationale: We can approximate the token by token autoregressive process by an Action by Action autoregressive process (and approximate action with sentence). If an action is effective, it should transform the problem into a simpler problem, leading to a higher continued generation success rate.
