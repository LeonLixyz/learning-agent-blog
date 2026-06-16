# Language Models Can Think, But Not Learn

*Language models can reason over what is in the prompt, but they do not have a reliable way to turn new data or new experience into lasting memory. This essay is about that missing write path.*

In the movie Memento, the main character cannot form new memories. He has a conversation, solves a problem, learns a name, and minutes later it is all gone. To cope, he covers himself in tattoos and Polaroids and re-reads them every morning, just to work out who he is and what he is doing.

This is exactly where a language model is today. After pretraining and post-training its weights are frozen: the durable store of everything it knows stops changing. Whatever it works out during a session lives only in the context window, its working memory, and the moment the session ends that is wiped. Inside that window it can *think*, but it cannot *learn*: nothing it figures out from new data or new experience becomes part of the model. As Richard Sutton puts it, deployed models spend almost all of their compute not learning from the experience they are having ([Sutton, 2025](https://www.dwarkesh.com/p/richard-sutton)).

> **Figure: Context fills, then is wiped at session end**

# I · What is learning

## 1. What learning is

Learning is not the same as having information nearby. A database has information. A context window has information. A learner is different: it turns experience into compressed, reusable knowledge and skills. If you read a proof, practice a serve, or get sick from a food, the point is not that you can replay the episode or recite the fact, but that later, in a situation similar but not the same, you can use what you took from it. That is generalization, and it is what learning is for.

Why not just write the experience straight into a language model's weights with backpropagation? The catch is how knowledge forms. A model turns text into reliably stored, extractable knowledge only after it has seen a fact many times during training. Allen-Zhu and Li find that reaching a model's full storage capacity, roughly two bits per parameter, takes on the order of a thousand exposures per fact, and that training on ten times fewer exposures roughly halves what it retains ([Allen-Zhu & Li, 2024](https://arxiv.org/abs/2404.05405)). A corpus that states something once sits far below that, so the practical fix is to synthesize many diverse restatements and train on those ([Yang et al., 2024](https://arxiv.org/abs/2409.07431)). This is the **curse of learning**: turning a single observation into knowledge the model can use and generalize from takes on the order of a thousand varied repetitions, while your source gives you one.

# II · Long context is not the solution

## 2. Long context as working memory

People have tried plenty of ways around the curse, and the most obvious is to not write to the weights at all: just keep everything in the context window or an external database. Context, the model's working memory, has a famous name: **in-context learning**. The name is misleading. Given a few examples in the prompt, a model gets better at the task without any training. That is impressive, but it is mostly **inference**: the model thinks with the evidence in front of it. The "update" is a change in the prompt, the cache, or the temporary state of the run, not a change in the durable learner.

But why not just make the window enormous and pour everything in? Three reasons.

- **Data.** To use a 100M-token window the model must be trained on tasks that truly need 100M-token integration, and that data barely exists, so effective context lags far behind the advertised number ([Liu et al., 2023](https://arxiv.org/abs/2307.03172); [Hsieh et al., 2024](https://arxiv.org/abs/2404.06654)).
- **Compute.** Every query re-reads the whole history, and attention's cost grows with the square of the context length, so past some point the window is simply too expensive to run.
- **No abstraction.** A system that never compresses never builds skills; it re-reads the whole context and works each problem from scratch instead of getting better over time.

There's a whole industry making that working memory go further: building external memory systems, compacting the conversation, shrinking the KV cache by eviction or quantization. These genuinely help: they stretch how long a context can run, which matters for long-document inference and for long-horizon *agents* that would otherwise overflow. All of them optimize the same thing: the *size and reach* of the working memory. They improve the conditions for thinking. They do not create a durable write. However much the context is compressed, it is gone when the session ends. Bigger, cheaper RAM is still RAM. The same applies to long-context-style updates: if the update lives only in the run, it is a better thought, not a learned change.

## 3. Linear attention compresses and decays in working memory

The first architectural "fix" trades attention's exact-but-expensive log for a *fixed-size* summary. State-space models like Mamba keep a running state $h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$, learning to selectively remember and forget ([Gu & Dao, 2023](https://arxiv.org/abs/2312.00752)). Constant memory per token, runs forever, but the summary is lossy and must overwrite old information. Gu calls it "database vs. brain." It's a hard Pareto tradeoff: recall is bounded by state size ([Arora et al., 2024](https://arxiv.org/abs/2402.18668)).

Forgetting itself is not the problem; selective forgetting is what makes abstraction possible. Borges' Funes the Memorious remembers everything and is, as a result, incapable of thought: he cannot see why "dog" should cover so many different animals. Abstraction is forgetting on purpose, and the slow cortex earns its concepts by dropping particulars. The trouble is that today's architectures forget in **exactly the wrong place**. Mamba, linear attention, TTT, even Titans bake decay into the *working-memory* state, which confuses working memory with knowledge. Working memory should not forget at all; it should let the model look back at anything it has seen, exactly, and on that axis **attention is right**. Forgetting belongs in the slow consolidation into long-term knowledge, not in the working-memory buffer.

> **Figure: Where decay belongs: long-term store, not working memory**
>
> Forgetting-as-abstraction belongs in the slow long-term store. Compressed-state models misapply it to working memory, where attention's refusal to forget is exactly what you want.

## 4. Test-time training is just linear attention

The most interesting "fix": make the working-memory state itself a small model, and read by *training* it. Test-time training (TTT) takes a gradient step on that state for every token, $W_t = W_{t-1} - \eta\,\nabla_W\,\ell(W_{t-1};x_t)$ ([Sun et al., 2024](https://arxiv.org/abs/2407.04620)). It sounds like the model is finally learning as it reads. It is not.

The point is not the exact algebra. It is that test-time training is basically just another form of linear attention: in the linear case, TTT-Linear is literally **DeltaNet**, a delta-rule linear attention model. Calling it "training" makes it sound like real learning, but write the update out, name every piece, and the magic disappears.

> **First, the pieces.** Each token is projected into a *key* $k_t$ and a *value* $v_t$, exactly as attention makes keys and values: the key says what the token is about, the value is what to return for it. The memory is a matrix $S$ that maps a key to a predicted value, $\hat v = Sk$. To store the association "key $k_t$ returns value $v_t$," you add the *outer product* $v_tk_t^\top$ to $S$, because that is the matrix which, fed $k_t$, gives back $v_t$. And $\beta_t$ is just how hard you write this token, a per-step learning rate.
> **DeltaNet** does not write blindly. For the current key it first reads what the memory already returns, $\hat v_t=S_{t-1}k_t$, takes the error $v_t-\hat v_t$, and writes only that error back along the key:
> $$S_t=S_{t-1}+\beta_t\,(v_t-S_{t-1}k_t)\,k_t^\top.$$
> **TTT-Linear** calls the same matrix $W$ and treats it as a tiny linear model $f_W(k)=Wk$. It trains $f$ on the single target $v_t$ with squared loss $\ell_t(W)=\tfrac12\lVert Wk_t-v_t\rVert^2$. The gradient is $\nabla_W\ell_t=(Wk_t-v_t)k_t^\top$, so one gradient step is:
> $$W_t=W_{t-1}-\eta\,\nabla_W\ell_t=W_{t-1}+\eta\,(v_t-W_{t-1}k_t)\,k_t^\top.$$
> Rename $W$ to $S$ and the learning rate $\eta$ to $\beta_t$, and the two lines are identical: the same rank-one write of the same prediction error along the same key. The only difference is the story, recurrent context state versus fast weights.

> **Figure: The same rank-one update: context state vs fast weights**
>
> DeltaNet and TTT-Linear are the clean identity. DeltaNet stores the correction in a sequence-local recurrent state $S$; TTT-Linear stores the same correction in a temporary linear weight matrix $W$. The math is the same rank-one write, but the storage location is different, and both are normally wiped unless another system saves and consolidates the result.

DeltaNet and TTT-Linear are just two entries in a whole family of recurrent-update models that trade attention's growing cache for a fixed-size state ([Gu & Dao, 2023](https://arxiv.org/abs/2312.00752); [Dao & Gu, 2024](https://arxiv.org/abs/2405.21060); [Kimi Team, 2025](https://arxiv.org/abs/2510.26692); [Hatamizadeh et al., 2026](https://arxiv.org/abs/2605.22791)). They differ mostly in how they erase and write the state, as the table below lays out.

> **Figure: The recurrent-update family**
>
> Linear attention, DeltaNet, Gated DeltaNet, KDA, Gated DeltaNet-2, and TTT-Linear share a read-correct-write core. Mamba is a selective SSM cousin rather than a delta-rule model. TTT-MLP and TTT-E2E generalize the write into ordinary gradient descent on a richer inner model or selected model weights.

TTT actually runs gradient *training* at test time, and it still does not solve learning. Since TTT-Linear is exactly DeltaNet, every limit TTT hits, DeltaNet hits too:

- **The state usually resets.** It is re-initialized every sequence and never folded back into the base model, so the training is thrown away when the sequence ends.
- **Even a real gradient write is slow to make safe.** A fast write into shared weights overwrites neighboring knowledge (the catastrophic interference of §6), so a safe update has to be slow.
- **Fixed-state models give up attention's exact recall** to make long context cheaper. Richer variants like TTT-MLP and TTT-E2E push the write further ([Tandon et al., 2025](https://arxiv.org/abs/2512.23675)), but the update is still context-local unless something saves it.

Calling the update training does not make it learning. These are useful long-context mechanisms, not a fix for the missing write path.

> **Figure: Pass-key retrieval vs context length**
>
> Pass-key retrieval as the context grows. Full attention keeps near-perfect recall all the way to 128K. The fixed-state family, Mamba-2, Gated DeltaNet, and TTT, collapses toward zero, because a constant-size state cannot hold everything. That lost recall is the price of trading attention for a cheap fixed state ([Tandon et al., 2025](https://arxiv.org/abs/2512.23675)).

## 5. The agentic hack

The other "fix" is to optimize everything *around* a frozen model, escalating in cleverness: prompt engineering → retrieval (RAG) → context compaction → agentic scaffolds → and the recursive end, where the system optimizes its own harness and even evolves its own scaffold. The escalation *is* the point: it makes a better inferencer out of the same frozen learner. Even when an agent rewrites the agent that writes its own scaffold, it never touches the weights.

It rests on a quiet category error: it treats context as the model's **memory**, when context is really the model's **senses**. The tell is compaction: when the window fills, you summarize and discard, a lossy digest that nothing consolidates. This is the *Memento* problem from the opening of this essay. The notes can be brilliant, but the agent still wakes each session no smarter than before. Context optimization, long-context methods, RAG, and long-context-style updates will keep getting useful, but they fail as a theory of learning for the same reason: they sharpen inference while leaving memory unwritten. The harness is the tattoos, a remarkable way to cope with amnesia, not a cure.

# III · How brains do it, and what we have tried

## 6. Complementary learning systems: two stores and replay

Brains face our exact problem (learn fast without overwriting everything), and the answer is well developed: **Complementary Learning Systems** ([McClelland, McNaughton & O'Reilly, 1995](https://stanford.edu/~jlmcc/papers/McClellandMcNaughtonOReilly95.pdf); [Kumaran, Hassabis & McClelland, 2016](https://www.cell.com/trends/cognitive-sciences/fulltext/S1364-6613(16)30043-2)). Two systems with opposite settings: a **hippocampus** that records episodes fast, one-shot, kept separate; and a **neocortex** that integrates slowly over many exposures into structured knowledge.

Why two? If you write fast and hard into a dense, distributed memory, it bleeds into everything nearby, **catastrophic interference** (McCloskey & Cohen, 1989; French, 1999). The cortex avoids it by learning slowly and interleaved. The slowness is a feature. It's what protects old knowledge. (Same slowness as the thousand-exposures result. Now you know why.) But slow learning alone couldn't remember this morning, so the hippocampus grabs the episode now and, during rest and sleep, **replays** it to the cortex, interleaved, so it consolidates safely.

Mapped onto an LLM, the neocortex is the only part with an analog, and even it has stopped learning:

- The **weights are the neocortex**: general knowledge, slowly consolidated over many exposures during training. But a neocortex keeps consolidating for life, while the weights were frozen the day training ended and never updated since. We have the store, not the slow learner.
- **No hippocampus**: no fast, durable, one-shot store. The context window looks like one, but it is wiped when the session ends, so it is scratch paper, not memory.
- **No replay**: nothing that consolidates the day's experience into the weights.

The model has a frozen store and a whiteboard wiped between meetings. That is the gap. The good news: brains prove fast, durable, one-shot writing is possible and need *not* be a slow gradient grind. The hippocampus writes by association in one shot, much closer to a Hopfield network ([Ramsauer et al., 2020](https://arxiv.org/abs/2008.02217)) than to a thousand-step optimization.

> **Figure: The three memory stages: brain vs LLM**
>
> The brain solves "learn fast without overwriting everything" with three stages: a hippocampus that writes each episode in one shot, replay that consolidates it during sleep, and a neocortex that slowly builds durable knowledge. Mapped onto an LLM, only the neocortex survives, as the frozen weights. The fast write and the consolidation that would carry a session's experience into those weights are both missing, which is why the model wakes each session no smarter than before.

## 7. The primitive attempts

If we wanted a model that learns, it would need three capabilities. None is solved, but each has a serious 2025–26 attempt, and how each falls short tells you what's hard.

- **Write new knowledge.** Model-editing methods such as MEMIT ([Meng et al., 2022](https://arxiv.org/abs/2210.07229)) can patch a fact, and SEAL-style systems ([Zweiger et al., 2025](https://arxiv.org/abs/2506.10943)) make a model generate its own training data before updating. The hard part is propagation: the fact should affect related answers without causing drift or forgetting.
- **Internalize experience.** STaR-style bootstrapping ([Zelikman et al., 2022](https://arxiv.org/abs/2203.14465)) turns solved examples into training data, but it depends on a verifier and usually improves only inside a bounded task.
- **Keep long-term memory.** Titans-like memory modules ([Behrouz et al., 2025](https://arxiv.org/abs/2501.00663)) and external memory systems carry more, but most are still better working memory, not durable self-modification.

All of this needs a **signal**: what is worth keeping, and how good an attempt was. That, plus a store for experience and a write method, is the toolbox the next section specifies.

# IV · Towards real self-improving agents

## 8. The immediate hack: learning as a toolbox

We have endless data about the world, but almost none about *how to learn*. Today learning is a fixed recipe: pretraining, then SFT, then RL, the same pipeline for every model and task. People do not learn on a fixed recipe. You imitate, practice, ask a tutor, reflect, and above all *consolidate*, replaying and recasting an experience until it generalizes, switching between these by what the task allows. Learning should be a **tool the model picks up**, not a rule it is stuck with.

> **Figure: The learning agent's toolbox**
>
> Three tools, run as a loop. Signal: what to learn and how good an attempt was. Data: hold experience and synthesize the replay. Optimization: write it to the weights through an interchangeable method (SFT, RL, distillation, memory edit).

The toolbox holds three tools.

- The **signal** decides what to learn and scores an attempt: evals, verifiers, and environment returns externally; a learned value function or critic internally. External signal is sparse but reliable, internal signal dense but gameable.
- The **data tool** holds experience and synthesizes the training set, the consolidation step. The curse is paid here: one observation is expanded into the many representations retention requires, paraphrases, QA pairs, cloze deletions, contrastive negatives, counterexamples, tool-use traces, schema links, eval probes. EntiGraph (synthetic continued pretraining) and SEAL (self-edits) are instances ([Yang et al., 2024](https://arxiv.org/abs/2409.07431); [Zweiger et al., 2025](https://arxiv.org/abs/2506.10943)).
- The **optimization tool** writes to the weights through an interchangeable method, SFT, RL, on-policy distillation, self-distillation, or a localized weight edit, selected by the signal: dense reward to RL, a reference trajectory to SFT, a graded rollout to distillation. Making the write method a runtime choice rather than a fixed pipeline is most of what separates a trained model from a learning one.

## 9. From wrapped LLM to native learner

The range has two ends. At one, what we have now: a frozen LLM wrapped in agents and scaffolding. At the other, a native learner rebuilt from scratch, with all four primitives built in, a one-shot hippocampus, a consolidation loop that runs during "sleep," exploration, persistent memory. That is the principled end, and almost certainly where it lands, but brutally hard: redesign the stack, retrain from scratch, and beat a paradigm with a decade of optimization behind it. Every piece is an open problem.

> **Figure: Wrapped LLM, learning agent, native learner**
>
> The range, left to right: the LLM-plus-agents stack we have now, the learning agent (a working hack), and a native learner trained end to end. The middle is the prototype for the right end.

The **learning agent** sits in between, and it is what can be built today: the frozen transformer wrapped in a loop that generates data, fine-tunes, checks, and repeats, training on data it produces itself rather than data from a larger teacher (the shift SEAL points at). A hack, bolting memory on from outside, but a working one. Two objections bite here. The Bitter Lesson says just scale, but continual learning is the half of it we have not scaled. And periodic retraining suffices only until the model must adapt to one user, now, from a handful of examples. The whole range gets explored, and the learning agent in the middle is the prototype for the right end.

## 10. How to train the learning agent

One question is left: who tunes the agent itself? You should not hand-design the recipe, which data to generate, which method to use, what to keep, any more than you hand-design the knowledge. Let the agent search for it, the way evolutionary methods search: propose a variation, evaluate it, keep what works, archive it, repeat. AlphaEvolve, FunSearch, the Darwin Gödel Machine, and Hyperagents show this propose-evaluate-archive loop genuinely discovers: AlphaEvolve even found a way to multiply 4×4 complex matrices in 48 scalar multiplications, beating Strassen's 49 ([Novikov et al., 2025](https://arxiv.org/abs/2506.13131)). Aimed at the learning process itself, that loop lets the agent learn how to learn: it evolves better recipes for turning experience into durable skill, instead of running one fixed recipe forever.

> **Figure: The evolutionary search loop**
>
> Evolutionary search: propose a variation, evaluate it, select the winners, archive them, repeat (AlphaEvolve, FunSearch, the Darwin Gödel Machine, Hyperagents). The weights are never updated; gains accumulate in the archive, so the loop drives discovery but is not itself the write path.

---

## References

1. Geva et al. (2021), *Transformer Feed-Forward Layers Are Key-Value Memories.* arXiv:2012.14913
2. Lample et al. (2019), *Large Memory Layers with Product Keys.* arXiv:1907.05242
3. Berges et al. (2024), *Memory Layers at Scale.* arXiv:2412.09764
4. Allen-Zhu & Li (2024), *Physics of Language Models 3.3: Knowledge Capacity Scaling Laws.* arXiv:2404.05405
5. Morris et al. (2025), *How Much Do Language Models Memorize?* arXiv:2505.24832
6. von Oswald et al. (2023), *Transformers Learn In-Context by Gradient Descent.* arXiv:2212.07677
7. Shen et al. (2023), *Do Pretrained Transformers Really Learn In-Context by Gradient Descent?* arXiv:2310.08540
8. Hendel et al. (2023), *In-Context Learning Creates Task Vectors.* arXiv:2310.15916
9. Liu et al. (2023), *Lost in the Middle.* arXiv:2307.03172
10. Hsieh et al. (2024), *RULER.* arXiv:2404.06654
11. Eyuboglu et al. (2025), *Cartridges.* arXiv:2506.06266
12. Gu & Dao (2023), *Mamba.* arXiv:2312.00752
13. Dao & Gu (2024), *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality.* arXiv:2405.21060
14. Arora et al. (2024), *Based.* arXiv:2402.18668
15. Sun et al. (2024), *Learning to (Learn at Test Time).* arXiv:2407.04620
16. Yang, Kautz & Hatamizadeh (2024/2025), *Gated Delta Networks: Improving Mamba2 with Delta Rule.* arXiv:2412.06464; Kimi Team (2025), *Kimi Linear.* arXiv:2510.26692; Hatamizadeh, Choi & Kautz (2026), *Gated DeltaNet-2.* arXiv:2605.22791.
17. Tandon et al. (2025), *End-to-End Test-Time Training for Long Context.* arXiv:2512.23675
18. Ba, Hinton et al. (2016), *Using Fast Weights to Attend to the Recent Past.* arXiv:1610.06258; Hinton & Plaut (1987); Schmidhuber (1992).
19. Schlag et al. (2021), *Linear Transformers Are Secretly Fast Weight Programmers.* arXiv:2102.11174
20. Wang et al. (2025), *Test-Time Regression.* arXiv:2501.12352
21. Bailey, Kandel & Harris (2015), *Structural Components of Synaptic Plasticity and Memory Consolidation.* Cold Spring Harbor Perspectives in Biology; Kandel (2009), *The Biology of Memory: A Forty-Year Perspective.* Journal of Neuroscience.
22. Schultz, Dayan & Montague (1997), *A Neural Substrate of Prediction and Reward.* Science.
23. McCloskey & Cohen (1989), *Catastrophic Interference in Connectionist Networks*; McClelland, McNaughton & O'Reilly (1995); Kumaran, Hassabis & McClelland (2016), Complementary Learning Systems.
24. Tse et al. (2007), *Schemas and Memory Consolidation.* Science.
25. Ramsauer et al. (2020), *Hopfield Networks Is All You Need.* arXiv:2008.02217
26. Zweiger et al. (2025), *SEAL.* arXiv:2506.10943; Meng et al. (2022), *MEMIT.* arXiv:2210.07229; Hase et al. (2023), arXiv:2301.04213.
27. Zelikman et al. (2022), *STaR.* arXiv:2203.14465; Yue et al. (2025), arXiv:2504.13837.
28. Burda et al. (2018), *RND.* arXiv:1810.12894; Kirk et al. (2023), arXiv:2310.06452.
29. Madaan et al. (2023), *Self-Refine: Iterative Refinement with Self-Feedback.* arXiv:2303.17651
30. Behrouz et al. (2025), *Titans.* arXiv:2501.00663; *Nested Learning.* arXiv:2512.24695.
31. Novikov et al. (2025), *AlphaEvolve.* arXiv:2506.13131; Romera-Paredes et al. (2024), *FunSearch* (Nature); Zhang et al. (2025), *Darwin Gödel Machine.* arXiv:2505.22954.
32. Yang et al. (2024), *Synthetic Continued Pretraining (EntiGraph).* arXiv:2409.07431.
33. Luo et al. (2023), arXiv:2308.08747; Villalobos et al. (2024), arXiv:2211.04325.
34. Sutton (2019), *The Bitter Lesson*; Sutton (2025), Dwarkesh Podcast; Silver & Sutton (2025), *The Era of Experience*; Borges (1942), *Funes the Memorious.*
35. Mitchell (1997), *Machine Learning.* McGraw-Hill (the experience / task / performance definition, p. 2).
36. Hebb (1949), *The Organization of Behavior.* Wiley (the neurophysiological postulate, p. 62).
37. Hassabis (2026), interview with Alex Kantrowitz, *Big Technology.*
