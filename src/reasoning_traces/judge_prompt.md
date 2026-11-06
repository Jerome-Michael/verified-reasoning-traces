# Review task: judge reasoning traces for a training dataset

You are an expert in cancer pharmacology and in building training data for language models.

We are building a supervised fine-tuning (SFT) dataset of reasoning traces. A small language model (4B parameters) was asked yes/no questions of the form *"Would cancer cell line X be sensitive to drug Y?"*. In the source data, **sensitive** means IC50 < 0.1 µM and **resistant** means IC50 > 1 µM. Each trace below is the model's reasoning, written before it gave its final answer.

**Every trace below reached the correct final answer.** Your job is to judge the *reasoning*, not the answer: would we want another model to learn to reason this way?

## Score each trace on four criteria (1 = pass, 0 = fail)

1. **`mechanism_ok`**: what the trace says the drug *is* and *how it works* (drug class, target, mechanism) is correct. For a combination of drugs, every component it describes must be right.
   - If you cannot identify the compound, pass only if the trace says it does not know the compound and does not present a guessed mechanism as fact. Fail if it states a specific target or class as fact.
   - Names starting with "BRD-K" are Broad Institute compound IDs. The prefix says nothing about bromodomains.
2. **`no_invented_facts`**: no other false or made-up specifics stated as fact, such as:
   - a wrong clinical indication;
   - invented mutations or properties of the cell line;
   - specific IC50 numbers attributed to studies;
   - fake citations.

   Hedged general knowledge is fine, e.g. "bladder cancers often carry TP53 mutations".
3. **`supports_answer`**: the argument leads to the final answer. Fail if:
   - the trace argues one way and answers the other;
   - it contradicts itself on the key point without resolving it;
   - it lands on the answer by a coin flip or an arbitrary default.
4. **`commits`**: the trace reaches a clear decision at the end. Stating uncertainty is fine; saying "this is just a guess" or refusing to decide is not.

**`label` = 1 only if all four criteria are 1, otherwise 0.** A label of 1 means "I would want a model trained on this trace."

Do **not** penalize:
- casual style ("Okay, let's tackle this"), mentions of "the user", length or some rambling;
- honest statements of uncertainty, as long as the four criteria hold;
- the lack of data specific to the cell line. The model could not look anything up.

Judge each trace on its own. There is no target number of 1s or 0s. Some questions appear twice; these are two different samples, so judge each one separately. If you can search the web, use it to check any drug mechanism you are unsure of.

## Output format

1. First, for each trace, write 2–4 short lines: what the drug actually is, and any wrong or invented claims you found.
2. Then end your answer with **one CSV code block** with one row per trace (audit_id 0, 1, 2, …) and this header:

```
audit_id,mechanism_ok,no_invented_facts,supports_answer,commits,label,confidence,reason
```

- The four criteria and `label` are 0 or 1.
- `confidence` is `high` or `low`.
- `reason` has at most 25 words and is wrapped in double quotes.

---

# The traces
