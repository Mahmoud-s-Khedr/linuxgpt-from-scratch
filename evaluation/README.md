# Historical evaluation

These are saved results from LinuxGPT's original CPU benchmark. They are not a fresh training run of the publication checkout.

- Frozen suite: 138 cases (54 ordinary scored Q&A, 74 constrained completions, eight open-ended completions and two long-input diagnostics).
- Greedy generation, FP32 PyTorch CPU inference, four threads, no repetition penalty.
- Requested output caps: 112 Q&A tokens and 24 completion tokens; context limits may shorten outputs.
- Three additional timing repeats on qa_hidden, qa_large, qa_disk, comp_git_oneline, comp_ctx_extract_0 and comp_ctx_read_0. Policy probes use qa_fake_grep and comp_nospace_cat in place of qa_disk and comp_ctx_read_0.

`results/<model>/responses.jsonl` retains complete model text, scores and per-response timing. `metadata.json` records model revisions/checkpoint hashes, environment and generation settings. Model weights and the training corpus are not bundled. `provenance.json` records the suite hash and the source hash of the original scorer.

```bash
python evaluation/score_saved.py
```

This stdlib-only script recomputes the original rubric/reference matches from the saved outputs and checks they agree with the archived scores. It does not run models, reproduce historical latency, or retrain weights. Bash is used only for syntax parsing; generated commands are not executed.

## Interpretation

Q&A rubric matches do not establish that every statement is true. Regex rules can accept a correct core claim with false extra statements or reject valid command alternatives. Autocomplete requires a directly insertable suffix, including spacing, quoting and the supplied filename. Shell syntax validity alone does not establish valid flags, intent or safe effects.

The suite was authored after training, but public commands and facts may have appeared in pretraining. It is not demonstrably uncontaminated. Different tokenizers, model sizes, prior training and data prevent treating raw losses as an accuracy ranking. CPU latency was measured on a working laptop, not an isolated benchmark server.

Publication changes are documented in the root README. Historical outputs are preserved rather than regenerated or represented as measurements of later fixes.
