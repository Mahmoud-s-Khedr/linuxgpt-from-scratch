# Data format

Training corpora are not included. Tokenizer/pretraining preparation accepts cleaned UTF-8 `*.txt` files, with a line containing `---` between documents. At least two documents per source are needed to leave any source documents for training.

SFT extraction selects recognized filenames and formats rather than treating arbitrary prose as instruction data:

- `05_tldr_pages.txt`: Markdown command pages with `# command`, description lines beginning `> `, and task/command pairs.
- `06_arch_wiki.txt`: titled conceptual documents.
- `11_unix_stackexchange.txt`, `12_askubuntu.txt`, `13_serverfault.txt`, `15_superuser_linux.txt`: question/answer documents, including accepted-answer markers.
- `16_shell_scripts_and_configs.txt`, `18_cli_cheatsheets.txt`: shell examples/code blocks.

A TLDR-style document looks like this (the outer fence is for display):

````markdown
# cat
> Display file contents.

- Display a report:

`cat report.txt`

---
````

The task and backtick command must be separated by a blank line. See `build_sft_data.py` for the exact parsers and source caps. The examples above illustrate formatting, not a sufficient training corpus. The current splitter reserves at least 200 unique commands and 200 Q&A prompts for validation; inspect the generated metadata to ensure both splits contain usable examples.

SFT rows contain `mode`, `prompt` and `completion`. Q&A uses `<|qa|><|user|>...<|assistant|>`; autocomplete uses `<|complete|><|ctx|>...<|cmd|>...<|pred|>`. Targets end with `<|end|>`. Preserve all command-prefix and completion whitespace.

Use sources you can use for your project and keep their provenance. This publication distributes the training pipeline and a synthetic evaluation suite, not the original documentation/forum corpus.
