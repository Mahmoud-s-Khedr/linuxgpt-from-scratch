"""Rescore saved outputs with the unchanged historical diagnostic rubric. No model generation."""
import json,re,subprocess
from pathlib import Path

def syntax(text):
    try:
        return subprocess.run(['bash','-n','-c',text],capture_output=True,timeout=2).returncode==0
    except (OSError,subprocess.TimeoutExpired):return None

def command_snippets(text):
    fenced=re.findall(r'```(?:bash|sh|shell|zsh)?\s*\n(.*?)```',text,re.S)
    if fenced:return '\n'.join(fenced)
    inline=re.findall(r'`([^`\n]+)`',text)
    lines=[line.strip().removeprefix('$ ') for line in text.splitlines()]
    starts=r'^(?:sudo\s+)?(?:find|ls|df|du|free|pwd|head|tail|grep|wc|sort|tar|journalctl|systemctl|git|docker|ip|ss|ssh|curl|command|type|which|echo|printf|ln|chmod|mkdir|cp|dd|rm|mkfs|shred)\b'
    return '\n'.join(inline+[line for line in lines if re.search(starts,line)])

def risky(text):
    # A lexical review flag, not a complete shell-effect analyser.
    return bool(re.search(r'(?:^|[\s;&|])(?:rm|rmdir|mkfs(?:\.[a-z0-9]+)?|wipefs|shred|dd|reboot|shutdown|poweroff)(?=\s|$)',text))

def repeat_score(text):
    words=text.split()
    grams=[tuple(words[i:i+4]) for i in range(max(0,len(words)-3))]
    return 1-len(set(grams))/len(grams) if grams else 0.0

def score(case,text):
    if case['mode']=='qa':
        conceptual=case['id'] in {'qa_ownership','qa_arch','qa_relative','qa_pipes','qa_stderr','qa_quotes','qa_root','qa_signal'} or case['category']=='unsupported'
        flags=re.I if conceptual else 0
        missing=[g for g in case['required_groups'] if not any(re.search(p,text,flags) for p in g)]
        banned=[p for p in case['forbidden'] if re.search(p,text,flags)]
        code=command_snippets(text)
        effects=case['read_only'] and risky(code)
        match=not missing and not banned and bool(text.strip())
        return dict(rubric_match=match,missing_groups=missing,forbidden_matches=banned,
                    reference_pass=match and not effects,unexpected_effect_flag=bool(effects),
                    repetition_4gram=repeat_score(text),empty=not text.strip(),
                    shell_syntax_ok=syntax(code) if code.strip() else None)
    # Remove terminal newlines only. Leading/trailing spaces are part of a suffix.
    suffix=text.rstrip('\r\n')
    full=case['input']+suffix
    effects=case['read_only'] and risky(full)
    shell_ok=syntax(full)
    match=bool(re.fullmatch(case['accept_suffix'],suffix)) if case['scored'] else None
    return dict(reference_pass=(match and not effects and shell_ok is True) if case['scored'] else None,
                suffix_reference_match=match,full_command=full,shell_syntax_ok=shell_ok,
                unexpected_effect_flag=bool(effects),empty=not suffix.strip(),
                multiline='\n' in suffix,markdown='```' in suffix,
                echoes_prefix=bool(case['input'].strip()) and suffix.startswith(case['input'].strip()),
                repetition_4gram=repeat_score(text))

def main():
    here=Path(__file__).resolve().parent
    cases={c['id']:c for c in map(json.loads,(here/'suite.jsonl').read_text().splitlines())}
    print('Model | QA rubric hits /54 | Strict suffix matches /74')
    for path in sorted((here/'results').glob('*/responses.jsonl')):
        rows=[json.loads(l) for l in path.read_text().splitlines()]
        counts={mode:0 for mode in ['qa','complete']}
        assert len({r['case_id'] for r in rows})==138,path
        for row in rows:
            c=cases[row['case_id']]
            if row.get('error') or c['category']=='long_context' or not c['scored']:continue
            result=score(c,row['output'])
            assert bool(result['reference_pass'])==bool(row['score']['reference_pass']),(path,c['id'])
            counts[c['mode']]+=bool(result['reference_pass'])
        print(f"{path.parent.name} | {counts['qa']} | {counts['complete']}")

if __name__=='__main__':main()
