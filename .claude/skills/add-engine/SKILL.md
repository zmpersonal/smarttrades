---
description: Add a new screening engine or tab to the dashboard, following the existing gate-then-score structure. Use when the user wants a new screener, a new tab, or a new signal type added.
argument-hint: [engine-name]
---

Add the `$ARGUMENTS` engine. Ask what it screens for before writing anything —
the gates come from the thesis, not the other way round.

## Design it in this order

1. **What disqualifies a name?** Write the gates first. Hard pass/fail, each
   with a stated reason returned on failure. If you cannot name what makes a
   candidate ineligible, the engine is not ready to build.
2. **What ranks the survivors?** Weighted components, each scored 0–100 and
   returned individually. The UI shows why a name ranks; a bare score is not
   acceptable output.
3. **What is the honest failure mode?** Every existing engine names its own —
   dark pool false-positives on heavily shorted names, dividends on yield
   traps, recovery on multiple-only theses. Find this engine's and either gate
   or penalise it.
4. **How stale is the data?** State the real latency. This is what reshaped the
   dark pool and politician engines, and it changes what the engine can claim.

## Implementation checklist

- `engines/<name>.py` with `<name>_gates() -> list[str]` and
  `score_<name>() -> dict` including a `components` key.
- Register in `run_all.py`: a runner, a `CADENCE` entry, and any new loader as
  an explicit `NotImplementedError` stub.
- Add to `ENGINES` in `index.html` with a distinct accent hex, column
  definitions, and a `comps` list matching the component keys.
- Add to `ORDER` and the tape `why` map. Check the tape grid column count still
  matches the number of engines.
- Include at least one deliberate gate-failure row in the sample data, with its
  failures listed. Every existing tab has one so the filter logic is visible
  rather than implied.
- If it should alert, add a transition rule in `alerts.py` — see the
  `check-alerts` skill for the constraints.

## Verify before reporting done

```bash
node -e "const fs=require('fs');const s=fs.readFileSync('index.html','utf8').match(/<script>([\s\S]*?)<\/script>/)[1];new Function(s);console.log('JS OK')"
python -c "import ast;ast.parse(open('engines/<name>.py').read());print('OK')"
python run_all.py --only <name> --force
```

The JS check matters: `index.html` is edited by string replacement and a
missing comma between row objects has broken it before.
