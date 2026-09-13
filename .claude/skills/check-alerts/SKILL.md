---
description: Preview, test, or tune the Slack alerts without sending anything. Use when the user asks what alerts would fire, wants to test a rule, thinks alerts are too noisy or too quiet, or wants to add an alert.
disable-model-invocation: true
allowed-tools: Bash(python run_all.py --notify --dry-run) Bash(python run_all.py --digest --dry-run) Bash(python alerts.py)
---

## Current alert state

!`cat data/alert_state.json 2>/dev/null || echo "no state file yet — every alert would be treated as first-time"`

## Rules

Always use `--dry-run`. Never send a real message to test something; the
webhook posts to a channel the user actually reads.

```bash
python run_all.py --notify --dry-run     # what would fire right now
python run_all.py --digest --dry-run     # preview the biweekly digest
python alerts.py                         # dry run against a synthetic transition
```

## When adding or changing an alert

Every rule fires on the **transition into** a state, never on the state
persisting. Before adding one, answer: if this condition stays true for a week,
how many messages does the user get? If the answer is more than one, the rule
is wrong.

Check the new rule against all three:

1. **Transition, not state.** Compare `cur` to `prev` and fire only on change.
2. **Cooldown.** Set `cooldown_days` to the shortest interval at which a repeat
   is genuinely new information.
3. **Redundancy.** If a stronger alert already covers the situation, suppress
   the weaker one — see how `btc_armed` is suppressed when `btc_entry` fires.

Then verify by running the same payload twice: the second run must send zero.
An alert that re-fires on unchanged input is a bug, not a preference.

## When tuning noise

Too noisy: lengthen cooldowns first, tighten thresholds second, and only remove
a rule last. Too quiet is usually correct behaviour — most days genuinely have
nothing to say, and silence is the design. Check `data/alert_state.json` before
concluding a rule is broken; it may simply be inside its cooldown.

## Slack limits worth remembering

50 blocks per message, 3,000 characters per text object. The digest currently
sits at 13 blocks and about 2KB, so there is room, but a per-stock block would
blow the block cap at 50 rows.
