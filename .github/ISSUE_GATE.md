# Issue Gate Pilot

The issue gate checks whether a pull request references exactly one open issue
with the `state:ready` label. It runs on `pull_request_target`, checks out only
the trusted default branch, and never fetches or executes pull request code.

## Repository setup

Create these labels before activating the pilot:

- Issue lifecycle: `state:inbox`, `state:design`, `state:ready`,
  `state:in-progress`, and `state:verification`
- Gate results: `issue-gate:passed`, `issue-gate:exempt`, and
  `needs-ready-issue`
- Maintainer exceptions: `issue-gate:override` and `policy:legacy`

Configure these repository variables:

| Variable | Required | Value |
| --- | --- | --- |
| `ISSUE_GATE_MODE` | No | `observe` (default), `block`, or `close` |
| `ISSUE_GATE_CUTOFF` | Yes to activate | ISO 8601 timestamp for the first PR covered by the pilot |
| `ISSUE_GATE_READY_LABEL` | No | Defaults to `state:ready` |

Without `ISSUE_GATE_CUTOFF`, the workflow uses a future cutoff and treats every
pull request as legacy. This makes the workflow inert until maintainers choose
the activation time.

## Modes

- `observe`: Invalid pull requests receive a successful `Issue gate` status,
  an explanatory comment, and the `needs-ready-issue` label.
- `block`: Invalid pull requests receive a failing status. Add `Issue gate` to
  the `main` ruleset's required checks only after observation is complete.
- `close`: Invalid pull requests receive a failing status and are closed after
  the comment is posted.

The script refuses to enter `block` or `close` mode without a configured cutoff.
Pull requests created by supported dependency automation, or labeled
`issue-gate:override` or `policy:legacy`, remain exempt.

## Testing the pilot

Run the `Issue Gate` workflow manually with a pull request number, or edit a
pull request description to trigger it again. Test at least these cases before
changing modes:

1. No implementation issue
2. A nonexistent, closed, or non-ready issue
3. More than one implementation issue
4. One open issue labeled `state:ready`
5. A pull request before the cutoff
6. A pull request with `issue-gate:override`

Keep the gate in `observe` mode until the sample contains at least 20 new pull
requests with no false positives.
