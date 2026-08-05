import assert from "node:assert/strict";
import test from "node:test";

import {
  evaluatePullRequest,
  parseClosingReferences,
  parseIssueReferences,
  resolveCutoff,
  shouldSynchronize,
} from "./issue-gate.mjs";

const repository = "crewAIInc/crewAI";

function pullRequest(overrides = {}) {
  return {
    body: "Implements #123",
    created_at: "2026-08-12T00:00:00Z",
    labels: [],
    user: { login: "contributor" },
    ...overrides,
  };
}

function issue(overrides = {}) {
  return {
    number: 123,
    state: "open",
    labels: [{ name: "state:ready" }],
    ...overrides,
  };
}

async function evaluate(pullOverrides = {}, issueOverrides = {}) {
  return evaluatePullRequest({
    pullRequest: pullRequest(pullOverrides),
    repository,
    cutoff: "2026-08-11T00:00:00Z",
    getIssue: async () => issue(issueOverrides),
  });
}

test("parses shorthand and full repository references", () => {
  assert.deepEqual(
    parseIssueReferences(
      "Implements #123\nImplements crewAIInc/crewAI#456",
      repository,
    ),
    [
      { repository, number: 123 },
      { repository, number: 456 },
    ],
  );
});

test("parses prohibited GitHub closing references", () => {
  assert.deepEqual(
    parseClosingReferences(
      "Fixes #123\nCloses crewAIInc/crewAI#456\nResolves #789",
      repository,
    ),
    [
      { keyword: "Fixes", repository, number: 123 },
      { keyword: "Closes", repository, number: 456 },
      { keyword: "Resolves", repository, number: 789 },
    ],
  );
});

test("rejects closing keywords even with a valid Implements reference", async () => {
  const closingKeywords = [
    "Close",
    "Closes",
    "Closed",
    "Fix",
    "Fixes",
    "Fixed",
    "Resolve",
    "Resolves",
    "Resolved",
  ];
  for (const keyword of closingKeywords) {
    const result = await evaluate({
      body: `Implements #123\n${keyword} #123`,
    });

    assert.equal(result.ok, false);
    assert.match(result.reason, /prohibited closing reference/);
    assert.match(result.reason, new RegExp(keyword));
  }
});

test("deduplicates repeated references", () => {
  assert.deepEqual(
    parseIssueReferences("Implements #123\nimplements #123", repository),
    [{ repository, number: 123 }],
  );
});

test("passes an open ready issue", async () => {
  const result = await evaluate();

  assert.equal(result.ok, true);
  assert.equal(result.exempt, false);
  assert.equal(result.issueNumber, 123);
});

test("passes an open in-progress issue", async () => {
  const result = await evaluate({}, {
    labels: [{ name: "state:in-progress" }],
  });

  assert.equal(result.ok, true);
  assert.match(result.reason, /in progress/);
});

test("rejects a missing issue reference", async () => {
  const result = await evaluate({ body: "No issue yet" });

  assert.equal(result.ok, false);
  assert.match(result.reason, /does not contain/);
});

test("rejects more than one implementation issue", async () => {
  const result = await evaluate({
    body: "Implements #123\nImplements #456",
  });

  assert.equal(result.ok, false);
  assert.match(result.reason, /more than one/);
});

test("rejects an issue in another repository", async () => {
  const result = await evaluate({
    body: "Implements another/project#123",
  });

  assert.equal(result.ok, false);
  assert.match(result.reason, /must belong/);
});

test("rejects a pull request reference masquerading as an issue", async () => {
  const result = await evaluate({}, { pull_request: { url: "example" } });

  assert.equal(result.ok, false);
  assert.match(result.reason, /is not an issue/);
});

test("rejects a closed issue", async () => {
  const result = await evaluate({}, { state: "closed" });

  assert.equal(result.ok, false);
  assert.match(result.reason, /is not open/);
});

test("rejects an issue without the ready label", async () => {
  const result = await evaluate({}, { labels: [{ name: "state:design" }] });

  assert.equal(result.ok, false);
  assert.match(result.reason, /neither/);
});

test("exempts pull requests created before the cutoff", async () => {
  const result = await evaluate({
    body: null,
    created_at: "2026-08-10T23:59:59Z",
  });

  assert.equal(result.ok, true);
  assert.equal(result.exempt, true);
  assert.equal(result.exemption, "cutoff-legacy");
  assert.match(result.reason, /predates/);
});

test("distinguishes an explicit legacy exemption from the cutoff", async () => {
  const result = await evaluate({
    body: null,
    labels: [{ name: "policy:legacy" }],
  });

  assert.equal(result.ok, true);
  assert.equal(result.exempt, true);
  assert.equal(result.exemption, "policy-legacy");
  assert.match(result.reason, /policy:legacy/);
});

test("exempts pull requests with an override label", async () => {
  const result = await evaluate({
    body: null,
    labels: [{ name: "issue-gate:override" }],
  });

  assert.equal(result.ok, true);
  assert.equal(result.exempt, true);
  assert.match(result.reason, /override/);
});

test("exempts supported automation accounts", async () => {
  const result = await evaluate({
    body: null,
    user: { login: "dependabot[bot]" },
  });

  assert.equal(result.ok, true);
  assert.equal(result.exempt, true);
  assert.match(result.reason, /automation account/);
});

test("fails fast for an invalid cutoff", async () => {
  await assert.rejects(
    evaluatePullRequest({
      pullRequest: pullRequest(),
      repository,
      cutoff: "not-a-date",
      getIssue: async () => issue(),
    }),
    /not a valid date/,
  );
});

test("keeps unconfigured observe mode inert", () => {
  assert.equal(resolveCutoff("observe", ""), "9999-12-31T00:00:00Z");
});

test("requires an explicit cutoff for enforcement modes", () => {
  for (const mode of ["block", "close"]) {
    assert.throws(() => resolveCutoff(mode, ""), /ISSUE_GATE_CUTOFF is required/);
  }

  assert.equal(
    resolveCutoff("block", "2026-08-11T00:00:00Z"),
    "2026-08-11T00:00:00Z",
  );
});

test("synchronizes explicit legacy exemptions but not pre-cutoff PRs", () => {
  assert.equal(shouldSynchronize({ exemption: "policy-legacy" }), true);
  assert.equal(shouldSynchronize({ exemption: "cutoff-legacy" }), false);
});
