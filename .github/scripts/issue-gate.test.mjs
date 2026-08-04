import assert from "node:assert/strict";
import test from "node:test";

import {
  evaluatePullRequest,
  parseIssueReferences,
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

test("ignores GitHub closing keywords and template placeholders", () => {
  assert.deepEqual(
    parseIssueReferences(
      "Fixes #123\nCloses #456\nImplements #<issue-number>",
      repository,
    ),
    [],
  );
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
  assert.match(result.reason, /does not have/);
});

test("exempts pull requests created before the cutoff", async () => {
  const result = await evaluate({
    body: null,
    created_at: "2026-08-10T23:59:59Z",
  });

  assert.equal(result.ok, true);
  assert.equal(result.exempt, true);
  assert.match(result.reason, /predates/);
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
