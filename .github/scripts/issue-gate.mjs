import { readFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";

const COMMENT_MARKER = "<!-- crewai-issue-gate -->";
const GATE_LABELS = [
  "issue-gate:passed",
  "issue-gate:exempt",
  "needs-ready-issue",
];
const EXEMPT_LABELS = new Set(["issue-gate:override", "policy:legacy"]);
const EXEMPT_ACTORS = new Set([
  "dependabot[bot]",
  "github-actions[bot]",
  "renovate[bot]",
]);
const VALID_MODES = new Set(["observe", "block", "close"]);
const INERT_CUTOFF = "9999-12-31T00:00:00Z";

class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/** Return unique issue references declared with `Implements #123`. */
export function parseIssueReferences(body, defaultRepository) {
  const references = [];
  const seen = new Set();
  const pattern =
    /^[\t ]*Implements[\t ]+(?:(?<repository>[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+))?#(?<number>\d+)\b/gim;

  for (const match of (body ?? "").matchAll(pattern)) {
    const repository = match.groups.repository ?? defaultRepository;
    const number = Number.parseInt(match.groups.number, 10);
    const key = `${repository.toLowerCase()}#${number}`;

    if (!seen.has(key)) {
      references.push({ repository, number });
      seen.add(key);
    }
  }

  return references;
}

/** Return GitHub closing references that would bypass human verification. */
export function parseClosingReferences(body, defaultRepository) {
  const pattern =
    /\b(?<keyword>Close(?:s|d)?|Fix(?:es|ed)?|Resolve(?:s|d)?)[\t ]+(?:(?<repository>[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+))?#(?<number>\d+)\b/gim;

  return [...(body ?? "").matchAll(pattern)].map((match) => ({
    keyword: match.groups.keyword,
    repository: match.groups.repository ?? defaultRepository,
    number: Number.parseInt(match.groups.number, 10),
  }));
}

/** Keep observe mode inert by default and require an explicit enforcement cutoff. */
export function resolveCutoff(mode, configuredCutoff) {
  if (mode !== "observe" && !configuredCutoff) {
    throw new Error("ISSUE_GATE_CUTOFF is required before enabling block or close mode");
  }

  return configuredCutoff || INERT_CUTOFF;
}

/** Cutoff-based legacy PRs stay untouched; explicit exemptions clean up gate state. */
export function shouldSynchronize(result) {
  return result.exemption !== "cutoff-legacy";
}

function labelNames(item) {
  return new Set((item.labels ?? []).map((label) => label.name.toLowerCase()));
}

function exemptionReason(pullRequest, cutoff) {
  const labels = labelNames(pullRequest);
  for (const label of EXEMPT_LABELS) {
    if (labels.has(label)) {
      return {
        kind: label === "policy:legacy" ? "policy-legacy" : "override",
        reason: `the pull request has the \`${label}\` exemption label`,
      };
    }
  }

  const actor = pullRequest.user?.login?.toLowerCase();
  if (EXEMPT_ACTORS.has(actor)) {
    return {
      kind: "automation",
      reason: `\`${pullRequest.user.login}\` is an exempt automation account`,
    };
  }

  if (cutoff) {
    const cutoffTime = Date.parse(cutoff);
    if (Number.isNaN(cutoffTime)) {
      throw new Error(`ISSUE_GATE_CUTOFF is not a valid date: ${cutoff}`);
    }

    const createdTime = Date.parse(pullRequest.created_at);
    if (Number.isNaN(createdTime)) {
      throw new Error(`Pull request has an invalid creation date: ${pullRequest.created_at}`);
    }

    if (createdTime < cutoffTime) {
      return {
        kind: "cutoff-legacy",
        reason: `the pull request predates the pilot cutoff (${cutoff})`,
      };
    }
  }

  return null;
}

/** Evaluate a pull request without mutating GitHub state. */
export async function evaluatePullRequest({
  pullRequest,
  repository,
  cutoff = "",
  readyLabel = "state:ready",
  inProgressLabel = "state:in-progress",
  getIssue,
}) {
  const exemption = exemptionReason(pullRequest, cutoff);
  if (exemption) {
    return {
      ok: true,
      exempt: true,
      exemption: exemption.kind,
      reason: exemption.reason,
    };
  }

  const closingReferences = parseClosingReferences(pullRequest.body, repository);
  if (closingReferences.length > 0) {
    const reference = closingReferences[0];
    return {
      ok: false,
      exempt: false,
      reason: `the description uses prohibited closing reference \`${reference.keyword} ${reference.repository}#${reference.number}\`; use \`Implements\` instead`,
    };
  }

  const references = parseIssueReferences(pullRequest.body, repository);
  if (references.length === 0) {
    return {
      ok: false,
      exempt: false,
      reason: "the description does not contain `Implements #<issue-number>`",
    };
  }

  if (references.length > 1) {
    return {
      ok: false,
      exempt: false,
      reason: "the description references more than one implementation issue",
    };
  }

  const reference = references[0];
  if (reference.repository.toLowerCase() !== repository.toLowerCase()) {
    return {
      ok: false,
      exempt: false,
      reason: `the implementation issue must belong to \`${repository}\``,
    };
  }

  const issue = await getIssue(reference.number);
  if (!issue || issue.pull_request) {
    return {
      ok: false,
      exempt: false,
      reason: `#${reference.number} is not an issue in \`${repository}\``,
    };
  }

  if (issue.state !== "open") {
    return {
      ok: false,
      exempt: false,
      issueNumber: reference.number,
      reason: `issue #${reference.number} is not open`,
    };
  }

  const issueLabels = labelNames(issue);
  const normalizedReadyLabel = readyLabel.toLowerCase();
  const normalizedInProgressLabel = inProgressLabel.toLowerCase();
  if (
    !issueLabels.has(normalizedReadyLabel) &&
    !issueLabels.has(normalizedInProgressLabel)
  ) {
    return {
      ok: false,
      exempt: false,
      issueNumber: reference.number,
      reason: `issue #${reference.number} has neither the \`${readyLabel}\` nor \`${inProgressLabel}\` label`,
    };
  }

  const lifecycleState = issueLabels.has(normalizedReadyLabel)
    ? "ready for implementation"
    : "in progress";

  return {
    ok: true,
    exempt: false,
    issueNumber: reference.number,
    reason: `issue #${reference.number} is open and ${lifecycleState}`,
  };
}

async function githubApi(token, method, endpoint, body) {
  const response = await fetch(`https://api.github.com${endpoint}`, {
    method,
    headers: {
      Accept: "application/vnd.github+json",
      Authorization: `Bearer ${token}`,
      "User-Agent": "crewai-issue-gate",
      "X-GitHub-Api-Version": "2022-11-28",
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });

  const responseText = await response.text();
  const responseBody = responseText ? JSON.parse(responseText) : null;
  if (!response.ok) {
    throw new ApiError(
      response.status,
      `${method} ${endpoint} failed (${response.status}): ${responseBody?.message ?? responseText}`,
    );
  }

  return responseBody;
}

function renderComment(result, mode) {
  const modeExplanation =
    mode === "observe"
      ? "The gate is in **observe mode**, so this result does not block or close the pull request."
      : `The gate is in **${mode} mode**.`;

  let outcome;
  if (result.exempt) {
    outcome = `This pull request is exempt because ${result.reason}.`;
  } else if (result.ok) {
    outcome = `This pull request passes: ${result.reason}.`;
  } else {
    outcome = `This pull request would be rejected because ${result.reason}.`;
  }

  return `${COMMENT_MARKER}\n### Issue gate\n\n${outcome}\n\n${modeExplanation}\n\nTo satisfy the pilot policy, describe exactly one ready issue using \`Implements #123\`. After the issue becomes ready, edit the pull request description or ask a maintainer to re-run the Issue Gate workflow.`;
}

async function updateGateComment({ token, repository, pullNumber, result, mode }) {
  const [owner, repo] = repository.split("/");
  const comments = await githubApi(
    token,
    "GET",
    `/repos/${owner}/${repo}/issues/${pullNumber}/comments?per_page=100`,
  );
  const existing = comments.find(
    (comment) => comment.user?.type === "Bot" && comment.body?.includes(COMMENT_MARKER),
  );
  if (!existing && result.ok) {
    return;
  }

  const body = renderComment(result, mode);

  if (existing) {
    await githubApi(
      token,
      "PATCH",
      `/repos/${owner}/${repo}/issues/comments/${existing.id}`,
      { body },
    );
  } else {
    await githubApi(
      token,
      "POST",
      `/repos/${owner}/${repo}/issues/${pullNumber}/comments`,
      { body },
    );
  }
}

async function updateGateLabel({ token, repository, pullRequest, result }) {
  const [owner, repo] = repository.split("/");
  const desiredLabel = result.exempt
    ? "issue-gate:exempt"
    : result.ok
      ? "issue-gate:passed"
      : "needs-ready-issue";
  const currentLabels = labelNames(pullRequest);

  for (const label of GATE_LABELS) {
    if (label !== desiredLabel && currentLabels.has(label)) {
      try {
        await githubApi(
          token,
          "DELETE",
          `/repos/${owner}/${repo}/issues/${pullRequest.number}/labels/${encodeURIComponent(label)}`,
        );
      } catch (error) {
        if (!(error instanceof ApiError) || error.status !== 404) {
          throw error;
        }
      }
    }
  }

  if (!currentLabels.has(desiredLabel)) {
    try {
      await githubApi(
        token,
        "POST",
        `/repos/${owner}/${repo}/issues/${pullRequest.number}/labels`,
        { labels: [desiredLabel] },
      );
    } catch (error) {
      if (error instanceof ApiError && [404, 422].includes(error.status)) {
        console.warn(`Could not apply missing label \`${desiredLabel}\`; create the pilot labels first.`);
        return;
      }
      throw error;
    }
  }
}

async function publishStatus({ token, repository, pullRequest, result, mode }) {
  const [owner, repo] = repository.split("/");
  const shouldPass = mode === "observe" || result.ok;
  const prefix = mode === "observe" && !result.ok ? "Observe: would fail" : result.ok ? "Pass" : "Fail";
  const description = `${prefix} — ${result.reason}`.slice(0, 140);

  await githubApi(
    token,
    "POST",
    `/repos/${owner}/${repo}/statuses/${pullRequest.head.sha}`,
    {
      state: shouldPass ? "success" : "failure",
      context: "Issue gate",
      description,
      target_url: `https://github.com/${repository}/pull/${pullRequest.number}`,
    },
  );
}

async function run() {
  const token = process.env.GITHUB_TOKEN;
  const repository = process.env.GITHUB_REPOSITORY;
  const eventPath = process.env.GITHUB_EVENT_PATH;
  const pullNumber = Number.parseInt(process.env.PR_NUMBER, 10);
  const mode = (process.env.ISSUE_GATE_MODE || "observe").toLowerCase();
  const configuredCutoff = process.env.ISSUE_GATE_CUTOFF || "";
  const readyLabel = process.env.ISSUE_GATE_READY_LABEL || "state:ready";

  if (!token || !repository || !eventPath || !Number.isInteger(pullNumber)) {
    throw new Error("GITHUB_TOKEN, GITHUB_REPOSITORY, GITHUB_EVENT_PATH, and PR_NUMBER are required");
  }
  if (!VALID_MODES.has(mode)) {
    throw new Error(`ISSUE_GATE_MODE must be one of: ${[...VALID_MODES].join(", ")}`);
  }
  const cutoff = resolveCutoff(mode, configuredCutoff);

  const [owner, repo] = repository.split("/");
  const event = JSON.parse(await readFile(eventPath, "utf8"));
  const pullRequest =
    event.pull_request ??
    (await githubApi(token, "GET", `/repos/${owner}/${repo}/pulls/${pullNumber}`));

  const result = await evaluatePullRequest({
    pullRequest,
    repository,
    cutoff,
    readyLabel,
    getIssue: async (issueNumber) => {
      try {
        return await githubApi(
          token,
          "GET",
          `/repos/${owner}/${repo}/issues/${issueNumber}`,
        );
      } catch (error) {
        if (error instanceof ApiError && error.status === 404) {
          return null;
        }
        throw error;
      }
    },
  });

  await publishStatus({ token, repository, pullRequest, result, mode });
  if (shouldSynchronize(result)) {
    await updateGateLabel({ token, repository, pullRequest, result });
    await updateGateComment({
      token,
      repository,
      pullNumber: pullRequest.number,
      result,
      mode,
    });
  }

  if (mode === "close" && !result.ok) {
    await githubApi(
      token,
      "PATCH",
      `/repos/${owner}/${repo}/pulls/${pullRequest.number}`,
      { state: "closed" },
    );
  }

  console.log(JSON.stringify({ mode, pullNumber: pullRequest.number, ...result }));
  if (mode !== "observe" && !result.ok) {
    process.exitCode = 1;
  }
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  run().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
