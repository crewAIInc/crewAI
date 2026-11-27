# Security policy

Thank you for helping keep CrewAI and the people who use it safe. This page
explains how to report a security issue, what to include, what is in scope,
and what you can expect from us.

## How to report

- **A vulnerability in this repository's code:** open a private report at
  https://github.com/crewAIInc/crewAI/security/advisories/new. The details
  stay private, and we can work on the fix, the advisory, and the CVE with
  you in one place.
- **Anything else** (the CrewAI platform at app.crewai.com, Factory,
  crewAI-tools, another CrewAI service, or you are not sure where it
  belongs): email **security@crewai.com**.

Please do not report security issues through public GitHub issues, pull
requests, discussions, or social media.

## What to include

- The product and the version, commit, or URL affected.
- Where the problem is: endpoint, file, function, or setting.
- Steps to reproduce, or a proof of concept. Plain text beats screenshots;
  please do not send executables.
- What an attacker could do with it.
- How you would like to be credited, if at all.

## Scope

- crewAI (this repository) and crewAI-tools
- The CrewAI AMP platform at app.crewai.com
- CrewAI Factory releases

## Out of scope

- Third-party services and sites CrewAI does not operate
- Denial of service, load testing, and other volumetric testing
- Social engineering and physical attacks
- Scanner output without demonstrated impact: missing headers, SPF, DMARC
  or CAA records, version banners, rate limiting
- Generic "the LLM can be jailbroken" findings without a crewAI-specific
  defect
- trust.crewai.com, which is operated by Vanta, except where the finding
  concerns CrewAI's own content on it

If a report is out of scope, we will tell you so in one reply.

## What we commit to

- Acknowledgment within 2 business days.
- A substantive response within 10 business days: confirmed, not
  reproducible, out of scope, already known, or still investigating with a
  date for the next update.
- An update at least every 14 days until the report is closed.
- Coordinated disclosure. We publish the fix and the advisory together, by
  default within 90 days of acknowledgment; earlier if the fix ships sooner,
  later only by agreement with you.
- A CVE, through GitHub's CNA, for confirmed vulnerabilities in publicly
  distributed CrewAI products.
- Credit in the advisory if you want it.

## Bounties

CrewAI does not run a paid bug bounty program. We credit reporters who want
credit.

## Safe harbor

CrewAI will not pursue or support legal action against anyone who reports a
security issue in good faith and follows these rules:

- Test only against your own account, your own self-hosted Factory instance,
  or your own copy of our open-source software.
- Do not access, modify, copy, or keep data that is not yours. If you
  encounter customer or personal data, stop and tell us immediately.
- Do not degrade our service: no denial-of-service testing, no bulk automated
  scanning against AMP, no spam.
- No social engineering of CrewAI staff, customers, or vendors; no physical
  attempts against CrewAI property.
- Report promptly via private vulnerability reporting for this repository's
  code, or security@crewai.com for other issues. Give us the agreed time to
  fix before disclosing publicly.

Good-faith research within these rules is authorized access for the purposes
of applicable computer misuse laws. Activity outside these rules is not
covered.

## Canonical policy

This page is the canonical version of CrewAI's disclosure policy. A copy at
https://crewai.com/security will link back here.
