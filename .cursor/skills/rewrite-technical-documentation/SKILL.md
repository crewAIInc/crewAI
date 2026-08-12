---
name: rewrite-technical-documentation
description: Rewrite technical documentation so it is clear, accurate, and easy to use. Use when rewriting docs, editing technical prose, clarifying developer guides, or when the user asks to rewrite, clean up, or improve documentation.
disable-model-invocation: true
---

# Rewrite Technical Documentation

Rewrite technical content so that it is clear, accurate, and easy to use. Apply the workflow to any source content. Do not assume a particular product, programming language, documentation system, or file format.

## Establish the source of truth

- Read the existing documentation and preserve its intended purpose.
- When source files are available, inspect the implementation, public interfaces, types, schemas, configuration, tests, examples, and generated output that define the documented behavior.
- Treat verified implementation behavior as authoritative when it conflicts with existing prose. Do not silently document an intended behavior that the source does not implement.
- Distinguish current behavior from proposed or planned behavior.
- Do not infer guarantees, validation rules, defaults, limits, permissions, error conditions, security properties, or compatibility from names or assumptions.
- If the available source is insufficient to verify a claim, either omit the claim, state the uncertainty, or ask for the missing source.

## Write the documentation

- Use plain, direct, technical English.
- Use short sentences and concrete descriptions.
- Keep the tone neutral and matter-of-fact.
- Use active voice when it makes the behavior clearer.
- Explain an unfamiliar term at its first necessary use.
- Use the same term for the same concept throughout the document.
- State required, optional, default, supported, and unsupported behavior explicitly when verified.
- Preserve exact names of commands, parameters, fields, functions, files, environment variables, and error messages.
- Use headings, lists, tables, and code blocks when they make information easier to find.
- Keep examples minimal and ensure that every example matches the actual interface and behavior.

## Remove stylistic noise

Remove or avoid:

- Aphorisms, slogans, metaphors, jokes, and rhetorical flourishes.
- Marketing language, hype, persuasion, and unsupported praise.
- Vague claims such as “seamless,” “powerful,” “robust,” or “easy” unless they have a specific, verifiable meaning.
- Repetition, unnecessary introductions, and commentary about the writing itself.
- Ambiguous pronouns and words such as “simply” when they do not add information.

The result should read like concise developer documentation: descriptive, precise, and impersonal.

## Preserve content and structure

- Preserve front matter, anchors, navigation metadata, links, identifiers, and required formatting unless the task asks for changes.
- Preserve technical meaning, caveats, limitations, and warnings.
- Do not remove an important detail merely to make the prose shorter.
- Do not add sections that are not useful for the content.
- Organize information in the order a reader needs it. Use an overview, prerequisites, inputs, outputs, procedure, examples, errors, or limitations only when those sections apply.

## Verify the result

Before finishing:

1. Compare each factual statement with the available source of truth.
2. Check names, syntax, examples, defaults, edge cases, and error behavior.
3. Check that links, code blocks, tables, and metadata remain valid.
4. Review the diff when editing files and avoid unrelated changes.
5. Report unresolved contradictions or missing source information instead of guessing.

When the user requests a rewrite of supplied text only, rewrite that text directly. When the user asks to update files, inspect and edit the relevant files, then report the files changed and any facts that could not be verified.
