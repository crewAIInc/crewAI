"""Commands whose shape is not settled yet.

What lives here still works exactly like the rest of the CLI — it is listed in
`crewai --help` and invoked the same way. The package says something narrower:
its behaviour, its output and its options may change between releases without
the deprecation cycle the settled commands get.

`crewai eval` is here because most of what it depends on lives outside this
repository — an AMP endpoint and the evaluator behind it — and both are new
enough that the report's shape is still moving. The command itself is
deliberately tolerant of that: it prints whatever the evaluation graded rather
than a list of its own.
"""
