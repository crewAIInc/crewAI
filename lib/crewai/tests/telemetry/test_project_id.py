"""Tests for the project_id used to link OSS usage to an enterprise account."""

import uuid

import pytest

from crewai_core.project import (
    get_or_create_project_id,
    get_project_id,
    parse_toml,
)


CREW_PYPROJECT = """\
[project]
name = "my_crew"
version = "0.1.0"
dependencies = ["crewai"]

[tool.crewai]
type = "crew"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
"""


@pytest.fixture
def pyproject(tmp_path):
    path = tmp_path / "pyproject.toml"
    path.write_text(CREW_PYPROJECT)
    return path


def test_returns_none_when_no_id_configured(pyproject):
    assert get_project_id(pyproject) is None


def test_mints_and_persists_an_id(pyproject):
    project_id = get_or_create_project_id(pyproject)

    assert uuid.UUID(project_id)
    assert get_project_id(pyproject) == project_id


def test_id_is_stable_across_calls(pyproject):
    first = get_or_create_project_id(pyproject)
    second = get_or_create_project_id(pyproject)

    assert first == second, "must not mint a second id"
    assert uuid.UUID(first)


def test_id_lands_in_the_tool_crewai_table(pyproject):
    project_id = get_or_create_project_id(pyproject)

    data = parse_toml(pyproject.read_text())
    assert data["tool"]["crewai"]["project_id"] == project_id
    assert data["tool"]["crewai"]["type"] == "crew", "existing keys must survive"


def test_other_tables_are_preserved(pyproject):
    get_or_create_project_id(pyproject)

    data = parse_toml(pyproject.read_text())
    assert data["project"]["name"] == "my_crew"
    assert data["project"]["dependencies"] == ["crewai"]
    assert data["build-system"]["build-backend"] == "hatchling.build"


def test_comments_and_formatting_are_preserved(tmp_path):
    """Raw-text editing rather than a TOML round-trip, so comments survive."""
    path = tmp_path / "pyproject.toml"
    path.write_text(
        '# top comment\n[project]\nname = "x"  # inline comment\n\n[tool.crewai]\ntype = "flow"\n'
    )

    get_or_create_project_id(path)

    content = path.read_text()
    assert "# top comment" in content
    assert "# inline comment" in content


@pytest.mark.parametrize(
    ("source", "label"),
    [
        ('[project]\nname = "x"\n\n[tool.crewai]\ntype = "crew"\n', "table then EOF"),
        ('[tool.crewai]\ntype = "crew"', "no trailing newline"),
        ('[project]\nname = "x"\n[tool.crewai]\n[other]\na = 1\n', "empty table"),
        (
            '[tool.crewai]\ntype = "crew"\n\n\n[build-system]\nrequires = []\n',
            "blank lines before next table",
        ),
    ],
)
def test_produces_valid_toml_for_varied_layouts(tmp_path, source, label):
    path = tmp_path / "pyproject.toml"
    path.write_text(source)

    project_id = get_or_create_project_id(path)

    assert project_id is not None, label
    data = parse_toml(path.read_text())
    assert data["tool"]["crewai"]["project_id"] == project_id, label


def test_id_does_not_leak_into_a_neighbouring_table(tmp_path):
    """The key must never land under [build-system]."""
    path = tmp_path / "pyproject.toml"
    path.write_text(
        '[tool.crewai]\ntype = "crew"\n\n[build-system]\nrequires = ["hatchling"]\n'
    )

    get_or_create_project_id(path)

    data = parse_toml(path.read_text())
    assert "project_id" in data["tool"]["crewai"]
    assert "project_id" not in data["build-system"]


def test_absent_tool_crewai_table_is_never_created(tmp_path):
    """Refuse to mint rather than rewrite a non-CrewAI project's pyproject.toml.

    `crewai run` in any directory that merely happens to have a pyproject.toml
    must not gain a [tool.crewai] table as a side effect.
    """
    path = tmp_path / "pyproject.toml"
    original = '[project]\nname = "unrelated"\n'
    path.write_text(original)

    assert get_or_create_project_id(path) is None
    assert path.read_text() == original, "unrelated project was modified"


def test_missing_file_is_not_an_error(tmp_path):
    assert get_or_create_project_id(tmp_path / "nope.toml") is None


def test_malformed_toml_is_not_an_error(tmp_path):
    path = tmp_path / "pyproject.toml"
    path.write_text("this is not [valid toml")

    assert get_project_id(path) is None


def test_read_only_file_is_not_an_error(pyproject):
    """A read-only checkout must not break the command that called this."""
    pyproject.chmod(0o444)
    try:
        project_id = get_or_create_project_id(pyproject)
    finally:
        pyproject.chmod(0o644)

    assert project_id is None


def test_get_project_id_never_creates_anything(pyproject):
    """Library code calls the read-only variant; it must not mutate the file."""
    before = pyproject.read_text()

    assert get_project_id(pyproject) is None

    assert pyproject.read_text() == before


@pytest.mark.parametrize("blank", ['""', "'   '", '"\\t"'])
def test_blank_or_whitespace_id_is_treated_as_absent(tmp_path, blank):
    """Whitespace is truthy in Python but is not an identity.

    Accepting it would propagate a useless value into login payloads and
    tracing context.
    """
    path = tmp_path / "pyproject.toml"
    path.write_text(f'[tool.crewai]\ntype = "crew"\nproject_id = {blank}\n')

    assert get_project_id(path) is None


def test_malformed_toml_is_never_written_to(tmp_path):
    """Appending to a file we cannot parse would corrupt it further."""
    path = tmp_path / "pyproject.toml"
    original = 'this is not [valid toml\nproject_id = "x'
    path.write_text(original)

    assert get_or_create_project_id(path) is None
    assert path.read_text() == original, "malformed file must be left untouched"


@pytest.mark.parametrize("blank", ['""', "''", '"   "', '"\\t\\t"'])
def test_blank_existing_id_is_replaced_not_duplicated(tmp_path, blank):
    """A blank id reads as absent; appending would make a duplicate key."""
    path = tmp_path / "pyproject.toml"
    path.write_text(f'[tool.crewai]\ntype = "crew"\nproject_id = {blank}\n')

    project_id = get_or_create_project_id(path)

    content = path.read_text()
    assert content.count("project_id") == 1, f"duplicate key: {content!r}"
    data = parse_toml(content)  # would raise on a duplicate key
    assert data["tool"]["crewai"]["project_id"] == project_id
    assert data["tool"]["crewai"]["type"] == "crew"
    assert uuid.UUID(project_id), "must mint a real id, not keep the blank one"


def test_non_string_existing_id_is_replaced(tmp_path):
    path = tmp_path / "pyproject.toml"
    path.write_text("[tool.crewai]\nproject_id = 42\n")

    project_id = get_or_create_project_id(path)

    data = parse_toml(path.read_text())
    assert data["tool"]["crewai"]["project_id"] == project_id
    assert isinstance(project_id, str)


@pytest.mark.parametrize(
    "header",
    [
        "[tool.crewai]  # crewai config",
        "[tool.crewai]# no space",
        "[tool.crewai]\t# tab then comment",
    ],
)
def test_table_header_with_trailing_comment_is_found(tmp_path, header):
    """A commented header is valid TOML; missing it appends a duplicate table."""
    path = tmp_path / "pyproject.toml"
    path.write_text(f'{header}\ntype = "crew"\n')

    project_id = get_or_create_project_id(path)

    content = path.read_text()
    assert content.count("[tool.crewai]") == 1, f"duplicate table: {content!r}"
    data = parse_toml(content)  # would raise on a redefined table
    assert data["tool"]["crewai"]["project_id"] == project_id
    assert data["tool"]["crewai"]["type"] == "crew"


def test_similar_table_names_are_not_matched(tmp_path):
    """[tool.crewai-extra] must not be mistaken for [tool.crewai]."""
    path = tmp_path / "pyproject.toml"
    path.write_text('[tool.crewai-extra]\nfoo = 1\n\n[tool.crewai]\ntype = "crew"\n')

    project_id = get_or_create_project_id(path)

    data = parse_toml(path.read_text())
    assert data["tool"]["crewai"]["project_id"] == project_id
    assert "project_id" not in data["tool"]["crewai-extra"]


def test_crlf_line_endings_are_preserved(tmp_path):
    """read_text/write_text would silently rewrite the whole file as LF."""
    path = tmp_path / "pyproject.toml"
    path.write_bytes(b'[project]\r\nname = "x"\r\n\r\n[tool.crewai]\r\ntype = "crew"\r\n')

    project_id = get_or_create_project_id(path)

    raw = path.read_bytes()
    assert b"\r\n" in raw
    assert raw.count(b"\n") == raw.count(b"\r\n"), "mixed line endings introduced"
    assert parse_toml(raw.decode())["tool"]["crewai"]["project_id"] == project_id


def test_lf_file_stays_lf(tmp_path):
    path = tmp_path / "pyproject.toml"
    path.write_bytes(b'[tool.crewai]\ntype = "crew"\n')

    get_or_create_project_id(path)

    assert b"\r\n" not in path.read_bytes()


def test_concurrent_minting_converges_on_one_id(tmp_path):
    """Concurrent minters must all return the id that ends up on disk.

    Uses threads in one process, so it covers the read-modify-write race rather
    than the cross-process lock backend itself.
    """
    import threading

    workers = 8
    path = tmp_path / "pyproject.toml"
    path.write_text(CREW_PYPROJECT)

    returned: list[str | None] = []
    results_lock = threading.Lock()
    # Timed out rather than unbounded: a thread dying before the barrier, or
    # blocking on the lock, would otherwise hang CI instead of failing.
    start = threading.Barrier(workers, timeout=30)

    def mint() -> None:
        start.wait()
        project_id = get_or_create_project_id(path)
        with results_lock:
            returned.append(project_id)

    threads = [threading.Thread(target=mint) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not [t for t in threads if t.is_alive()], "thread did not finish in time"
    assert len(returned) == workers, f"only {len(returned)}/{workers} threads returned"

    persisted = parse_toml(path.read_text())["tool"]["crewai"]["project_id"]
    assert set(returned) == {persisted}, (
        f"callers disagreed with disk: returned={set(returned)} persisted={persisted}"
    )


def test_file_mode_is_preserved(tmp_path):
    """The atomic replace must not widen permissions on pyproject.toml."""
    path = tmp_path / "pyproject.toml"
    path.write_text(CREW_PYPROJECT)
    path.chmod(0o600)

    get_or_create_project_id(path)

    assert path.stat().st_mode & 0o777 == 0o600


def test_no_temp_files_left_behind(tmp_path):
    path = tmp_path / "pyproject.toml"
    path.write_text(CREW_PYPROJECT)

    get_or_create_project_id(path)

    assert [p.name for p in tmp_path.iterdir()] == ["pyproject.toml"]


def test_ids_are_unique_across_projects(tmp_path):
    ids = set()
    for name in ("a", "b", "c"):
        path = tmp_path / name / "pyproject.toml"
        path.parent.mkdir()
        path.write_text(CREW_PYPROJECT)
        project_id = get_or_create_project_id(path)
        ids.add(project_id)

    assert len(ids) == 3
