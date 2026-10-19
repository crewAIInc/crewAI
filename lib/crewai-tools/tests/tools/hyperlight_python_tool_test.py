import json
import os
import shutil
import stat
import sys
import textwrap
import time

from crewai_tools import HyperlightPythonTool
from pydantic import ValidationError
import pytest


# A stand-in for hluk that runs the guest's code with the host's Python, as one
# process so that stopping it stops the code. It logs its arguments, one JSON
# list per line, to FAKE_HLUK_LOG.
FAKE_HLUK = textwrap.dedent(
    """\
    import json
    import os
    import sys

    args = sys.argv[1:]
    with open(os.environ["FAKE_HLUK_LOG"], "a") as log:
        log.write(json.dumps(args) + "\\n")
    if args == ["--version"]:
        if os.environ.get("FAKE_HLUK_VERSION_FAILS"):
            sys.stderr.write("GLIBC_2.39 not found\\n")
            sys.exit(1)
        print(os.environ.get("FAKE_HLUK_VERSION", "hluk 0.18.0"))
        sys.exit(0)
    env = dict(os.environ)
    code = script = None
    rest = args[1:]
    while rest:
        arg = rest.pop(0)
        if arg == "--env":
            key, value = rest.pop(0).split("=", 1)
            env[key] = value
        elif arg == "--exec":
            code = rest.pop(0)
        elif arg.startswith("--"):
            rest.pop(0)
        else:
            script = arg
    if code == "pass" and os.environ.get("FAKE_HLUK_FAIL_START"):
        sys.stderr.write("no hypervisor found\\n")
        sys.exit(1)
    argv = [sys.executable, "-c", code] if code is not None else [sys.executable, script]
    os.execve(sys.executable, argv, env)
    """
)


@pytest.fixture
def hluk_log(tmp_path, monkeypatch):
    log = tmp_path / "hluk.log"
    monkeypatch.setenv("FAKE_HLUK_LOG", str(log))
    return log


@pytest.fixture
def fake_hluk(tmp_path, hluk_log):
    path = tmp_path / "hluk"
    path.write_text(f"#!{sys.executable}\n{FAKE_HLUK}")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def hluk_calls(log):
    return [json.loads(line) for line in log.read_text().splitlines()]


pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the stand-in hluk is a script run by its shebang"
)


def test_returns_stdout_and_exit_code(fake_hluk):
    tool = HyperlightPythonTool(hluk_path=fake_hluk)

    result = tool.run(code="print(6 * 7)")

    assert result == {"exit_code": 0, "stdout": "42\n", "stderr": "", "timed_out": False}


def test_reports_failing_code(fake_hluk):
    tool = HyperlightPythonTool(hluk_path=fake_hluk)

    result = tool.run(code="raise ValueError('boom')")

    assert result["exit_code"] == 1
    assert "ValueError: boom" in result["stderr"]
    assert result["timed_out"] is False


def test_stops_code_that_runs_too_long(fake_hluk):
    tool = HyperlightPythonTool(hluk_path=fake_hluk, timeout=1)
    tool.run(code="pass")

    started = time.monotonic()
    result = tool.run(code="print('started', flush=True)\nwhile True: pass")

    assert time.monotonic() - started < 10
    assert result["timed_out"] is True
    assert result["exit_code"] is None
    assert result["stdout"] == "started\n"
    assert "Timed out after 1 second;" in result["stderr"]


def test_keeps_beginning_and_end_of_long_output(fake_hluk):
    tool = HyperlightPythonTool(hluk_path=fake_hluk, max_output_bytes=100)

    result = tool.run(code="print('start' + 'x' * 10_000 + 'end')")

    assert result["stdout"].startswith("start")
    assert result["stdout"].endswith("end\n")
    assert "bytes of output truncated" in result["stdout"]
    assert len(result["stdout"]) < 200


def test_passes_sandbox_options_to_hluk(fake_hluk, hluk_log):
    tool = HyperlightPythonTool(
        hluk_path=fake_hluk,
        runtime="python",
        net_allow=["pypi.org"],
        envs={"GREETING": "hello"},
        scratch_mb=512,
        hluk_args=["--mount", "/data:/data:ro"],
    )

    result = tool.run(code="import os; print(os.environ['GREETING'])")

    assert result["stdout"] == "hello\n"
    run = hluk_calls(hluk_log)[-1]
    assert run[:3] == ["run", "--runtime", "python"]
    assert run[3:-1] == [
        "--net-allow",
        "pypi.org",
        "--env",
        "GREETING=hello",
        "--scratch-mb",
        "512",
        "--mount",
        "/data:/data:ro",
    ]
    assert run[-1].endswith(".py")


def test_prepares_the_sandbox_once(fake_hluk, hluk_log):
    tool = HyperlightPythonTool(hluk_path=fake_hluk)

    tool.run(code="print(1)")
    tool.run(code="print(2)")

    calls = hluk_calls(hluk_log)
    assert calls.count(["--version"]) == 1
    assert sum(call[-2:] == ["--exec", "pass"] for call in calls) == 1
    assert len(calls) == 4


def test_removes_the_script_after_each_call(fake_hluk, hluk_log):
    tool = HyperlightPythonTool(hluk_path=fake_hluk)

    tool.run(code="print(1)")

    assert not os.path.exists(hluk_calls(hluk_log)[-1][-1])


def test_returns_when_something_hluk_left_holds_its_output(tmp_path, fake_hluk):
    # A wrapper that leaves a process behind with hluk's output pipes
    wrapper = tmp_path / "hluk-wrapper"
    wrapper.write_text(f'#!/bin/sh\nsleep 5 &\nexec {fake_hluk} "$@"\n')
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
    tool = HyperlightPythonTool(hluk_path=str(wrapper))
    tool.run(code="pass")

    started = time.monotonic()
    result = tool.run(code="print('done')")

    assert result["stdout"] == "done\n"
    assert time.monotonic() - started < 3


def test_reports_a_sandbox_that_cannot_start(fake_hluk, monkeypatch):
    monkeypatch.setenv("FAKE_HLUK_FAIL_START", "1")
    tool = HyperlightPythonTool(hluk_path=fake_hluk)

    with pytest.raises(RuntimeError, match=r"\(exit status 1\): no hypervisor found"):
        tool.run(code="print(1)")


@pytest.mark.parametrize("version", ["hluk 0.17.2", "hluk 0.18.0-rc1", "unknown"])
def test_rejects_an_old_hluk(fake_hluk, monkeypatch, version):
    monkeypatch.setenv("FAKE_HLUK_VERSION", version)
    tool = HyperlightPythonTool(hluk_path=fake_hluk)

    with pytest.raises(RuntimeError, match="needs hluk v0.18.0 or later"):
        tool.run(code="print(1)")


def test_reports_a_missing_hluk(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    tool = HyperlightPythonTool()

    with pytest.raises(RuntimeError, match="Could not find the `hluk` binary on PATH"):
        tool.run(code="print(1)")


def test_reports_a_hluk_that_is_not_executable(tmp_path):
    hluk = tmp_path / "hluk"
    hluk.write_text("")
    tool = HyperlightPythonTool(hluk_path=str(hluk))

    with pytest.raises(RuntimeError, match="is not executable"):
        tool.run(code="print(1)")


def test_reports_a_wrong_hluk_path(tmp_path):
    tool = HyperlightPythonTool(hluk_path=str(tmp_path / "nowhere" / "hluk"))

    with pytest.raises(RuntimeError, match="Could not find the `hluk` binary at"):
        tool.run(code="print(1)")


def test_prepares_the_sandbox_again_when_its_configuration_changes(
    fake_hluk, hluk_log
):
    tool = HyperlightPythonTool(hluk_path=fake_hluk)
    tool.run(code="print(1)")

    tool.runtime = "python"
    tool.run(code="print(2)")

    warm_ups = [call for call in hluk_calls(hluk_log) if call[-2:] == ["--exec", "pass"]]
    assert [call[2] for call in warm_ups] == ["agent", "python"]


def test_cuts_long_output_between_characters(fake_hluk):
    # Each half (51 bytes) ends in the middle of a two-byte character
    tool = HyperlightPythonTool(hluk_path=fake_hluk, max_output_bytes=102)

    result = tool.run(code="print('é' * 1000, end='')")

    assert "bytes of output truncated" in result["stdout"]
    assert "�" not in result["stdout"]


def test_runs_code_with_a_lone_surrogate(fake_hluk):
    tool = HyperlightPythonTool(hluk_path=fake_hluk)

    result = tool.run(code="print(len('\ud83d'))")

    assert result["stdout"] == "1\n"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("timeout", 0),
        ("timeout", float("inf")),
        ("startup_timeout", 0),
        ("startup_timeout", float("nan")),
        ("timeout", 1e7),
        ("max_output_bytes", 0),
    ],
)
def test_rejects_limits_that_are_not_positive_and_finite(field, value):
    with pytest.raises(ValidationError):
        HyperlightPythonTool(**{field: value})


@pytest.mark.parametrize(
    ("timeout", "text"),
    [(1, "1 second "), (1.0000001, "1 second "), (2.5, "2.5 seconds"), (1e6, "1000000 seconds")],
)
def test_states_the_time_limit_readably(timeout, text):
    assert text in HyperlightPythonTool(timeout=timeout).description


def test_keeps_whole_version_output_whatever_the_output_limit(fake_hluk):
    tool = HyperlightPythonTool(hluk_path=fake_hluk, max_output_bytes=4)

    assert tool.run(code="print('ok')")["exit_code"] == 0


def test_reports_a_hluk_that_cannot_run(fake_hluk, monkeypatch):
    monkeypatch.setenv("FAKE_HLUK_VERSION_FAILS", "1")
    tool = HyperlightPythonTool(hluk_path=fake_hluk)

    with pytest.raises(RuntimeError, match="--version` failed: GLIBC_2.39 not found"):
        tool.run(code="print(1)")


def test_checks_the_version_once_per_binary(fake_hluk, hluk_log):
    tool = HyperlightPythonTool(hluk_path=fake_hluk)
    tool.run(code="print(1)")

    tool.runtime = "python"
    tool.run(code="print(2)")

    assert hluk_calls(hluk_log).count(["--version"]) == 1


def test_reports_a_bare_name_missing_from_path(tmp_path, monkeypatch):
    (tmp_path / "hluk").write_text("")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", "")
    tool = HyperlightPythonTool(hluk_path="hluk")

    with pytest.raises(RuntimeError, match="Could not find the `hluk` binary at"):
        tool.run(code="print(1)")


def test_describes_what_the_sandbox_offers():
    default = HyperlightPythonTool()
    assert "numpy" in default.description
    assert "no network access" in default.description
    assert "60 seconds" in default.description

    tool = HyperlightPythonTool(runtime="python", net_allow=["pypi.org"], timeout=5)
    assert "numpy" not in tool.description
    assert "can connect only to: pypi.org" in tool.description
    assert "5 seconds" in tool.description

    tool.timeout = 9
    assert "9 seconds" in tool.description


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("pypi.org.", "pypi.org. Install packages with pip before importing them."),
        ("60 seconds", "60 seconds or uses more than 512 MiB of memory"),
    ],
)
def test_keeps_a_generated_description_the_caller_edited(old, new):
    generated = HyperlightPythonTool(net_allow=("pypi.org",)).description
    edited = generated.replace(old, new)

    tool = HyperlightPythonTool(net_allow=("pypi.org",), description=edited)

    assert tool.description == edited


@pytest.mark.parametrize("host", ["", "pypi.org ", "a.com,b.com"])
def test_rejects_a_host_that_is_not_one(host):
    with pytest.raises(ValidationError, match="not a host name or IP"):
        HyperlightPythonTool(net_allow=(host,))


@pytest.mark.parametrize("name", ["", "A=B"])
def test_rejects_an_environment_variable_name_that_is_not_one(name):
    with pytest.raises(ValidationError, match="not an environment variable name"):
        HyperlightPythonTool(envs={name: "x"})


@pytest.mark.parametrize(
    "args", [["--net"], ["--net-block", "a.com"], ["--port", "8080"], ["--port=all"]]
)
def test_describes_network_access_opened_by_hluk_args(args):
    tool = HyperlightPythonTool(hluk_args=args)

    assert "may have network access" in tool.description
    assert "no network access" not in tool.description


def test_prepares_the_sandbox_again_after_a_timeout(fake_hluk, hluk_log):
    tool = HyperlightPythonTool(hluk_path=fake_hluk, timeout=1)
    tool.run(code="while True: pass")

    tool.run(code="print(1)")

    warm_ups = [call for call in hluk_calls(hluk_log) if call[-2:] == ["--exec", "pass"]]
    assert len(warm_ups) == 2


def test_rejects_an_environment_variable_name_added_in_place(fake_hluk):
    tool = HyperlightPythonTool(hluk_path=fake_hluk, envs={})
    tool.envs["A=B"] = "1"

    with pytest.raises(ValueError, match="not an environment variable name"):
        tool.run(code="print(1)")


def test_keeps_the_end_of_long_output_that_is_not_utf8(fake_hluk):
    tool = HyperlightPythonTool(hluk_path=fake_hluk, max_output_bytes=100)

    result = tool.run(
        code="import sys; sys.stdout.buffer.write(b'a' * 200 + b'\\x80' * 60 + b'tail')"
    )

    assert result["stdout"].endswith("\ufffd" * 43 + "tail")


def test_a_timed_out_call_does_not_wait_for_another_warm_up(fake_hluk, monkeypatch):
    tool = HyperlightPythonTool(hluk_path=fake_hluk, timeout=1)
    command = tool._base_command(fake_hluk)
    # Past its own preparation, while another thread's warm-up holds the lock
    monkeypatch.setattr(tool, "_prepare", lambda: command)
    with tool._lock:
        started = time.monotonic()
        result = tool.run(code="while True: pass")
        elapsed = time.monotonic() - started

    assert result["timed_out"] is True
    assert elapsed < 5


def test_keeps_a_subclass_description():
    class FinanceTool(HyperlightPythonTool):
        description: str = "Run the finance team's analysis scripts."

    assert FinanceTool().description == "Run the finance team's analysis scripts."


def test_never_caches_results():
    tool = HyperlightPythonTool()

    assert tool.cache_function({"code": "print(1)"}, {"exit_code": 0}) is False


def test_keeps_a_description_the_caller_wrote():
    tool = HyperlightPythonTool(description="Run Python.", runtime="python")
    assert tool.description == "Run Python."

    tool.timeout = 9
    assert tool.description == "Run Python."


def test_describes_the_agent_image_by_its_full_reference():
    published = "ghcr.io/hyperlight-dev/hyperlight-unikraft/agent:initrd-v0.18.0"
    assert "numpy" in HyperlightPythonTool(runtime=published).description

    own = "registry.example.com/team/agent:1"
    assert "numpy" not in HyperlightPythonTool(runtime=own).description


def test_keeps_the_description_in_sync_after_a_round_trip():
    tool = HyperlightPythonTool.model_validate(HyperlightPythonTool().model_dump())
    tool.timeout = 2
    assert "2 seconds" in tool.description

    changed = HyperlightPythonTool.model_validate(
        {**HyperlightPythonTool().model_dump(), "timeout": 5}
    )
    assert "5 seconds" in changed.description


@pytest.mark.skipif(
    shutil.which("hluk") is None
    or (
        sys.platform == "linux"
        # The released hluk is built for KVM only on Linux
        and not os.access("/dev/kvm", os.R_OK | os.W_OK)
    ),
    reason="needs hluk and a hypervisor",
)
def test_runs_code_in_a_real_micro_vm():
    tool = HyperlightPythonTool()

    result = tool.run(code="import numpy as np\nprint(np.arange(5).sum())")

    assert result["exit_code"] == 0
    assert result["stdout"] == "10\n"
