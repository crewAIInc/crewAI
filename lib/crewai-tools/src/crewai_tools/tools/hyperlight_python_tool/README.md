# HyperlightPythonTool

Run agent-written Python in a local, hardware-isolated [Hyperlight](https://github.com/hyperlight-dev/hyperlight) micro-VM through the [`hluk`](https://github.com/hyperlight-dev/hyperlight-unikraft) CLI. No account, API key, or daemon is needed.

Each call runs in a fresh micro-VM restored from a warm snapshot, so nothing carries over between calls, except in host directories mounted with `hluk_args`. The guest has no network access unless hosts are listed in `net_allow`.

## Installation

```shell
uv add crewai-tools
curl -fsSL https://raw.githubusercontent.com/hyperlight-dev/hyperlight-unikraft/main/install.sh | sh
```

`hluk` needs a hypervisor: KVM on Linux, WHP on Windows, Hypervisor.framework on Apple silicon macOS.

## Example

```python
from crewai_tools import HyperlightPythonTool

tool = HyperlightPythonTool()
result = tool.run(code="import numpy as np\nprint(np.arange(10).sum())")
# {'exit_code': 0, 'stdout': '45\n', 'stderr': '', 'timed_out': False}
```

| Parameter | Default | Description |
| --- | --- | --- |
| `hluk_path` | `hluk` on `PATH` | Path to the `hluk` binary |
| `runtime` | `"agent"` | Runtime image: CPython 3.12 with numpy, pandas, scipy, scikit-learn, matplotlib, pillow, requests |
| `net_allow` | `()` | Hosts or IPs the guest may connect to; a tuple, changed by assigning a new one |
| `envs` | `None` | Environment variables inside the sandbox; passed on the `hluk` command line, so not for secrets |
| `scratch_mb` | image default | Guest memory in MiB |
| `hluk_args` | `()` | Extra `hluk run` arguments, e.g. `["--mount", "/data:/data:ro"]` |
| `timeout` | `60` | Seconds a call's code may run |
| `startup_timeout` | `300` | Seconds the first call may take to pull the image and save the snapshot |
| `max_output_bytes` | 1 MiB | Bytes kept of stdout and of stderr; longer output keeps its beginning and end |
