# ExtraDesktopTool

## Description

The `ExtraDesktopTool` provides autonomous desktop computer-use and GUI automation capabilities for CrewAI agents across Linux, Windows, and macOS, powered by [Project Extra](https://github.com/AIYantra/extra).

It enables agents to interact directly with desktop operating systems and native GUI applications with sub-3ms perception, live Bezier mouse movements, instant Win32/uinput typing, window management, and compound macro actions.

## Key Capabilities

- **Cross-Platform Support**: Native automation across Windows (Win32), Linux (Wayland/X11), and macOS (Quartz/Accessibility).
- **Application Lifecycle**: Visibly launch and manage desktop applications (`calc`, `notepad`, `code`, browsers, terminals).
- **Visual Grounding & Actuation**: Click UI elements via semantic target labels or pixel coordinates with human-like Bezier cursor flight.
- **Hardware Event Injection**: Sub-millisecond keystroke typing and multi-key global hotkey combinations (`Ctrl+S`, `Alt+Tab`).
- **Screen Perception**: High-speed desktop screen capture and OCR/accessibility tree inspection.
- **Window Layouts**: Sub-15ms programmatic window snapping (`side_by_side`).
- **Compound Workflows**: Atomic batch action execution and closed-loop macro intent compilation.

## Installation

Install the `extra-desktop` package alongside CrewAI tools:

```bash
pip install extra-desktop crewai-tools
```

Or install with optional extras:

```bash
pip install 'crewai-tools[extra-desktop]'
```

## Example Usage

```python
from crewai import Agent, Task, Crew
from crewai_tools import ExtraDesktopTool

# Initialize Extra Desktop Tool
desktop_tool = ExtraDesktopTool()

# Define an autonomous desktop operator agent
operator = Agent(
    role="Desktop Automation Specialist",
    goal="Automate repetitive GUI tasks, open desktop tools, and compute results",
    backstory="An expert operator capable of piloting any desktop environment with precision.",
    tools=[desktop_tool],
    verbose=True,
)

# Define task to launch calculator and perform a calculation
task = Task(
    description="Launch the desktop calculator and calculate 125 * 8.",
    expected_output="Calculation executed on the desktop screen and confirmation returned.",
    agent=operator,
)

crew = Crew(agents=[operator], tasks=[task])
crew.kickoff()
```

## Direct Tool Invocation

You can also invoke actions directly:

```python
from crewai_tools import ExtraDesktopTool

tool = ExtraDesktopTool()

# Launch an application
tool.run(action="launch", app_name="calc")

# Type into the focused application
tool.run(action="type", text="125*8=", press_enter=True)

# Capture the desktop screen
result = tool.run(action="screenshot")
print(result)

# Snap windows side-by-side
tool.run(action="snap_layout", layout="side_by_side", left_window="Calculator", right_window="Notepad")
```

## Arguments

- `action` (required): The action to perform (`"launch"`, `"click"`, `"type"`, `"hotkey"`, `"screenshot"`, `"snap_layout"`, `"batch_actions"`, `"subtask_run"`).
- `app_name` (optional): Name, executable, or path of application to launch.
- `args` (optional): Command line arguments when launching an application.
- `target` (optional): Semantic label or query of UI element to click.
- `x`, `y` (optional): Pixel coordinates for mouse click or movement.
- `button` (optional): Mouse button (`"left"`, `"right"`, `"middle"`). Default is `"left"`.
- `text` (optional): Text to type into the active window.
- `press_enter` (optional): Whether to press Enter after typing. Default is `False`.
- `keys` (optional): List of keys for keyboard shortcuts (e.g., `["ctrl", "s"]`).
- `intent` (optional): Closed-loop natural language intent for `subtask_run`.
- `actions` (optional): List of compound hardware action dictionaries for `batch_actions`.
- `layout` (optional): Window snap layout (`"side_by_side"`).
- `left_window`, `right_window` (optional): Window titles for side-by-side snapping.
