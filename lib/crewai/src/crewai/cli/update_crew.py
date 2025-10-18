import os
import shutil

import tomli_w

from crewai.cli.utils import read_toml


def update_crew() -> None:
    """Update the pyproject.toml of the Crew project to use uv."""
    migrate_pyproject("pyproject.toml", "pyproject.toml")


def migrate_pyproject(input_file, output_file):
    """
    Migrate the pyproject.toml to the new format.

    This function is used to migrate the pyproject.toml to the new format.
    And it will be used to migrate the pyproject.toml to the new format when uv is used.
    When the time comes that uv supports the new format, this function will be deprecated.
    """
    poetry_data = {}
    # Read the input pyproject.toml
    pyproject_data = read_toml()

    # Initialize the new project structure
    new_pyproject = {
        "project": {},
        "build-system": {"requires": ["hatchling"], "build-backend": "hatchling.build"},
    }

    # Migrate project metadata
    if "tool" in pyproject_data and "poetry" in pyproject_data["tool"]:
        poetry_data = pyproject_data["tool"]["poetry"]
        new_pyproject["project"]["name"] = poetry_data.get("name")
        new_pyproject["project"]["version"] = poetry_data.get("version")
        new_pyproject["project"]["description"] = poetry_data.get("description")
        new_pyproject["project"]["authors"] = [
            {
                "name": author.split("<")[0].strip(),
                "email": author.split("<")[1].strip(">").strip(),
            }
            for author in poetry_data.get("authors", [])
        ]
        new_pyproject["project"]["requires-python"] = poetry_data.get("python")
    else:
        # If it's already in the new format, just copy the project and tool sections
        new_pyproject["project"] = pyproject_data.get("project", {})
        new_pyproject["tool"] = pyproject_data.get("tool", {})

    # Migrate or copy dependencies
    if "dependencies" in new_pyproject["project"]:
        # If dependencies are already in the new format, keep them as is
        pass
    elif poetry_data and "dependencies" in poetry_data:
        new_pyproject["project"]["dependencies"] = []
        for dep, version in poetry_data["dependencies"].items():
            if isinstance(version, dict):  # Handle extras
                extras = ",".join(version.get("extras", []))
                new_dep = f"{dep}[{extras}]"
                if "version" in version:
                    new_dep += parse_version(version["version"])
            elif dep == "python":
                new_pyproject["project"]["requires-python"] = version
                continue
            else:
                new_dep = f"{dep}{parse_version(version)}"
            new_pyproject["project"]["dependencies"].append(new_dep)

    # Migrate or copy scripts
    if poetry_data and "scripts" in poetry_data:
        new_pyproject["project"]["scripts"] = poetry_data["scripts"]
    elif pyproject_data.get("project", {}) and "scripts" in pyproject_data["project"]:
        new_pyproject["project"]["scripts"] = pyproject_data["project"]["scripts"]
    else:
        new_pyproject["project"]["scripts"] = {}

    if (
        "run_crew" not in new_pyproject["project"]["scripts"]
        and len(new_pyproject["project"]["scripts"]) > 0
    ):
        # Extract the module name from any existing script
        existing_scripts = new_pyproject["project"]["scripts"]
        module_name = next(
            (value.split(".")[0] for value in existing_scripts.values() if "." in value)
        )

        new_pyproject["project"]["scripts"]["run_crew"] = f"{module_name}.main:run"

    # Migrate optional dependencies
    if poetry_data and "extras" in poetry_data:
        new_pyproject["project"]["optional-dependencies"] = poetry_data["extras"]

    # Backup the old pyproject.toml
    backup_file = "pyproject-old.toml"
    shutil.copy2(input_file, backup_file)

    # Rename the poetry.lock file
    lock_file = "poetry.lock"
    lock_backup = "poetry-old.lock"
    if os.path.exists(lock_file):
        os.rename(lock_file, lock_backup)
    else:
        pass

    # Write the new pyproject.toml
    with open(output_file, "wb") as f:
        tomli_w.dump(new_pyproject, f)


def parse_version(version: str) -> str:
    """Parse and convert version specifiers."""
    if version.startswith("^"):
        main_lib_version = version[1:].split(",")[0]
        addtional_lib_version = None
        if len(version[1:].split(",")) > 1:
            addtional_lib_version = version[1:].split(",")[1]

        return f">={main_lib_version}" + (
            f",{addtional_lib_version}" if addtional_lib_version else ""
        )
    return version
