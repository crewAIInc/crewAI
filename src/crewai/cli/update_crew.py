import shutil

import tomli_w
import tomllib


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

    # Read the input pyproject.toml
    with open(input_file, "rb") as f:
        pyproject = tomllib.load(f)

    # Initialize the new project structure
    new_pyproject = {
        "project": {},
        "build-system": {"requires": ["hatchling"], "build-backend": "hatchling.build"},
    }

    # Migrate project metadata
    if "tool" in pyproject and "poetry" in pyproject["tool"]:
        poetry = pyproject["tool"]["poetry"]
        new_pyproject["project"]["name"] = poetry.get("name")
        new_pyproject["project"]["version"] = poetry.get("version")
        new_pyproject["project"]["description"] = poetry.get("description")
        new_pyproject["project"]["authors"] = [
            {
                "name": author.split("<")[0].strip(),
                "email": author.split("<")[1].strip(">").strip(),
            }
            for author in poetry.get("authors", [])
        ]
        new_pyproject["project"]["requires-python"] = poetry.get("python")
    else:
        # If it's already in the new format, just copy the project section
        new_pyproject["project"] = pyproject.get("project", {})

    # Migrate or copy dependencies
    if "dependencies" in new_pyproject["project"]:
        # If dependencies are already in the new format, keep them as is
        pass
    elif "dependencies" in poetry:
        new_pyproject["project"]["dependencies"] = []
        for dep, version in poetry["dependencies"].items():
            if isinstance(version, dict):  # Handle extras
                extras = ",".join(version.get("extras", []))
                new_dep = f"{dep}[{extras}]"
                if "version" in version:
                    new_dep += f"{version['version']}"
            elif dep == "python":
                new_pyproject["project"]["requires-python"] = version
                continue
            else:
                new_dep = f"{dep}{version}"
            new_pyproject["project"]["dependencies"].append(new_dep)

    # Migrate or copy scripts
    if "scripts" in poetry:
        new_pyproject["project"]["scripts"] = poetry["scripts"]
    elif "scripts" in pyproject.get("project", {}):
        new_pyproject["project"]["scripts"] = pyproject["project"]["scripts"]

    # Migrate optional dependencies
    if "extras" in poetry:
        new_pyproject["project"]["optional-dependencies"] = poetry["extras"]

    # Backup the old pyproject.toml
    backup_file = "pyproject-old.toml"
    shutil.copy2(input_file, backup_file)
    print(f"Original pyproject.toml backed up as {backup_file}")

    # Write the new pyproject.toml
    with open(output_file, "wb") as f:
        tomli_w.dump(new_pyproject, f)

    print(f"Migration complete. New pyproject.toml written to {output_file}")
