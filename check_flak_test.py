import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# Conda environments to test
ENVS = ["crewai3.10", "crewaienv", "crewai3.12"]

# Test command arguments (after `python -m`)
PYTEST_ARGS = [
    "-k", "llama3_2",
    "/Users/viditostwal/Desktop/crewAI/tests/utilities/test_converter.py"
]


def run_single_test(env, index):
    """Run a single test in the given conda environment."""
    try:
        result = subprocess.run(
            ["conda", "run", "-n", env, "python", "-m", "pytest"] + PYTEST_ARGS,
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout + result.stderr
        if "1 passed" in output:
            return (index, True)
        else:
            # print(f"[{env}] ❌ Run {index}: Output missing '1 passed'\n{output}")
            return (index, False)
    except subprocess.CalledProcessError as e:
        # print(f"[{env}] ❌ Run {index}: Subprocess error\n{e.stderr or e.stdout}")
        return (index, False)


def run_tests_for_env(env):
    """Run 50 test runs in parallel for a given conda environment."""
    print(f"\n=== Running tests in conda environment: {env} ===")
    success_count = 0

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(run_single_test, env, i + 1) for i in range(50)]

        for future in as_completed(futures):
            index, success = future.result()
            if success:
                # print(f"[{env}] ✅ Run {index}: Success")
                success_count += 1

    print(f"\nSummary for {env}: {success_count} successful runs out of 50\n")


if __name__ == "__main__":
    for env in ENVS:
        run_tests_for_env(env)
