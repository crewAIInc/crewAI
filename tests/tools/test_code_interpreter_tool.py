import unittest
from unittest.mock import patch

from crewai_tools.tools.code_interpreter_tool.code_interpreter_tool import (
    CodeInterpreterTool,
)


class TestCodeInterpreterTool(unittest.TestCase):
    @patch("crewai_tools.tools.code_interpreter_tool.code_interpreter_tool.docker")
    def test_run_code_in_docker(self, docker_mock):
        tool = CodeInterpreterTool()
        code = "print('Hello, World!')"
        libraries_used = "numpy,pandas"
        expected_output = "Hello, World!\n"

        docker_mock.from_env().containers.run().exec_run().exit_code = 0
        docker_mock.from_env().containers.run().exec_run().output = (
            expected_output.encode()
        )
        result = tool.run_code_in_docker(code, libraries_used)

        self.assertEqual(result, expected_output)

    @patch("crewai_tools.tools.code_interpreter_tool.code_interpreter_tool.docker")
    def test_run_code_in_docker_with_error(self, docker_mock):
        tool = CodeInterpreterTool()
        code = "print(1/0)"
        libraries_used = "numpy,pandas"
        expected_output = "Something went wrong while running the code: \nZeroDivisionError: division by zero\n"

        docker_mock.from_env().containers.run().exec_run().exit_code = 1
        docker_mock.from_env().containers.run().exec_run().output = (
            b"ZeroDivisionError: division by zero\n"
        )
        result = tool.run_code_in_docker(code, libraries_used)

        self.assertEqual(result, expected_output)
