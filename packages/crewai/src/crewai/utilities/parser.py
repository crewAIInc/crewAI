import re


class YamlParser:
    @staticmethod
    def parse(file):
        """
        Parses a YAML file, modifies specific patterns, and checks for unsupported 'context' usage.
        Args:
            file (file object): The YAML file to parse.
        Returns:
            str: The modified content of the YAML file.
        Raises:
            ValueError: If 'context:' is used incorrectly.
        """
        content = file.read()

        # Replace single { and } with doubled ones, while leaving already doubled ones intact and the other special characters {# and {%
        modified_content = re.sub(r"(?<!\{){(?!\{)(?!\#)(?!\%)", "{{", content)
        modified_content = re.sub(
            r"(?<!\})(?<!\%)(?<!\#)\}(?!})", "}}", modified_content
        )

        # Check for 'context:' not followed by '[' and raise an error
        if re.search(r"context:(?!\s*\[)", modified_content):
            raise ValueError(
                "Context is currently only supported in code when creating a task. "
                "Please use the 'context' key in the task configuration."
            )

        return modified_content
