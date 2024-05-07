import re


class YamlParser:
    def parse(file):
        content = file.read()
        # Replace single { and } with doubled ones, while leaving already doubled ones intact and the other special characters {# and {%
        modified_content = re.sub(r"(?<!\{){(?!\{)(?!\#)(?!\%)", "{{", content)
        modified_content = re.sub(
            r"(?<!\})(?<!\%)(?<!\#)\}(?!})", "}}", modified_content
        )
        return modified_content
