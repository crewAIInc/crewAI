import os
import tempfile
from unittest.mock import Mock, patch

from crewai_tools.rag.base_loader import LoaderResult
from crewai_tools.rag.loaders.mdx_loader import MDXLoader
from crewai_tools.rag.source_content import SourceContent
import pytest


class TestMDXLoader:
    def _write_temp_mdx(self, content):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".mdx", delete=False)
        f.write(content)
        f.close()
        return f.name

    def _load_from_file(self, content):
        path = self._write_temp_mdx(content)
        try:
            loader = MDXLoader()
            return loader.load(SourceContent(path)), path
        finally:
            os.unlink(path)

    def test_load_basic_mdx_file(self):
        content = """
import Component from './Component'
export const meta = { title: 'Test' }

# Test MDX File

This is a **markdown** file with JSX.

<Component prop="value" />

Some more content.

<div className="container">
    <p>Nested content</p>
</div>
"""
        result, path = self._load_from_file(content)

        assert isinstance(result, LoaderResult)
        assert all(
            tag not in result.content
            for tag in ["import", "export", "<Component", "<div", "</div>"]
        )
        assert all(
            text in result.content
            for text in [
                "# Test MDX File",
                "markdown",
                "Some more content",
                "Nested content",
            ]
        )
        assert result.metadata["format"] == "mdx"
        assert result.source == path

    def test_mdx_multiple_imports_exports(self):
        content = """
import React from 'react'
import { useState } from 'react'
import CustomComponent from './custom'

export default function Layout() { return null }
export const config = { test: true }

# Content

Regular markdown content here.
"""
        result, _ = self._load_from_file(content)
        assert "# Content" in result.content
        assert "Regular markdown content here." in result.content
        assert "import" not in result.content and "export" not in result.content

    def test_complex_jsx_cleanup(self):
        content = """
# MDX with Complex JSX

<div className="alert alert-info">
    <strong>Info:</strong> This is important information.
    <ul><li>Item 1</li><li>Item 2</li></ul>
</div>

Regular paragraph text.

<MyComponent prop1="value1">Nested content inside component</MyComponent>
"""
        result, _ = self._load_from_file(content)
        assert all(
            tag not in result.content
            for tag in ["<div", "<strong>", "<ul>", "<MyComponent"]
        )
        assert all(
            text in result.content
            for text in [
                "Info:",
                "Item 1",
                "Regular paragraph text.",
                "Nested content inside component",
            ]
        )

    def test_whitespace_cleanup(self):
        content = """


# Title


Some content.


More content after multiple newlines.



Final content.
"""
        result, _ = self._load_from_file(content)
        assert result.content.count("\n\n\n") == 0
        assert result.content.startswith("# Title")
        assert result.content.endswith("Final content.")

    def test_only_jsx_content(self):
        content = """
<div>
    <h1>Only JSX content</h1>
    <p>No markdown here</p>
</div>
"""
        result, _ = self._load_from_file(content)
        assert all(tag not in result.content for tag in ["<div>", "<h1>", "<p>"])
        assert "Only JSX content" in result.content
        assert "No markdown here" in result.content

    @patch("crewai_tools.security.safe_requests._raw_get")
    def test_load_mdx_from_url(self, mock_get):
        mock_get.return_value = Mock(
            text="# MDX from URL\n\nContent here.\n\n<Component />",
            raise_for_status=lambda: None,
        )
        loader = MDXLoader()
        result = loader.load(SourceContent("https://example.com/content.mdx"))
        assert "# MDX from URL" in result.content
        assert "<Component />" not in result.content

    @patch("crewai_tools.security.safe_requests._raw_get")
    def test_load_mdx_with_custom_headers(self, mock_get):
        mock_get.return_value = Mock(
            text="# Custom headers test", raise_for_status=lambda: None
        )
        loader = MDXLoader()
        loader.load(
            SourceContent("https://example.com"),
            headers={"Authorization": "Bearer token"},
        )
        assert mock_get.call_args[1]["headers"] == {"Authorization": "Bearer token"}

    @patch("crewai_tools.security.safe_requests._raw_get")
    def test_mdx_url_fetch_error(self, mock_get):
        mock_get.side_effect = Exception("Network error")
        with pytest.raises(ValueError, match="Error fetching content from URL https://example.com: Network error"):
            MDXLoader().load(SourceContent("https://example.com"))

    def test_load_inline_mdx_text(self):
        content = """# Inline MDX\n\nimport Something from 'somewhere'\n\nContent with <Component prop=\"value\" />.\n\nexport const meta = { title: 'Test' }"""
        loader = MDXLoader()
        result = loader.load(SourceContent(content))
        assert "# Inline MDX" in result.content
        assert "Content with ." in result.content

    def test_empty_result_after_cleaning(self):
        content = """
import Something from 'somewhere'
export const config = {}
<div></div>
"""
        result, _ = self._load_from_file(content)
        assert result.content.strip() == ""

    def test_edge_case_parsing(self):
        content = """
# Title

<Component>
Multi-line
JSX content
</Component>

import { a, b } from 'module'

export { x, y }

Final text.
"""
        result, _ = self._load_from_file(content)
        assert "# Title" in result.content
        assert "JSX content" in result.content
        assert "Final text." in result.content
        assert all(
            phrase not in result.content
            for phrase in ["import {", "export {", "<Component>"]
        )

    def test_fenced_code_keeps_imports(self):
        content = "```python\nimport json\nprint(json.loads('{}'))\n```"
        result = MDXLoader().load(SourceContent(content))
        assert result.content == "```python\nimport json\nprint(json.loads('{}'))\n```"

    def test_fenced_code_keeps_comparisons(self):
        content = "```text\nassert 0 < count and count > 2\n```"
        result = MDXLoader().load(SourceContent(content))
        assert "assert 0 < count and count > 2" in result.content

    def test_inline_code_keeps_jsx_like_text(self):
        content = "Use `List<T>` in the schema."
        result = MDXLoader().load(SourceContent(content))
        assert result.content == "Use `List<T>` in the schema."

    def test_inline_code_keeps_html_like_text(self):
        content = "The `<div>` tag renders a block."
        result = MDXLoader().load(SourceContent(content))
        assert "`<div>`" in result.content

    def test_tilde_fence_keeps_code(self):
        content = "~~~sh\nexport FOO=bar\n~~~"
        result = MDXLoader().load(SourceContent(content))
        assert result.content == "~~~sh\nexport FOO=bar\n~~~"

    def test_long_fence_keeps_code(self):
        content = "````\nimport os\n````"
        result = MDXLoader().load(SourceContent(content))
        assert result.content == "````\nimport os\n````"

    def test_fenced_code_blank_lines_are_preserved(self):
        content = "```python\nx = 1\n\n\n\ny = 2\n```"
        result = MDXLoader().load(SourceContent(content))
        assert result.content == "```python\nx = 1\n\n\n\ny = 2\n```"

    def test_mdx_syntax_outside_code_is_still_removed(self):
        content = "import A from 'a'\n\ntext <B />\n\nexport const x = 1"
        result = MDXLoader().load(SourceContent(content))
        assert result.content.strip() == "text"

    def test_jsx_attribute_backticks_are_not_inline_code(self):
        content = "Attr <Foo bar={`x`} /> here"
        result = MDXLoader().load(SourceContent(content))
        assert "<Foo" not in result.content
        assert result.content.strip() == "Attr  here"

    def test_fenced_block_with_surrounding_mdx(self):
        content = (
            "import A from 'a'\n\n"
            "```python\nimport json\n```\n\n"
            "<B />\n\n"
            "export const x = 1"
        )
        result = MDXLoader().load(SourceContent(content))
        assert "import A" not in result.content
        assert "export const" not in result.content
        assert "<B" not in result.content
        assert "```python\nimport json\n```" in result.content
