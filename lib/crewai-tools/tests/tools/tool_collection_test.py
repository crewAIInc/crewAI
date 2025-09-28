import unittest
from unittest.mock import MagicMock

from crewai.tools import BaseTool
from crewai_tools.adapters.tool_collection import ToolCollection


class TestToolCollection(unittest.TestCase):
    def setUp(self):
        self.search_tool = self._create_mock_tool(
            "SearcH", "Search Tool"
        )  # Tool name is case sensitive
        self.calculator_tool = self._create_mock_tool("calculator", "Calculator Tool")
        self.translator_tool = self._create_mock_tool("translator", "Translator Tool")

        self.tools = ToolCollection(
            [self.search_tool, self.calculator_tool, self.translator_tool]
        )

    def _create_mock_tool(self, name, description):
        mock_tool = MagicMock(spec=BaseTool)
        mock_tool.name = name
        mock_tool.description = description
        return mock_tool

    def test_initialization(self):
        self.assertEqual(len(self.tools), 3)
        self.assertEqual(self.tools[0].name, "SearcH")
        self.assertEqual(self.tools[1].name, "calculator")
        self.assertEqual(self.tools[2].name, "translator")

    def test_empty_initialization(self):
        empty_collection = ToolCollection()
        self.assertEqual(len(empty_collection), 0)
        self.assertEqual(empty_collection._name_cache, {})

    def test_initialization_with_none(self):
        collection = ToolCollection(None)
        self.assertEqual(len(collection), 0)
        self.assertEqual(collection._name_cache, {})

    def test_access_by_index(self):
        self.assertEqual(self.tools[0], self.search_tool)
        self.assertEqual(self.tools[1], self.calculator_tool)
        self.assertEqual(self.tools[2], self.translator_tool)

    def test_access_by_name(self):
        self.assertEqual(self.tools["search"], self.search_tool)
        self.assertEqual(self.tools["calculator"], self.calculator_tool)
        self.assertEqual(self.tools["translator"], self.translator_tool)

    def test_key_error_for_invalid_name(self):
        with self.assertRaises(KeyError):
            _ = self.tools["nonexistent"]

    def test_index_error_for_invalid_index(self):
        with self.assertRaises(IndexError):
            _ = self.tools[10]

    def test_negative_index(self):
        self.assertEqual(self.tools[-1], self.translator_tool)
        self.assertEqual(self.tools[-2], self.calculator_tool)
        self.assertEqual(self.tools[-3], self.search_tool)

    def test_append(self):
        new_tool = self._create_mock_tool("new", "New Tool")
        self.tools.append(new_tool)

        self.assertEqual(len(self.tools), 4)
        self.assertEqual(self.tools[3], new_tool)
        self.assertEqual(self.tools["new"], new_tool)
        self.assertIn("new", self.tools._name_cache)

    def test_append_duplicate_name(self):
        duplicate_tool = self._create_mock_tool("search", "Duplicate Search Tool")
        self.tools.append(duplicate_tool)

        self.assertEqual(len(self.tools), 4)
        self.assertEqual(self.tools["search"], duplicate_tool)

    def test_extend(self):
        new_tools = [
            self._create_mock_tool("tool4", "Tool 4"),
            self._create_mock_tool("tool5", "Tool 5"),
        ]
        self.tools.extend(new_tools)

        self.assertEqual(len(self.tools), 5)
        self.assertEqual(self.tools["tool4"], new_tools[0])
        self.assertEqual(self.tools["tool5"], new_tools[1])
        self.assertIn("tool4", self.tools._name_cache)
        self.assertIn("tool5", self.tools._name_cache)

    def test_insert(self):
        new_tool = self._create_mock_tool("inserted", "Inserted Tool")
        self.tools.insert(1, new_tool)

        self.assertEqual(len(self.tools), 4)
        self.assertEqual(self.tools[1], new_tool)
        self.assertEqual(self.tools["inserted"], new_tool)
        self.assertIn("inserted", self.tools._name_cache)

    def test_remove(self):
        self.tools.remove(self.calculator_tool)

        self.assertEqual(len(self.tools), 2)
        with self.assertRaises(KeyError):
            _ = self.tools["calculator"]
        self.assertNotIn("calculator", self.tools._name_cache)

    def test_remove_nonexistent_tool(self):
        nonexistent_tool = self._create_mock_tool("nonexistent", "Nonexistent Tool")

        with self.assertRaises(ValueError):
            self.tools.remove(nonexistent_tool)

    def test_pop(self):
        popped = self.tools.pop(1)

        self.assertEqual(popped, self.calculator_tool)
        self.assertEqual(len(self.tools), 2)
        with self.assertRaises(KeyError):
            _ = self.tools["calculator"]
        self.assertNotIn("calculator", self.tools._name_cache)

    def test_pop_last(self):
        popped = self.tools.pop()

        self.assertEqual(popped, self.translator_tool)
        self.assertEqual(len(self.tools), 2)
        with self.assertRaises(KeyError):
            _ = self.tools["translator"]
        self.assertNotIn("translator", self.tools._name_cache)

    def test_clear(self):
        self.tools.clear()

        self.assertEqual(len(self.tools), 0)
        self.assertEqual(self.tools._name_cache, {})
        with self.assertRaises(KeyError):
            _ = self.tools["search"]

    def test_iteration(self):
        tools_list = list(self.tools)
        self.assertEqual(
            tools_list, [self.search_tool, self.calculator_tool, self.translator_tool]
        )

    def test_contains(self):
        self.assertIn(self.search_tool, self.tools)
        self.assertIn(self.calculator_tool, self.tools)
        self.assertIn(self.translator_tool, self.tools)

        nonexistent_tool = self._create_mock_tool("nonexistent", "Nonexistent Tool")
        self.assertNotIn(nonexistent_tool, self.tools)

    def test_slicing(self):
        slice_result = self.tools[1:3]
        self.assertEqual(len(slice_result), 2)
        self.assertEqual(slice_result[0], self.calculator_tool)
        self.assertEqual(slice_result[1], self.translator_tool)

        self.assertIsInstance(slice_result, list)
        self.assertNotIsInstance(slice_result, ToolCollection)

    def test_getitem_with_tool_name_as_int(self):
        numeric_name_tool = self._create_mock_tool("123", "Numeric Name Tool")
        self.tools.append(numeric_name_tool)

        self.assertEqual(self.tools["123"], numeric_name_tool)

        with self.assertRaises(IndexError):
            _ = self.tools[123]

    def test_filter_by_names(self):
        filtered = self.tools.filter_by_names(None)

        self.assertIsInstance(filtered, ToolCollection)
        self.assertEqual(len(filtered), 3)

        filtered = self.tools.filter_by_names(["search", "translator"])

        self.assertIsInstance(filtered, ToolCollection)
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0], self.search_tool)
        self.assertEqual(filtered[1], self.translator_tool)
        self.assertEqual(filtered["search"], self.search_tool)
        self.assertEqual(filtered["translator"], self.translator_tool)

        filtered = self.tools.filter_by_names(["search", "nonexistent"])

        self.assertIsInstance(filtered, ToolCollection)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0], self.search_tool)

        filtered = self.tools.filter_by_names(["nonexistent1", "nonexistent2"])

        self.assertIsInstance(filtered, ToolCollection)
        self.assertEqual(len(filtered), 0)

        filtered = self.tools.filter_by_names([])

        self.assertIsInstance(filtered, ToolCollection)
        self.assertEqual(len(filtered), 0)

    def test_filter_where(self):
        filtered = self.tools.filter_where(lambda tool: tool.name.startswith("S"))

        self.assertIsInstance(filtered, ToolCollection)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0], self.search_tool)
        self.assertEqual(filtered["search"], self.search_tool)

        filtered = self.tools.filter_where(lambda tool: True)

        self.assertIsInstance(filtered, ToolCollection)
        self.assertEqual(len(filtered), 3)
        self.assertEqual(filtered[0], self.search_tool)
        self.assertEqual(filtered[1], self.calculator_tool)
        self.assertEqual(filtered[2], self.translator_tool)

        filtered = self.tools.filter_where(lambda tool: False)

        self.assertIsInstance(filtered, ToolCollection)
        self.assertEqual(len(filtered), 0)
        filtered = self.tools.filter_where(lambda tool: len(tool.name) > 8)

        self.assertIsInstance(filtered, ToolCollection)
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0], self.calculator_tool)
        self.assertEqual(filtered[1], self.translator_tool)
