import unittest
from pydantic import BaseModel
from typing import List, Optional

from crewai.utilities.pydantic_schema_parser import PydanticSchemaParser


# Define test models
class InnerModel(BaseModel):
    inner_field: int


class OuterModel(BaseModel):
    simple_field: str
    list_field: List[int]
    optional_field: Optional[str]
    nested_model: InnerModel


# Test cases
class TestPydanticSchemaParser(unittest.TestCase):
    def test_simple_model(self):
        class SimpleModel(BaseModel):
            field1: int
            field2: str

        parser = PydanticSchemaParser(model=SimpleModel)
        expected_schema = """
{
    field1: int,
    field2: str
}""".strip()
        self.assertEqual(parser.get_schema(), expected_schema)

    def test_model_with_list(self):
        class ListModel(BaseModel):
            field1: List[int]

        parser = PydanticSchemaParser(model=ListModel)
        expected_schema = """
{
    field1: List[int]
}""".strip()
        self.assertEqual(parser.get_schema(), expected_schema)

    def test_model_with_optional(self):
        class OptionalModel(BaseModel):
            field1: Optional[int]

        parser = PydanticSchemaParser(model=OptionalModel)
        expected_schema = """
{
    field1: Optional[int]
}""".strip()
        self.assertEqual(parser.get_schema(), expected_schema)

    def test_nested_model(self):
        parser = PydanticSchemaParser(model=OuterModel)
        expected_schema = """
{
    simple_field: str,
    list_field: List[int],
    optional_field: Optional[str],
    nested_model: InnerModel
    {
        inner_field: int
    }
}""".strip()
        self.assertEqual(parser.get_schema(), expected_schema)


if __name__ == "__main__":
    unittest.main()
