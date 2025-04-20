import datetime
import pytest
from typing import Any, Dict, List

from pydantic import BaseModel

from crewai.utilities.jinja_templating import render_template, to_jinja_template

class Person(BaseModel):
    name: str
    age: int
    
    def __str__(self):
        return f"{self.name} ({self.age})"

class TestJinjaTemplating:
    def test_to_jinja_template(self):
        assert to_jinja_template("Hello {name}!") == "Hello {{name}}!"
        
        assert to_jinja_template("Hello {{name}}!") == "Hello {{name}}!"
        
        assert to_jinja_template("Hello {name} and {{title}}!") == "Hello {{name}} and {{title}}!"
        
        assert to_jinja_template("") == ""
        
        assert to_jinja_template("Hello world!") == "Hello world!"
    
    def test_render_template_simple_types(self):
        inputs = {"name": "John", "age": 30, "active": True, "height": 1.85}
        
        assert render_template("Hello {name}!", inputs) == "Hello John!"
        assert render_template("Age: {age}", inputs) == "Age: 30"
        assert render_template("Active: {active}", inputs) == "Active: True"
        assert render_template("Height: {height}", inputs) == "Height: 1.85"
        
        assert render_template("{name} is {age} years old", inputs) == "John is 30 years old"
    
    def test_render_template_container_types(self):
        inputs = {
            "items": ["apple", "banana", "orange"],
            "person": {"name": "John", "age": 30}
        }
        
        assert render_template("First item: {{items[0]}}", inputs) == "First item: apple"
        
        assert render_template("Person name: {{person.name}}", inputs) == "Person name: John"
        
        assert render_template(
            "Items: {% for item in items %}{{item}}{% if not loop.last %}, {% endif %}{% endfor %}",
            inputs
        ) == "Items: apple, banana, orange"
        
        assert render_template(
            "{% if items|length > 2 %}Many items{% else %}Few items{% endif %}",
            inputs
        ) == "Many items"
    
    def test_render_template_datetime(self):
        today = datetime.datetime.now()
        inputs = {"today": today}
        
        assert render_template("Today: {{today|date}}", inputs) == f"Today: {today.strftime('%Y-%m-%d')}"
        
        assert render_template("Today: {{today|date('%d/%m/%Y')}}", inputs) == f"Today: {today.strftime('%d/%m/%Y')}"
    
    def test_render_template_custom_objects(self):
        person = Person(name="John", age=30)
        inputs = {"person": person}
        
        assert render_template("Person: {person}", inputs) == "Person: John (30)"
        
        assert render_template("Person name: {{person.name}}", inputs) == "Person name: John"
    
    def test_render_template_error_handling(self):
        inputs = {"name": "John"}
        
        with pytest.raises(KeyError) as excinfo:
            render_template("Hello {age}!", inputs)
        assert "Template variable 'age' not found" in str(excinfo.value)
        
        with pytest.raises(ValueError) as excinfo:
            render_template("Hello {name}!", {})
        assert "Inputs dictionary cannot be empty" in str(excinfo.value)
