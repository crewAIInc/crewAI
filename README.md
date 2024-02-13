## Getting started

When setting up agents you can provide tools for them to use. Here you will find ready-to-use tools as well as simple helpers for you to create your own tools.

In order to create a new tool, you have to pick one of the available strategies.

### Subclassing `BaseTool`

```python
class MyTool(BaseTool):
    name: str = "Knowledge base"
    description: str = "A knowledge base with all the requirements for the project."

    def _run(self, question) -> str:
        return (
            tbl.search(embed_func([question])[0]).limit(3).to_pandas()["text"].tolist()
        )
```

As you can see, all you need to do is to create a new class that inherits from `BaseTool`, define `name` and `description` fields, as well as implement the `_run` method.

### Create tool from a function or lambda

```python
my_tool = Tool(
    name="Knowledge base",
    description="A knowledge base with all the requirements for the project.",
    func=lambda question: tbl.search(embed_func([question])[0])
    .limit(3)
    .to_pandas()["text"]
    .tolist(),
)
```

Here's it's a bit simpler, as you don't have to subclass. Simply create a `Tool` object with the three required fields and you are good to go.

### Use the `tool` decorator.

```python
@tool("Knowledge base")
def my_tool(question: str) -> str:
    """A knowledge base with all the requirements for the project."""
    return tbl.search(embed_func([question])[0]).limit(3).to_pandas()["text"].tolist()
```

By using the decorator you can easily wrap simple functions as tools. If you don't provide a name, the function name is going to be used. However, the docstring is required.

If you are using a linter you may see issues when passing your decorated tool in `tools` parameters that expect a list of `BaseTool`. If that's the case, you can use the `as_tool` helper.


## Contribution

This repo is open-source and we welcome contributions. If you're looking to contribute, please:

- Fork the repository.
- Create a new branch for your feature.
- Add your feature or improvement.
- Send a pull request.
- We appreciate your input!

### Installing Dependencies

```bash
poetry install
```

### Virtual Env

```bash
poetry shell
```

### Pre-commit hooks

```bash
pre-commit install
```

### Running Tests

```bash
poetry run pytest
```

### Running static type checks

```bash
poetry run pyright
```

### Packaging

```bash
poetry build
```

### Installing Locally

```bash
pip install dist/*.tar.gz
```
