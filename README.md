# CrewAI

## Virtual Env
```bash
poetry shell
```

## Running Tests
```bash
poetry run pytest
```

## Packaging
```bash
poetry build
```

### Installing Locally
```bash
pip install dist/*.tar.gz
```


# CrewAI

## Why?

## How?

## What is it?
Convention?

Convention of Roles
Convention over Tools
Convention over interactions (In what ways agents interact, how flexible, conversation patterns, where these happen [in a room, in isolation])
Convention over degree of guidance

You must be able to bring any tools <- Convention over tools

How does it compare to autogen?

Autogen is good to create conversational agents, these agents can then work 
together autonomously, but there is no concepts of process, a process would need 
to be programatically added.

ChatDev brings the idea of processes to the AI Agents but it's stiff and it's
customizations are not meant to be deployed at production settings

CrewAI is a python library that provide modules to build crews of AI Agents tha

What is the interface to interact with CrewAI?