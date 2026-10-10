// End-to-end check that CrewAI runs on Pyodide: loads the wheels built by
// build.py, then runs a crew (with a tool call), streaming, and a flow against
// a local mock of the OpenAI chat completions API.
//
//   npm install && npm run build && npm run test:smoke
//
// Set PYODIDE_INDEX_URL to load Pyodide's packages from a local distribution.

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createServer } from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadPyodide } from "pyodide";

import { loadCrewAI } from "../js/crewai-pyodide.mjs";

const WHEELS = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "dist", "wheels");

function completion(model, message) {
  return {
    id: "chatcmpl-test",
    object: "chat.completion",
    created: 0,
    model,
    choices: [{ index: 0, message, finish_reason: message.tool_calls ? "tool_calls" : "stop" }],
    usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
  };
}

function reply(request) {
  const last = request.messages.at(-1);
  if (request.tools && last.role !== "tool") {
    const name = request.tools[0].function.name;
    return {
      role: "assistant",
      content: null,
      tool_calls: [
        { id: "call_1", type: "function", function: { name, arguments: '{"a": 2, "b": 3}' } },
      ],
    };
  }
  if (last.role === "tool") return { role: "assistant", content: `The sum is ${last.content}` };
  return { role: "assistant", content: "Thought: done\nFinal Answer: Hello from WebAssembly" };
}

const server = createServer(async (req, res) => {
  if (req.method === "GET") {
    try {
      res.end(await readFile(path.join(WHEELS, path.basename(req.url))));
    } catch {
      res.writeHead(404).end();
    }
    return;
  }
  let body = "";
  for await (const chunk of req) body += chunk;
  const request = JSON.parse(body);
  const message = reply(request);
  if (!request.stream) {
    res.setHeader("content-type", "application/json");
    res.end(JSON.stringify(completion(request.model, message)));
    return;
  }
  res.setHeader("content-type", "text/event-stream");
  const chunk = (delta, finish_reason = null) =>
    `data: ${JSON.stringify({
      id: "chatcmpl-test",
      object: "chat.completion.chunk",
      created: 0,
      model: request.model,
      choices: [{ index: 0, delta, finish_reason }],
    })}\n\n`;
  const half = Math.floor(message.content.length / 2);
  res.write(chunk({ role: "assistant", content: message.content.slice(0, half) }));
  res.write(chunk({ content: message.content.slice(half) }));
  res.end(chunk({}, "stop") + "data: [DONE]\n\n");
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const baseURL = `http://127.0.0.1:${server.address().port}`;

try {
  const pyodide = await loadCrewAI({
    pyodide: await loadPyodide({ indexURL: process.env.PYODIDE_INDEX_URL }),
    wheelBaseURL: `${baseURL}/`,
  });
  pyodide.globals.set("base_url", `${baseURL}/v1`);
  const results = await pyodide.runPythonAsync(
    `
import crewai_pyodide
from crewai import LLM, Agent, Crew, Task
from crewai.flow.flow import Flow, listen, start
from crewai.tools import tool


@tool("add")
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def crew(stream=False, tools=()):
    llm = LLM(model="gpt-4o-mini", base_url=base_url, api_key="sk-test", stream=stream)
    agent = Agent(role="r", goal="g", backstory="b", llm=llm, tools=list(tools))
    task = Task(description="d", expected_output="o", agent=agent)
    return Crew(agents=[agent], tasks=[task])


class GreetingFlow(Flow):
    @start()
    def begin(self):
        return "hi"

    @listen(begin)
    async def greet(self, _):
        llm = LLM(model="gpt-4o-mini", base_url=base_url, api_key="sk-test")
        return await llm.acall("greet")


[
    (await crewai_pyodide.run(crew().akickoff())).raw,
    (await crewai_pyodide.run(crew(stream=True).akickoff())).raw,
    (await crewai_pyodide.run(crew(tools=[add]).akickoff())).raw,
    await crewai_pyodide.run(GreetingFlow().kickoff_async()),
]
`,
  );
  const [plain, streamed, withTool, flow] = results.toJs();
  assert.equal(plain, "Hello from WebAssembly");
  assert.equal(streamed, "Hello from WebAssembly");
  assert.equal(withTool, "The sum is 5");
  assert.match(flow, /Hello from WebAssembly/);
  console.log("crewai-pyodide smoke test passed");
} finally {
  server.close();
}
