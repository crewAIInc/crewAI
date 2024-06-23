from typing import Any, List, Optional, Union

from crewai.agents.third_party_agents.agent_executor_mixin import CrewAgentExecutorMixin

from llama_index_client import ChatMessage
from llama_index.core.chat_engine.types import (
    AGENT_CHAT_RESPONSE_TYPE,
    ChatResponseMode,
)
from crewai.utilities import I18N
from llama_index.core.instrumentation.events.agent import (
    AgentRunStepEndEvent,
    AgentRunStepStartEvent,
    AgentChatWithStepStartEvent,
    AgentChatWithStepEndEvent,
)
from llama_index.core.agent.types import TaskStep, TaskStepOutput
from llama_index.core.agent import ReActAgent

import llama_index.core.instrumentation as instrument

dispatcher = instrument.get_dispatcher(__name__)


class CrewLlamaReActAgentExecutor(ReActAgent, CrewAgentExecutorMixin):
    _i18n: I18N = I18N()
    should_ask_for_human_input: bool = False
    llm: Any = None
    iterations: int = 0
    task: Any = None
    crew_agent: Any = None
    crew: Any = None
    function_calling_llm: Any = None
    step_callback: Optional[Any] = None
    max_iterations: Optional[int] = 15
    force_answer_max_iterations: Optional[int] = None

    @dispatcher.span
    def _chat(
        self,
        message: str,
        chat_history: Optional[List[ChatMessage]] = None,
        tool_choice: Union[str, dict] = "auto",
        mode: ChatResponseMode = ChatResponseMode.WAIT,
    ) -> AGENT_CHAT_RESPONSE_TYPE:
        """Chat with step executor."""
        if chat_history is not None:
            self.memory.set(chat_history)
        task = self.create_task(message)

        result_output = None
        dispatcher.event(AgentChatWithStepStartEvent(user_msg=message))

        if self.task.human_input:
            self.should_ask_for_human_input = True

        self.iterations = 0

        # TODO request_within_rpm_limit
        while True:
            # pass step queue in as argument, assume step executor is stateless
            cur_step_output = self._run_step(
                task.task_id, mode=mode, tool_choice=tool_choice
            )
            if cur_step_output.is_last:
                result_output = cur_step_output

                if self.should_ask_for_human_input:
                    chat_response = self._ask_human_input(result_output)
                    task_made = self.create_task(chat_response)
                    self._run_step(
                        task_made.task_id, mode=mode, tool_choice=tool_choice
                    )
                break

            # ensure tool_choice does not cause endless loops
            tool_choice = "auto"
            self.iterations += 1

        result = self.finalize_response(
            task.task_id,
            result_output,
        )
        dispatcher.event(AgentChatWithStepEndEvent(response=result))
        return result

    def _run_step(
        self,
        task_id: str,
        step: Optional[TaskStep] = None,
        input: Optional[str] = None,
        mode: ChatResponseMode = ChatResponseMode.WAIT,
        **kwargs: Any,
    ) -> TaskStepOutput:
        """Execute step."""
        if self._should_force_answer():
            print("force final answer")
            error = self._i18n.errors("force_final_answer")
            self.have_forced_answer = True
            force_final_answer_task = self.create_task(error)
            AgentRunStepStartEvent(
                task_id=force_final_answer_task.task_id, step=step, input=error
            )
            return
        dispatcher.event(
            AgentRunStepStartEvent(task_id=task_id, step=step, input=input)
        )
        task = self.state.get_task(task_id)
        step_queue = self.state.get_step_queue(task_id)
        step = step or step_queue.popleft()
        # TODO: handle steps here
        if input is not None:
            step.input = input

        if self.verbose:
            print(f"> Running step {step.step_id}. Step input: {step.input}")

        # TODO: figure out if you can dynamically swap in different step executors
        # not clear when you would do that by theoretically possible

        if mode == ChatResponseMode.WAIT:
            cur_step_output = self.agent_worker.run_step(step, task, **kwargs)
        elif mode == ChatResponseMode.STREAM:
            cur_step_output = self.agent_worker.stream_step(step, task, **kwargs)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        # append cur_step_output next steps to queue
        next_steps = cur_step_output.next_steps
        step_queue.extend(next_steps)

        # add cur_step_output to completed steps
        completed_steps = self.state.get_completed_steps(task_id)
        completed_steps.append(cur_step_output)
        if self.step_callback:
            self.step_callback(cur_step_output)

        dispatcher.event(AgentRunStepEndEvent(step_output=cur_step_output))
        return cur_step_output
