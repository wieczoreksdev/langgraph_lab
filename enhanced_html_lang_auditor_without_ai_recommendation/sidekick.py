import uuid
import os
import asyncio
from datetime import datetime
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from typing import List, Any, Optional, Dict
from pydantic import BaseModel, Field
from sidekick_tools import setup_playwright, all_tools

load_dotenv(override=True)

class State(TypedDict):
    messages: Annotated[List[Any], add_messages]
    success_criteria: str
    feedback: Optional[str]
    success_criteria_met: bool
    user_input_needed: bool
    input_csv_file: str
    output_folder_path: str
    delegated_to: str
    worker_iteration: int
    language_worker_iteration: int 
    output_file_path: str

class ManagerOutput(BaseModel):
    delegated_to: str = Field(description="LLM name to which task is directed to, possible options: language_worker, worker")

# class LanguageWorkerOutput(BaseModel):
#     output_file_path: str = Field(description="Output file path which is returned from language tools")

class EvaluatorOutput(BaseModel):
    feedback: str = Field(description="Feedback on the assistant's response")
    success_criteria_met: bool = Field(description="Whether the success criteria have been met")
    user_input_needed: bool = Field(
        description="True if more input is needed from the user, or clarifications, or the assistant is stuck"
    )


class Sidekick:
    def __init__(self):
        self.worker_llm_with_tools = None
        self.language_worker_llm_with_tools = None
        self.evaluator_llm_with_output = None
        self.manager_llm_with_structured_output = None
        self.tools = None
        self.tools_indexed = None
        self.graph = None
        self.sidekick_id = str(uuid.uuid4())
        self.memory = MemorySaver()
        self.browser = None
       
    async def setup(self):
        self.tools, self.browser = await setup_playwright()
        self.tools += await all_tools()
        self.tools_indexed = {tool.name: tool for tool in self.tools}
        manager_llm = ChatOpenAI(model="gpt-4o-mini")
        self.manager_llm_with_structured_output = manager_llm.with_structured_output(ManagerOutput)
        worker_llm = ChatOpenAI(model="gpt-4o-mini")
        self.worker_llm_with_tools = worker_llm.bind_tools(self.tools)
        language_worker_llm = ChatOpenAI(model="gpt-4o-mini")
        self.language_worker_llm_with_tools = language_worker_llm.bind_tools(self.tools)
        evaluator_llm = ChatOpenAI(model="gpt-4o-mini")
        self.evaluator_llm_with_output = evaluator_llm.with_structured_output(EvaluatorOutput)
        await self.build_graph()

    def manager(self, state: State) -> State:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        system_message = f"""
        You are a helpful assistant that based on provided by user success criteria: {state["success_criteria"]} and message: {self.format_conversation(state["messages"])},
        you are able to determine what kind of llm to assign the task to. 
        You have two choices:
        - language_worker dedicated to detect page language specializing in using tool: {self.tools_indexed['page_langcodes_auditor']} 
            and tool: {self.tools_indexed['page_language_tags_auditor']}
        - regular worker assigned with the rest of tasks that can use all tools: {", ".join(list(self.tools_indexed.keys()))}
        Current time: {current_time}
        As a response provide: language_worker or worker
        """
        messages = state["messages"]
        messages = [m for m in messages if not isinstance(m, SystemMessage)]
        messages.insert(0, SystemMessage(content=system_message))
        # Invoke the LLM with tools
        manager_response = self.manager_llm_with_structured_output.invoke(messages)
        # Return updated state
        return  {
            "delegated_to": manager_response.delegated_to
        }

    def language_worker(self, state: State) -> State:
        # Dict[str, Any]
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        system_message = f"""
            You are a helpful assistant that can use tools to complete tasks
            Keep working until you either need clarification from the user or the success criteria is met.
            You have access to tools, including browsing the internet and running Python code (use print() to produce output).
            Here is a full list {", ".join(list(self.tools_indexed.keys()))}
            You MUST use the 'page_langcodes_auditor' {self.tools_indexed['page_langcodes_auditor']} 
            or alternatively 'page_language_tags_auditor': {self.tools_indexed['page_language_tags_auditor']} on the CSV file located at {state["input_csv_file"]}.
            Use the tool page_langcodes_auditor with arguments:
            - path: {state["input_csv_file"]}
            - output_folder_path: {state["output_folder_path"]}
            Save the output to the file path returned by either {self.tools_indexed['page_langcodes_auditor']} or {self.tools_indexed['page_language_tags_auditor']} tool in {state["output_folder_path"]}.
            Do NOT respond without creating the CSV if the success criteria involve CSV analysis.
            Current time: {current_time}
            Success criteria: {state["success_criteria"]}
            Reply either with a question for the user (clearly labeled 'Question: ...') or the final answer. Do not ask a question if you have finished.
            """
        if state.get("feedback"):
            system_message += f"""
                Previously your response did not meet the success criteria. Feedback:
                {state['feedback']}
                Use this feedback to improve your next response.
                """
        if state["input_csv_file"]:
            exists = os.path.exists(state["input_csv_file"])
            system_message += f"""
                Optional input file path: {state["input_csv_file"]}
                File exists: {exists}. Use this file as input for analysis.
                """
        if state.get("input_csv_file") and os.path.exists(state["input_csv_file"]):
            try:
                csv_snippet = self.tools_indexed['read_csv_snippet'](state["input_csv_file"])
                system_message += f"\nInput CSV snippet for context:\n{csv_snippet}\n"
            except Exception as e:
                system_message += f"\n[Failed to read CSV snippet: {e}]\n"
        
        messages = state["messages"]
        messages = [m for m in messages if not isinstance(m, SystemMessage)]
        messages.insert(0, SystemMessage(content=system_message))
        # Invoke the LLM with tools
        response = self.language_worker_llm_with_tools.invoke(messages)
        # Return updated state
        return {
            "messages": [response],
            "language_worker_iteration": state["language_worker_iteration"] + 1, 
            "delegated_to": "language_worker"
        }

    def worker(self, state: State) -> Dict[str, Any]:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        system_message = f"""
            You are a helpful assistant that can use tools to complete tasks.
            Keep working until you either need clarification from the user or the success criteria is met.
            You have access to tools, including browsing the internet and running Python code (use print() to produce output).
            Here is a full list {", ".join(list(self.tools_indexed.keys()))}
            Current time: {current_time}
            Success criteria: {state["success_criteria"]}
            Reply either with a question for the user (clearly labeled 'Question: ...') or the final answer. 
            Do not ask a question if you have finished.
            Do not ask questions if you did not fulfil success criteria.Check well success criteria and fullfill them
            As a response if you need to create csv file upload it in the following folder {state["output_folder_path"]} and name it result.csv.
            """
        if state.get("feedback"):
            system_message += f"""
                Previously your response did not meet the success criteria. Feedback:
                {state['feedback']}
                Use this feedback to improve your next response.
                """
        if state["input_csv_file"]:
            exists = os.path.exists(state["input_csv_file"])
            system_message += f"""
                Optional input file path: {state["input_csv_file"]}
                File exists: {exists}. Use this file as input for analysis.
                """
        if state.get("input_csv_file") and os.path.exists(state["input_csv_file"]):
            try:
                csv_snippet = self.tools_indexed['read_csv_snippet'](state["input_csv_file"])
                system_message += f"\nInput CSV snippet for context:\n{csv_snippet}\n"
            except Exception as e:
                system_message += f"\n[Failed to read CSV snippet: {e}]\n"
        messages = state["messages"]
        messages = [m for m in messages if not isinstance(m, SystemMessage)]
        messages.insert(0, SystemMessage(content=system_message))
        # Invoke the LLM with tools
        response = self.worker_llm_with_tools.invoke(messages)
        # Return updated state
        return {
            "messages": [response],
            "worker_iteration": state["worker_iteration"] + 1,
            "delegated_to": "worker"
        }

    def workers_router(self, state: State) -> str:
        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", None)
        if tool_calls:
            return "tools"
        else:
            return "evaluator"


    def capture_tool_output(self, state: State) -> State:
        for msg in reversed(state["messages"]):
            if not isinstance(msg, ToolMessage):
                continue
            content = msg.content
            # Handle dict payloads
            if isinstance(content, dict):
                output_path = content.get("output_file_path") or content.get("path")
            else:
                output_path = str(content).strip()
            if (
                output_path
                and output_path.endswith(".csv")
                and os.path.exists(output_path)
                and os.path.getsize(output_path) > 0
            ):
                return {"output_file_path": output_path}
        return {}

    def manager_router(self, state: State) -> str:
        if state["delegated_to"] == 'worker':
            return 'worker'
        else:
            return 'language_worker'

    def format_conversation(self, messages: List[Any]) -> str:
        conversation = "Conversation history:\n\n"
        for message in messages:
            if isinstance(message, HumanMessage):
                conversation += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                text = message.content or "[Tools use]"
                conversation += f"Assistant: {text}\n"
        return conversation


    def evaluator(self, state: State) -> State:
        system_message = """
            You are an evaluator assessing whether an Assistant has completed a task successfully.
            Provide:
            - Feedback on the assistant's last response
            - Whether success criteria are met (True/False)
            - Whether more input is needed from the user (True/False)
            """
        last_ai = next(
            m for m in reversed(state["messages"])
            if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None)
        )
        user_message = f"""
            Conversation history: {self.format_conversation(state['messages'])}
            Success criteria: {state['success_criteria']}
            Assistant's last response: {last_ai}
            Check optional input CSV: {state['input_csv_file']}
            Check output CSV in folder: {state['output_folder_path']}
            If prior feedback exists:
            {state.get('feedback', 'None')}
            """
        evaluator_messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_message),
        ]
        eval_result = self.evaluator_llm_with_output.invoke(evaluator_messages)
        return  {
            "feedback": eval_result.feedback,
            "success_criteria_met": eval_result.success_criteria_met,
            "user_input_needed": eval_result.user_input_needed,
        }

    def tool_router(self, state: State) -> str:
        """
        After tools + reducer, return control to the active worker
        """
        return state.get("delegated_to") or "worker"

    def route_based_on_evaluation(self, state: State) -> str:
    # If task is done or user input is needed, end
        if state["success_criteria_met"] or state["user_input_needed"]:
            state["delegated_to"] = ""
            return "END"
        # Check which worker was last active
        last_worker = state.get("delegated_to", "worker")
        # Terminate only if the active worker exceeded max iterations
        if last_worker == "worker" and state["worker_iteration"] >= 5:
            return "END"
        if last_worker == "language_worker" and state["language_worker_iteration"] >= 5:
            return "END"
        # Otherwise, continue with the same worker
        return last_worker

    async def build_graph(self):
        # Set up Graph Builder with State
        graph_builder = StateGraph(State)
        # Add nodes
        # add additinaol 2 llms 1 triager, 2 url_specific worker
        graph_builder.add_node("manager", self.manager)
        graph_builder.add_node("worker", self.worker)
        graph_builder.add_node("language_worker", self.language_worker)
        graph_builder.add_node("tools", ToolNode(tools=self.tools))
        graph_builder.add_node("capture_tool_output", self.capture_tool_output)
        graph_builder.add_node("evaluator", self.evaluator)
        
        # Add edges
        graph_builder.add_edge(START, "manager")
        graph_builder.add_conditional_edges(
            "manager", self.manager_router, {"worker": "worker", "language_worker": "language_worker"}
        )
        graph_builder.add_conditional_edges(
            "worker", self.workers_router, {"tools": "tools", "evaluator": "evaluator"}
        )
        graph_builder.add_conditional_edges(
            "language_worker", self.workers_router, {"tools": "tools", "evaluator": "evaluator"}
        )
        graph_builder.add_edge("tools", "capture_tool_output")
        graph_builder.add_conditional_edges(
            "capture_tool_output",
            self.tool_router,
            {"worker": "worker", "language_worker": "language_worker"}
        )
        # graph_builder.add_conditional_edges(
        #     "tools", self.tool_router, {"worker": "worker", "language_worker": "language_worker" }
        # )
        graph_builder.add_conditional_edges(
            "evaluator", self.route_based_on_evaluation, {"worker": "worker","language_worker":"language_worker", "END": END }
        )
        # Compile the graph
        self.graph = graph_builder.compile(checkpointer=self.memory)

    async def run_superstep(self, message, success_criteria, history, input_csv_file, output_folder):
        config = {"configurable": {"thread_id": self.sidekick_id}}
        print(f'OUTPUT FOLDER: {output_folder}')
        if isinstance(message, str):
            messages = [HumanMessage(content=message)]
        else:
            messages = message
        
        state = {
            "messages": messages,
            "success_criteria": success_criteria or "The answer should be clear and accurate",
            "feedback": None,
            "success_criteria_met": False,
            "user_input_needed": False,
            "input_csv_file": input_csv_file,
            "output_folder_path": output_folder,
            "delegated_to": "",
            "worker_iteration": 0,
            "language_worker_iteration": 0,
            "output_file_path": ""
        }
        state = await self.graph.ainvoke(state, config=config)
        user = {"role": "user", "content": message}
        last_ai = next(
            m for m in reversed(state["messages"])
            if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None)
        )
        reply = {
            "role": "assistant",
            "content": last_ai.content
        }
        feedback = {
            "role": "assistant",
            "content": state.get("feedback")
        }
        output_file_path = state.get("output_file_path") 
        #os.path.join(output_folder, "result.csv")
        if not os.path.exists(output_file_path) or os.path.getsize(output_file_path) == 0:
            output_file_path = None
        return history + [user, reply, feedback], output_file_path

    def cleanup(self):
        if self.browser:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.browser.close())
                # if self.playwright:
                #     loop.create_task(self.playwright.stop())
            except RuntimeError:
                # If no loop is running, do a direct run
                asyncio.run(self.browser.close())
                # if self.playwright:
                #     asyncio.run(self.playwright.stop())
if __name__ == "__main__":
    async def main():
        sidekick = Sidekick()
        await sidekick.setup()

        result = await sidekick.run_superstep(
            "What is the temperature in Krakow Poland",
            "Tell which tool you used",
            [], 
            "", 
            ""
        )

        print(result)
        print("Loaded sidekick")

        sidekick.cleanup()

    asyncio.run(main())
    
