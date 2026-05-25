import os
from openai import OpenAI
from langfuse import get_client, propagate_attributes
from dotenv import load_dotenv

load_dotenv()

# Langfuse v4 client
langfuse = get_client()

# NVIDIA NIM client
nvidia_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVIDIA_API_KEY"),
)

MODEL = "qwen/qwen3-coder-480b-a35b-instruct"
CYCLE_ROUNDS = 5
LOOP_TEST_SEARCH_QUERY = "Napoleon Hill Think and Grow Rich book revenue earnings"


def fake_search(query: str) -> str:
    return f"[Search result for '{query}']: Found relevant information."


def run_agent(user_question: str):
    print(f"\n[Agent] Question: {user_question}\n")

    with langfuse.start_as_current_observation(
        as_type="span",
        name="research-agent",
        input={"question": user_question},
    ) as root:
        final_answer = ""

        for round_index in range(1, CYCLE_ROUNDS + 1):
            print(f"\n[Loop Test] Redundant research round {round_index}/{CYCLE_ROUNDS}")

            # Step 1: Plan
            with langfuse.start_as_current_observation(
                as_type="generation",
                name="plan-step",
                model=MODEL,
                input={
                    "task": "generate search query",
                    "round": round_index,
                    "loop_test": True,
                },
            ) as plan_obs:
                plan_response = nvidia_client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "This is a loop-detection stress test. Always output exactly "
                                f"this one-line search query and nothing else: {LOOP_TEST_SEARCH_QUERY}"
                            ),
                        },
                        {"role": "user", "content": user_question},
                    ],
                    max_tokens=50,
                )
                search_query = plan_response.choices[0].message.content.strip()
                plan_obs.update(
                    output={"search_query": search_query},
                    usage={
                        "input": plan_response.usage.prompt_tokens,
                        "output": plan_response.usage.completion_tokens,
                        "total": plan_response.usage.total_tokens,
                    }
                )
                print(f"[Step 1 - Plan] Search query: {search_query}")

            # Step 2: Tool call
            with langfuse.start_as_current_observation(
                as_type="span",
                name="tool-search",
                input={
                    "query": search_query,
                    "round": round_index,
                    "loop_test": True,
                },
            ) as tool_obs:
                search_result = fake_search(search_query)
                tool_obs.update(output={"result": search_result})
                print(f"[Step 2 - Tool] {search_result}")

            # Step 3: Synthesize
            with langfuse.start_as_current_observation(
                as_type="generation",
                name="synthesize-answer",
                model=MODEL,
                input={
                    "question": user_question,
                    "search_result": search_result,
                    "round": round_index,
                    "loop_test": True,
                },
            ) as answer_obs:
                answer_response = nvidia_client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "This is a redundant-cycle test. Answer using only the search result. "
                                "Do not add new reasoning, do not mention the round number, and keep "
                                "the same concise wording if the same search result is provided again."
                            ),
                        },
                        {"role": "user", "content": f"Question: {user_question}\n\nSearch result: {search_result}"},
                    ],
                    max_tokens=200,
                )
                final_answer = answer_response.choices[0].message.content.strip()
                answer_obs.update(
                    output={"answer": final_answer},
                    usage={
                        "input": answer_response.usage.prompt_tokens,
                        "output": answer_response.usage.completion_tokens,
                        "total": answer_response.usage.total_tokens,
                    }
                )
                print(f"[Step 3 - Answer] {final_answer}")

        root.update(
            output={
                "final_answer": final_answer,
                "loop_test": True,
                "repeated_cycle": ["plan-step", "tool-search", "synthesize-answer"],
                "rounds": CYCLE_ROUNDS,
            }
        )

    langfuse.flush()
    print("\n[OK] Trace sent to Langfuse. Check your dashboard.")


if __name__ == "__main__":
    run_agent("How much revenue the Napolean Hill earned for his book 'Think and Grow Rich'?")
