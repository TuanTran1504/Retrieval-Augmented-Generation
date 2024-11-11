import asyncio
from langgraph.graph import StateGraph
from IPython.display import Image, display
from graph_function import *
from langgraph.graph import END, START
import os
import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, faithfulness, ResponseRelevancy, FactualCorrectness

# Set API Key
os.environ["OPENAI_API_KEY"] = "Your-OpenAI-Key"

# Define the workflow
workflow = StateGraph(GraphState)
workflow.add_node("websearch", web_search)  # web search
workflow.add_node("retrieve", retrieve)
workflow.add_node("retrieve2", retrieve2)# retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generate

workflow.set_conditional_entry_point(
    route_question,
    {
        "retrieve":"retrieve",
        "retrieve2":"retrieve2",
    }
)
# workflow.add_edge(START, "retrieve")
workflow.add_edge("websearch", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_edge("retrieve2", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "websearch",
        "generate": "generate",
    },
)
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
        "max retries": END,
    },
)

# Compile the graph
graph = workflow.compile()

async def generate_responses(questions):
    test_schema = []

    # Generate responses for each question asynchronously
    for question in questions:
        inputs = {"question": question, "max_retries": 3}
        result = await graph.invoke(inputs) if asyncio.iscoroutinefunction(graph.invoke) else graph.invoke(inputs)
        generation = result.get("generation").content if "generation" in result else None
        documents_retrieved = result.get("documents", [])
        documents = [doc.page_content for doc in documents_retrieved if hasattr(doc, 'page_content')]

        # Append the generated result to test_schema
        test_schema.append({
            "question": question,
            "answer": generation,
            "contexts": documents
        })

    return test_schema

async def evaluate_responses(test_schema):
    for entry in test_schema:
        if entry["answer"] is None:
            print(f"Skipping evaluation for question '{entry['question']}' due to missing generation.")
            continue

        # Prepare dataset for evaluation
        rag_dataset = {
            "question": entry["question"],
            "answer": entry["answer"],
            "contexts": entry["contexts"]
        }
        rag_df = pd.DataFrame([rag_dataset])
        rag_eval_dataset = Dataset.from_pandas(rag_df)

        # Evaluate the generated responses
        result = evaluate(rag_eval_dataset, metrics=[faithfulness, ResponseRelevancy()])
        entry["evaluation"] = result.to_dict() if hasattr(result, "to_dict") else str(result)
        print("Evaluation result:", result)

async def main():
    questions = []

    # Load questions from text file
    with open("questions.txt", "r") as file:
        for line in file:
            question = line.strip()
            if question:
                questions.append(question)

    print("Questions loaded:", questions)

    # Step 1: Generation Phase
    test_schema = await generate_responses(questions)

    # Save the generation results to a JSON file
    # with open("test_schema.json", "w") as json_file:
    #     json.dump(test_schema, json_file, indent=4)

    print("Generation phase complete. Proceeding to evaluation.")

    # Step 2: Evaluation Phase
    await evaluate_responses(test_schema)

    with open("test.json", "w") as eval_file:
        json.dump(test_schema, eval_file, indent=4)

# Run the async workflow
asyncio.run(main())
