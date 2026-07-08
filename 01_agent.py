#imports
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, Annotated, Literal
from tavily import TavilyClient
from datetime import datetime
import sqlite3
import re
import json

# Initializing
tavily= TavilyClient()
llm= ChatGroq(model="qwen/qwen3.6-27b", max_tokens=4096)

#State of agent
class ExamPrepState(TypedDict):
    exam: str
    subject: str
    topic: str
    content: str
    questions: list
    user_answers: list
    evaluation: str
    score: int

# Research Node
def research_content(state: ExamPrepState):
    exam= state["exam"]
    subject= state["subject"]
    topic= state["topic"]
    prompt1 = f"{topic} {subject} concepts"
    prompt2 = f"{topic} {exam} {subject} exam questions"
    research_content1= tavily.search(query= prompt1, max_results=2)
    research_content2= tavily.search(query= prompt2, max_results=2)
    return { "content": str(research_content1) + str(research_content2)}

# Generating question
def generate_questions(state: ExamPrepState):
    content= state["content"]
    system= f"""
    You are an expert exam question generator.

    I will provide study content. Generate multiple-choice questions (MCQs) based ONLY on the provided content.

    Requirements:
    - Generate 5 high-quality questions as the content reasonably supports.
    - Each question must have exactly 4 options: A, B, C, and D.
    - Only one option should be correct.
    - The question should test the understanding of a candidate not just memorization.
    - The questions should be exam specific not specific to a document or a single page.
    - The incorrect options should be plausible but clearly incorrect.
    - Do not introduce information that is not present in the provided content.
    - Avoid duplicate or trivial questions.
    - Cover all important concepts from the content.

    Return ONLY a valid JSON array in the following format. Do not include markdown, explanations, or any other text.

    [
        {{
            "question": "...",
            "options": {{
                "A": "...",
                "B": "...",
                "C": "...",
                "D": "..."
            }},
            "correct": "A"
        }}
    ]

    Content: {content}"""
    questions= llm.invoke(system)
    clean_questions= re.sub(r'<think>.*?</think>',  '', questions.content, flags= re.DOTALL).strip()
    try:
        questions= json.loads(clean_questions)
    except json.JSONDecodeError:
        questions = []
    return {"questions": list(questions)}

# Evaluating answers
def evaluate_answers(state: ExamPrepState):
    generated_question= state["questions"]
    user_answers= state["user_answers"]
    prompt = f"""
    You are an exam evaluator.

    Questions and correct answers: {state["questions"]}
    User answers: {state["user_answers"]}

    For each question:
    1. State if the user was correct or incorrect
    2. Explain why the correct answer is right
    3. Explain why the user's answer was wrong (if incorrect)

    Be clear and educational.
    """
    evaluation = llm.invoke(prompt)
    clean_evaluation = re.sub(r'<think>.*?</think>', '', evaluation.content, flags=re.DOTALL).strip()
    score=0
    for i,question in enumerate(state["questions"]):
        if i < len(state["user_answers"]):
            if state["user_answers"][i] == question["correct"]:
                score+=1
    return {"evaluation": str(clean_evaluation), "score": int(score)}

# Making Graph
graph= StateGraph(ExamPrepState)

# Adding nodes
graph.add_node("research_content", research_content)
graph.add_node("generate_questions", generate_questions)

# Entry point
graph.set_entry_point("research_content")

# Making edges
graph.add_edge("research_content", "generate_questions")

# Ending node
graph.add_edge("generate_questions", END)

# Memory
conn= sqlite3.connect("exam_prep.db", check_same_thread=False)
memory= SqliteSaver(conn)

# Compiling agent
agent= graph.compile(checkpointer=memory)

# Main loop
while True:
    exam = input("Select exam (SEBI/RBI/IFSCA/BARC) or quit: ")
    if exam == 'quit':
        break
    subject = input("Select subject (DBMS/OS/Networks/DSA): ")
    topic = input("Enter topic: ")
    config = {"configurable":{"thread_id":(exam + subject + topic).lower()}}
    # generate content+questions
    agent_response= agent.invoke({"exam": exam, "subject": subject, "topic": topic, "content":"", "questions":[], "evaluation":"", "user_answers":[], "score":0}, config)
    questions= agent_response["questions"]
    user_answers=[]
    for question in questions:
        print(question['question'])
        print(f"""
        A: {question['options']['A']}
        B: {question['options']['B']}
        C: {question['options']['C']}
        D: {question['options']['D']}""")
        user_answer= input("Select an option to lock it( A/B/C/D): ").upper().strip()
        user_answers.append(user_answer)
    result = evaluate_answers({
        "questions": questions,
        "user_answers": user_answers,
        "exam": exam,
        "subject": subject,
        "topic": topic,
        "content": "",
        "evaluation": "",
        "score": 0
    })
    print(f"\nEvaluation:\n{result['evaluation']}")
    print(f"\nYour Score: {result['score']}/{len(questions)}")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS progress (
            exam TEXT,
            subject TEXT,
            topic TEXT,
            score INTEGER,
            total INTEGER,
            date TEXT
        )
    """)
    cursor.execute("INSERT INTO progress VALUES (?, ?, ?, ?, ?, ?)",
        (exam, subject, topic, result["score"], len(questions), str(datetime.now())))
    conn.commit()

    
