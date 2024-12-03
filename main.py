from crewai import Agent, Task, Crew 
from crewai_tools import FileReadTool, RagTool 
from langchain_openai import ChatOpenAI
import os 
import warnings 
warnings.filterwarnings('ignore') 

os.environ["OPENAI_API_KEY"] = "sk-proj-1111" 

llm = ChatOpenAI(
    model="ollama/llama3.1:latest",
    base_url="http://localhost:11434/vi",
    )

# Define tools 
resume_read_tool = FileReadTool(file_path="./candidate_resume.md")
 
# Agent 1: Resume Analyzer
resume_analyzer = Agent(
    role="Resume Analyzer",
    goal="Extract relevant skills from the candidate's resume based on the job description{job_description}.",
    tools=[resume_read_tool, RagTool()],
    verbose=True,
    backstory=(
        "A detail-oriented analyzer proficient in identifying key skills from resumes and aligning them "
        "with the job requirements."
    ),
    llm=llm
)

# Agent 2: Question Generator
question_generator = Agent(
    role="Question Generator",
    goal="Generate interview questions based on the skills extracted from the resume.",
    tools=[],
    verbose=True,
    backstory=(
        "A creative thinker skilled at crafting insightful interview questions tailored to the "
        "candidate's skillset and the job description{job_description}"
    ),
    llm=llm
)

# Agent 3: Evaluate the candidates answers
candidate_evaluator = Agent(
    role="Candidate Evaluator",
    goal="Score the candidate's answers {candidate_answers} and determine how well they fit the job description. Do not answer the questions yourself.",
    tools=[],
    verbose=True,
    backstory=(
        "An analytical expert who evaluates candidate responses to assess their suitability "
        "for the job."
    ),
    llm=llm
)

# Task 1: Analyze Resume
resume_analysis_task = Task(
    description=(
        "Analyze the candidate's resume to extract skills and domains relevant to the job description{job_description}. "
        "Ensure the extracted skills align with the job requirements."
    ),
    expected_output=(
        "A structured list of relevant skills extracted from the resume."
    ),
    agent=resume_analyzer,
    async_execution=False
)

# Task 2: Prepare Interview Questions
question_preparation_task = Task(
    description=(
        "Using the skills identified by the Resume Analyzer, prepare five very short interview questions "
        "relevant to the job requirements."
    ),
    expected_output=(
        "A python list of only interview questions tailored to the candidate's skillset and the job description{job_description}. "
        "Do not return a data type of str, return a data type of list."
    ),
    context=[resume_analysis_task],
    agent=question_generator,
    async_execution=True
)

# Task 3: Evaluate Candidate
evaluation_task = Task(
    description=(
        "Evaluate the candidate's answers {candidate_answers} to the interview questions {questions} and determine their fit for the job. "
        "Do not answer the question yourself"
    ),
    expected_output=(
        "An evaluation summary indicating the candidate's fit for the job and score for each answer{candidate_answers}."
    ),
    context=[question_preparation_task],
    agent=candidate_evaluator,
    async_execution=False
)

# Define the Crew
hiring_process_crew = Crew(
    agents=[resume_analyzer, question_generator],
    tasks=[resume_analysis_task, question_preparation_task],
    verbose=True
)

# Inputs
hiring_inputs = {
    "candidate_resume_path": "./candidate_resume.md",
    "job_description": "junior frontend web developer"
}

# Kick off the process
result = hiring_process_crew.kickoff(inputs=hiring_inputs)

# Extract the generated questions from Task 2
input_string = question_preparation_task.output.raw
input_string = input_string.strip()[1:-1]
questions = [q.strip()[1:-1] for q in input_string.split('",')]

# Store the answers from the user for the questions
candidate_answers = []

print("\nPlease answer the following questions:")
for i, question in enumerate(questions, start=1):
    print(f"Question {i}: {question}")
    answer = input("Your Answer: ")
    candidate_answers.append(answer)

# Define evaluating process
evaluating_process_crew = Crew(
    agents=[candidate_evaluator],
    tasks=[evaluation_task],
    verbose=True
)

# Evaluation inputs
evaluation_inputs = {
    "candidate_answers": candidate_answers,
    "questions": questions
}

# Print inputs and process evaluation
print(evaluation_inputs)
evaluation_result = evaluating_process_crew.kickoff(inputs=evaluation_inputs)
print(evaluation_result)

