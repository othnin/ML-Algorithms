from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from dotenv import load_dotenv
import os
'''
Using the updates API
'''

class GradeOutputParser(BaseOutputParser):
    """Parse LLM output to determine if answer is correct"""
    def parse(self, text: str) -> bool:
        """
        Parse the output of an LLM call.
        Returns True if answer is correct, False if wrong.
        """
        # Check for explicit indicators of incorrect answers
        wrong_indicators = ['wrong', 'incorrect', 'not correct', 'false']
        return not any(indicator in text.lower() for indicator in wrong_indicators)

def create_grading_chain(api_key: str, temperature: float = 0):
    """Create a grading chain with improved prompt and parsing"""
    
    chat_model = ChatOpenAI(openai_api_key=api_key, temperature=temperature)
    
    prompt_template_text = """You are a high school history teacher grading homework assignments.
    
Question: {question}
Correct Answer: {correct_answer}
Student's Answer: {student_answer}

Evaluate if the student's answer is correct. Be explicit in your response by using the words "correct" or "wrong".
Consider minor spelling variations as acceptable. Provide your evaluation in a single sentence."""

    prompt = PromptTemplate(
        input_variables=["question", "correct_answer", "student_answer"],
        template=prompt_template_text
    )
    
    # Create chain with parser
    chain = prompt | chat_model | GradeOutputParser()
    return chain

def grade_answers(chain, question: str, correct_answer: str, student_answers: list) -> dict:
    """Grade multiple student answers and return results"""
    results = {}
    
    for answer in student_answers:
        try:
            result = chain.invoke({
                'question': question,
                'correct_answer': correct_answer,
                'student_answer': answer
            })
            results[answer] = result
        except Exception as e:
            results[answer] = f"Error processing answer: {str(e)}"
    
    return results

# Example usage
if __name__ == "__main__":
    
    load_dotenv('api_keys.env')
    chatgpt_key = os.getenv('CHATGPT_KEY')

    # Create grading chain
    grading_chain = create_grading_chain(chatgpt_key)
    
    # Test questions
    question = "Who was the 35th president of the United States of America?"
    correct_answer = "John F. Kennedy"
    student_answers = [
        "John F. Kennedy",
        "JFK",
        "FDR",
        "John F. Kenedy",
        "John Kennedy",
        "Jack Kennedy",
        "Jacquelin Kennedy",
        "Robert F. Kenedy"
    ]
    
    # Grade answers
    results = grade_answers(grading_chain, question, correct_answer, student_answers)
    
    # Print results
    for answer, is_correct in results.items():
        print(f"Answer: {answer:<20} Correct: {is_correct}")