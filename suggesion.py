import json
import requests
import logging
import os
from dotenv import load_dotenv
from tree import decisionTree
from user_history import (
    load_user_history,
    save_user_history,
    add_entry_to_history,
    update_user_info,
)

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
predictionState = "root"

openai_api_key = os.getenv("OPENAI_API_KEY")


def generate_openai_response(transcript):
    if not openai_api_key:
        logger.error("OPENAI_API_KEY is not set. Please check your .env file.")
        return None

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "gpt-4-turbo",
        "messages": [{"role": "user", "content": transcript}],
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        logger.info(f"OpenAI response: {response.status_code}")
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            logger.error(f"OpenAI API Error: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"OpenAI request failed: {str(e)}")

    return None


def interpret_response(user_response, question_node):
    options = list(question_node.keys())
    options.remove("question")

    prompt = f"""
    Given the user response: "{user_response}"
    And the question: "{question_node['question']}"
    Interpret the response and categorize it into one of the following options: {', '.join(options)}
    If the response doesn't properly address the question, return "invalid".

    Please return only one option from the list above, not multiple options.
    """

    interpreted_response = generate_openai_response(prompt)
    if interpreted_response is None:
        return "invalid"

    interpreted_response = interpreted_response.strip().lower()

    if interpreted_response in options:
        return interpreted_response
    elif ',' in interpreted_response:
        first_option = interpreted_response.split(',')[0].strip()
        return first_option if first_option in options else "invalid"

    return "invalid"


def update_user_history(question, answer, user_history):
    prompt = f"""
    Given the question: "{question}"
    And the user's response: "{answer}"
    Extract the following information: fname, lname, age, gender, height, weight.
    Return the extracted information in the format: {{ "fname": <fname>, "lname": <lname>, "age": <age>, "gender": <gender>, "height": <height>, "weight": <weight> }}.
    If any information is not available, return null for that field.
    """
    gpt_response = generate_openai_response(prompt)

    try:
        if gpt_response:
            extracted_info = json.loads(gpt_response)
            for key, value in extracted_info.items():
                if value is not None:
                    update_user_info(user_history, key, value)
    except json.JSONDecodeError:
        logger.error("Failed to parse GPT response for user information.")

    # Bullet point generation
    prompt = f"""
    Given the question: "{question}"
    And the user's response: "{answer}"
    Create a concise bullet point summary of the key information in the response. Do not have redundancy.
    """
    gpt_response = generate_openai_response(prompt)
    if gpt_response:
        bullet_points = [
            line.strip()
            for line in gpt_response.strip().split("\n")
            if line.strip()
        ]
    else:
        bullet_points = [f"Error processing: {answer}"]

    try:
        if isinstance(bullet_points, list):
            add_entry_to_history(user_history, bullet_points)
    except Exception as e:
        logger.error(f"Error adding bullet points to history: {str(e)}")

    return user_history


def rephrase_question(original_question, user_response, invalid_response=False, user_history=None):
    if user_history is None:
        user_history = {"entries": []}

    # Format user history
    user_history_list = []
    for entry in user_history.get("entries", []):
        for date, info in entry.items():
            user_history_list.extend([f"- {i}" for i in info])

    if "current_call" in user_history:
        user_history_list.extend(
            [f"- Current Call: {call}" for call in user_history["current_call"]]
        )

    user_history_list.extend([
        f"- Age: {user_history.get('age', 'N/A')}",
        f"- Gender: {user_history.get('gender', 'N/A')}",
        f"- Name: {user_history.get('fname', '')} {user_history.get('lname', '')}".strip(),
        f"- Weight: {user_history.get('weight', 'N/A')}",
        f"- Height: {user_history.get('height', 'N/A')}",
    ])

    user_history_formatted = "\n".join(user_history_list)
    print(user_history_formatted)

    if invalid_response:
        context = f"""
        The user responded: "{user_response}", which was not understood or is invalid.

        You need to:
        1. Politely inform the user that their response ("{user_response}") is invalid.
        2. Rephrase the original question to guide the user.
        3. Use the history below for personalization.
        4. Keep response under 30 words.

        User history:
        {user_history_formatted}

        Original Question: "{original_question}"
        """
    else:
        context = f"""
        Rephrase the question below for a personalized medical conversation. Use history for context.

        Original Question: "{original_question}"
        User History:
        {user_history_formatted}
        """

    response = generate_openai_response(context)
    if response is None:
        logger.error("OpenAI API returned None in rephrase_question.")
        return "Sorry, something went wrong. Please try again."

    return response.strip('"')


def gpt_call(user_response, phone_number):
    global predictionState

    user_history = load_user_history(phone_number)
    current_node = decisionTree[predictionState]
    current_question = current_node["question"]
    interpreted_response = interpret_response(user_response, current_node)

    if interpreted_response == "invalid":
        return rephrase_question(current_question, user_response, True, user_history)

    user_history = update_user_history(current_question, user_response, user_history)
    save_user_history(phone_number, user_history)

    if interpreted_response in current_node:
        predictionState = current_node[interpreted_response]
    else:
        return f"I couldn't understand your response. {current_question}"

    if predictionState not in decisionTree:
        return f"Based on your answers, you may have {predictionState}. Please consult a medical professional."

    next_question = decisionTree[predictionState]["question"]
    return rephrase_question(next_question, user_response, False, user_history)


def terminal():
    print("Welcome to the medical diagnosis assistant.")
    print("Please answer the following questions.")

    phone_number = input("Please enter your phone number: ")
    user_history = load_user_history(phone_number)

    rephrased_question = rephrase_question(
        decisionTree["root"]["question"], "", False, user_history
    )
    print(rephrased_question)
    user_input = input("Your answer: ")

    while True:
        response = gpt_call(user_input, phone_number)
        print(response)

        if "Based on your answers" in response:
            break

        user_input = input("Your answer: ")

    print("\nYour medical history:")
    print(json.dumps(user_history, indent=2))


if __name__ == "__main__":
    terminal()
