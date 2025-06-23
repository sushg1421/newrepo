import json
from datetime import datetime
import os

from flask import redirect, url_for

FOLDER_PATH = "static/user_data"


def load_user_history(phone_number):
    print("phone_number: ", phone_number)
    filename = f"{FOLDER_PATH}/user_history_{phone_number}.json"
    if os.path.exists(filename):
        print("filename: ", filename)
        with open(filename, "r") as f:
            return json.load(f)

    # return {
    #     "entries": [],
    #     "fname": "N/A",
    #     "lname": "N/A",
    #     "age": "N/A",
    #     "gender": "N/A",
    #     "height": "N/A",
    #     "weight": "N/A",
    #     "current_call": []
    # }
    print("phone_number: ", phone_number)
    return {
        "entries": [],
        "fname": "Nathan",
        "lname": "Zhao",
        "age": "30",
        "gender": "Other",
        "height": "170",
        "weight": "70",
        "current_call": [],
        "username": phone_number,
        "password": "password",
        "phone_number": phone_number,
    }


def save_user_history(phone_number, user_history):
    print("Saving user history...")
    filename = f"{FOLDER_PATH}/user_history_{phone_number}.json"
    with open(filename, "w") as f:
        json.dump(user_history, f, indent=2)
    print("User history saved successfully")


def add_entry_to_history(user_history, new_info):
    user_history["current_call"].extend(new_info)


def update_user_info(user_history, key, value):
    user_history[key] = value


def finalize_call(user_history):
    if user_history["current_call"]:
        current_time = datetime.now().strftime("%m/%d/%Y %I:%M%p")
        user_history["entries"].append({current_time: user_history["current_call"]})
        user_history["current_call"] = []  # Clear the current call information

    return redirect(
        url_for("medical_record", phone_number=user_history["phone_number"][1:])
    )