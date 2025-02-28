import subprocess
from mistralai import Mistral
import sys

api_key = ""
model = "mistral-small-latest"
client = Mistral(api_key=api_key)


def generate_suggestion(prompt):
    try:
        chat_response = client.chat.complete(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": f"{prompt}",
                },
            ]
        )
        print(chat_response.choices[0].message.content)
        return chat_response.choices[0].message.content

    except Exception as e:
        print(f"🚨 error occurred: {str(e)}")
        return None


def analyze_code(filepath):
    result = {}
    pylint_cmd = [sys.executable, "-m", "pylint", "--disable=all",
                  "--enable=too-many-arguments,too-many-locals,too-few-public-methods,"
                  "too-many-branches,too-many-statements,too-many-instance-attributes,"
                  "too-many-return-statements,too-many-nested-blocks,too-many-function-args,"
                  "duplicate-code,unused-argument,unused-variable,"
                  "unused-import,import-error,no-member,attribute-defined-outside-init",
                  filepath
                  ]

    try:
        pylint_output = subprocess.check_output(pylint_cmd, stderr=subprocess.STDOUT).decode()
        result['pylint'] = pylint_output
    except subprocess.CalledProcessError as e:
        print(f"\n🚨 Pylint error:\n{e.output.decode()}\n")
        result['pylint'] = e.output.decode()

    lines = result['pylint'].strip().split("\n")
    error_messages = []
    rating = ""

    for line in lines:
        if "Your code has been rated" in line:
            rating = line.split("at ")[1]
        elif line and not line.startswith("*************") and not line.startswith(
                "------------------------------------------------------------------"):
            error_messages.append(line.strip())

    return error_messages, rating
