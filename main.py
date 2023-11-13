import pandas as pd
import cohere
import google.generativeai as palm
import openai

from dotenv import load_dotenv
import os

import streamlit as st

load_dotenv()


def get_cohere_response(query):
    cohere_test_api_key = os.getenv("COHERE_TEST_API_KEY")
    co = cohere.Client(cohere_test_api_key)  # This is your trial API key
    response = co.chat(
        model="command",
        message=query,
        temperature=0.3,
        chat_history=[],
        prompt_truncation="auto",
        stream=False,
        citation_quality="accurate",
        connectors=[{"id": "web-search"}],
        documents=[],
    )

    print("Cohere finished responding")
    return response.text


def get_palm_response(query):
    palm_api_key = os.getenv("PALM_API_KEY")
    palm.configure(api_key=palm_api_key)
    defaults = {
        "model": "models/text-bison-001",
        "temperature": 0.7,
        "candidate_count": 1,
        "top_k": 40,
        "top_p": 0.95,
        "max_output_tokens": 1024,
        "stop_sequences": [],
        "safety_settings": [
            {"category": "HARM_CATEGORY_DEROGATORY", "threshold": 1},
            {"category": "HARM_CATEGORY_TOXICITY", "threshold": 1},
            {"category": "HARM_CATEGORY_VIOLENCE", "threshold": 2},
            {"category": "HARM_CATEGORY_SEXUAL", "threshold": 2},
            {"category": "HARM_CATEGORY_MEDICAL", "threshold": 2},
            {"category": "HARM_CATEGORY_DANGEROUS", "threshold": 2},
        ],
    }
    prompt = query

    response = palm.generate_text(**defaults, prompt=prompt)
    print("PALM finished responding")
    return response.result


def get_openai_response(query):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": query}],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    print("OpenAI finished responding")
    return response.choices[0].message.content


# st.header("Compare AI Tools")
# st.divider()
# query = st.text_area(label="Query")
# button_clicked = st.button("Submit")
# st.divider()
# if button_clicked:
#     with st.spinner("Waiting for responses..."):
#         model_responses = get_model_responses(query)

#     st.subheader("Cohere Response")
#     cohere_response = st.text(body=model_responses["cohere"])

#     st.subheader("OpenAI Response")
#     cohere_response = st.text(body=model_responses["openai"])

#     st.subheader("PALM Response")
#     cohere_response = st.text(body=model_responses["palm"])


def get_model_responses(query, error_number, model_name):
    print(f"Processing error {error_number} with {model_name}...")

    if model_name == "cohere":
        cohere_response = get_cohere_response(query)
        print("Cohere finished responding")
        return cohere_response

    elif model_name == "openai":
        openai_response = get_openai_response(query)
        print("OpenAI finished responding")
        return openai_response

    elif model_name == "palm":
        palm_response = get_palm_response(query)
        print("PALM finished responding")
        return palm_response

    else:
        print(f"Invalid model name: {model_name}")
        return None


def process_code_snippets(file_path):
    # Read the CSV file
    snippets_df = pd.read_csv(file_path)

    # Initialize a list to store the results
    results = []

    # Iterate over each row in the DataFrame
    for index, row in snippets_df.iterrows():
        error_number = row["number"]
        code_snippet = row["code"]
        query = f"What's the error here?\n\n{code_snippet}"

        # Get the model responses for each code snippet
        cohere_response = get_model_responses(query, error_number, "cohere")
        openai_response = get_model_responses(query, error_number, "openai")
        palm_response = get_model_responses(query, error_number, "palm")

        # Add the original snippet information with the new responses
        result = {
            "number": error_number,
            "title": row["title"],
            "code": row["code"],
            "explanation": row["explanation"],
            "cohere_response": cohere_response,
            "openai_response": openai_response,
            "palm_response": palm_response,
        }
        results.append(result)

    # Convert the results list to a DataFrame
    results_df = pd.DataFrame(results)

    # Write the DataFrame to a new CSV file
    results_df.to_csv("results.csv", index=False)
    print("Results saved to 'results.csv'.")


# Update the Streamlit UI to handle file uploads

st.header("Compare AI Tools")
st.divider()
uploaded_file = st.file_uploader("Upload CSV file with code snippets", type="csv")
button_clicked = st.button("Submit")
st.divider()

if button_clicked and uploaded_file:
    with st.spinner("Processing..."):
        # Process the uploaded CSV file
        process_code_snippets(uploaded_file)

    st.success("Processing complete! Check 'results.csv' for the model responses.")
