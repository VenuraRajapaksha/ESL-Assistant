from openai import OpenAI

# Initialize OpenAI client with your API key
api_key = "voided due to personal data"

client = OpenAI(api_key=api_key)

# Define function to generate completions
def generate_completions(prompt, model='gpt-4-1106-preview'):
    # Generate completions using OpenAI API
    completion = client.completions.create(model=model, prompt=prompt)
    return completion.choices[0].text

# Define function to simplify or explain the sentence
def simplify_or_explain(sentence):
    # Generate completions with OpenAI API
    prompt = f"if it contains an idiom or special phrase please give its short meaning?\n'{sentence}'"
    completion = generate_completions(prompt)
    return completion

# Take user input
user_input = input("Enter a sentence to simplify or explain: ")

# Simplify or explain the sentence
result = simplify_or_explain(user_input)

# Print the result
print(result)
