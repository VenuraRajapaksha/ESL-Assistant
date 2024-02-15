import openai

# Set your OpenAI API key
openai.api_key = "voided due to personal data"

def is_complex_or_special_phrase(sentence):
    # Define the prompt to ask the model
    prompt = f"Is the following sentence complex or a special phrase like an idiom?\n\"{sentence}\"\n--\n"

    # Send the prompt to the GPT-3 model for classification
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=1,
        temperature=0,
        stop="\n"
    )

    # Extract the classification label from the model's response
    label = response.choices[0].text.strip()

    return label

# Example usage
sentence = "It's raining cats and dogs."
classification = is_complex_or_special_phrase(sentence)
print("Classification:", classification)

