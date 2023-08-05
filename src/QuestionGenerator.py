from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1  # use GPU if available, otherwise CPU
question_generator = pipeline("text2text-generation", model="iarfmoose/t5-base-question-generator", device=device)

def gen_question(text):
    print(f"question_generator is on device = {next(question_generator.model.parameters()).device}")
    question = question_generator(text, max_length=100, min_length=0, do_sample=False)
    question_text = question[0]["generated_text"]
    return question_text

# text = "Albert Einstein was a German-born theoretical physicist. He developed the theory of relativity."
# text = "Moluccans are the Austronesian-speaking and Papuan-speaking ethnic groups inhabiting the Maluku Islands."
# text = "The term 'Moluccan' is an umbrella term that covers the various Austronesian and Papuan languages spoken on the islands."
# text = "The largest group of Moluccans are the Tolo-speaking people."
# text = "The Maluku Islands are a group of volcanic islands in eastern Indonesia, located about 1,000 kilometres east of Java and 2,000 kilometres south of New Guinea."
# question = gen_question(text)
# print("Generated question:", question)