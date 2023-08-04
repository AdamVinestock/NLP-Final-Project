from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1  # use GPU if available, otherwise CPU
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

def summarize(text):
    # print(f"device summarizer is on = {next(summarizer.model.parameters()).device}")
    summarized = summarizer(text, max_length=130, min_length=30, do_sample=False)
    summary_text = summarized[0]['summary_text']
    return summary_text

# ARTICLE = """ Traditional practices have deeply shaped the training routine at the Flight Academy for a considerable time. Omer, with his relentless pursuit of improvement, made numerous efforts to enhance cadet training and their daily routine, effectively boosting performance and learning rates. This called for thinking beyond conventional means, as the traditional practices were firmly embedded in the organization's culture.
#
# Initially, Omer recognized the instructional scope as being overly narrow and inadequate. To improve instructional quality, he introduced the innovative approach of pairing each trainee with an experienced pilot mentor. This unique mentorship focused not only on basic guidance of helicopter piloting, but also addressed broader aspects, including boosting confidence and resolving personal issues that could hinder learning. Additionally, Omer initiated a peer-assistance program where senior cadets were assigned to support their junior counterparts. This initiative facilitated a smoother transition into the program for the new cadets, built stronger connections among different classes in the Flight Academy, and elevated flight preparation quality. Lastly, Omer also pinpointed a significant gap in the traditional pre-flight briefing process, which was previously restricted to a private network within the airbase. This constraint prevented instructors and cadets from accessing to study materials and flight preparation resources when they were off base. Omer addressed this issue by transferring all flight materials and pre-flight briefing processes to the public internet network, developing a dedicated website for easier access, and establishing new practices to integrate the changes effectively.
#
# In his tenure as commander, responsible for overseeing cadets' day-to-day life and learning process, Omer implemented several innovative practices and reforms that drastically altered the cadets' experience during their helicopter training stage. Not only did training become more convenient, but performance also improved due to the enhanced tools and practices provided. His impactful changes and reforms have continued to influence the academy long after his departure.
#
# """
#
# print(summarizer(ARTICLE, max_length=130, min_length=30, do_sample=False))
