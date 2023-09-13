from transformers import AutoModel
import transformers
import torch
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

print("cette partie concerne le Chapitre 1 sur transformers\n:"," rien ne s'affiche")



classifier = pipeline("sentiment-analysis")
classifier("I've been waiting for a HuggingFace course my whole life.")
classifier(
    ["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!"]
)

classifier = pipeline("zero-shot-classification")
classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)

generator = pipeline("text-generation")
generator("In this course, we will teach you how to")

generator = pipeline("text-generation", model="distilgpt2")
generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)

unmasker = pipeline("fill-mask")
unmasker("This course will teach you all about <mask> models.", top_k=2)

ner = pipeline("ner", grouped_entities=True)
ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")

question_answerer = pipeline("question-answering")
question_answerer(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
)

#print("My name is Kant and I work at Hugging Face in Rome")

summarizer = pipeline("summarization")
summarizer("""America has changed dramatically during recent years.""")

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
translator("Ce cours est produit par Hugging Face.")




