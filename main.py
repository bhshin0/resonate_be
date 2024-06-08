from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()

text_splitter = CharacterTextSplitter(
    separator="-",
    chunk_size=500, #UP TO 200 characters
    chunk_overlap=0 #adds overlap thats shared between chunks?
)

loader = TextLoader("journal.txt")
docs = loader.load_and_split(
    text_splitter=text_splitter
) #text is loaded as a list

emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

def classify_emotions(text):
    results = emotion_classifier(text)
    return {result['label']: result['score'] for result in results[0]}

for doc in docs:
    chunk_text = doc.page_content
    emotions = classify_emotions(chunk_text)
    print(f"Text Chunk: {chunk_text}")
    print(f"Emotions: {emotions}")
    print("\n")