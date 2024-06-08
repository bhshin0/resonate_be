# from langchain.chat_models import ChatOpenAI
# from langchain.chains import LLMChain
# from langchain.prompts import MessagesPlaceholder, HumanMessagePromptTemplate, ChatPromptTemplate
# from langchain.document_loaders import TextLoader
# from langchain.text_splitter import CharacterTextSplitter
# from transformers import pipeline
# from dotenv import load_dotenv

# load_dotenv()

# chat = ChatOpenAI()

# text_splitter = CharacterTextSplitter(
#     separator="-",
#     chunk_size=500, #UP TO 200 characters
#     chunk_overlap=0 #adds overlap thats shared between chunks?
# )

# loader = TextLoader("journal.txt")
# docs = loader.load_and_split(
#     text_splitter=text_splitter
# ) #text is loaded as a list

# prompt = ChatPromptTemplate(
#     input_variables= ["content","messages"],
#     messages=[
#         #MessagesPlaceholder(variable_name="messages"), #look for messages(or memory_key)
#         HumanMessagePromptTemplate.from_template("{content}")
#     ]
# )

# chain= LLMChain(
#     llm=chat,
#     prompt=prompt,
#     #memory=memory
# )

# emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

# def classify_emotions(text):
#     results = emotion_classifier(text)
#     return {result['label']: result['score'] for result in results[0]}


# #make dictionary by day, with journal entries, emotions
# def emotion_dictionary():
#     for doc in docs:
#         chunk_text = doc.page_content
#         emotions = classify_emotions(chunk_text)
#         print(doc)
#         # print(f"Text Chunk: {chunk_text}")
#         # print(f"Emotions: {emotions}")
#         # print("\n")




# #get insights from journal entry
# def GPT_prompter(prompt):
#     happy = prompt
#     print(chain({"content": happy + docs[0].page_content})['text'])
#     return




from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import MessagesPlaceholder, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()

class JournalProcessor:
    def __init__(self, file_path, chunk_size=500, chunk_overlap=0):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.docs = self.load_and_split()

    def load_and_split(self):
        text_splitter = CharacterTextSplitter(
            separator="-",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        loader = TextLoader(self.file_path)
        return loader.load_and_split(text_splitter=text_splitter)


class EmotionClassifier:
    def __init__(self):
        self.classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

    def classify(self, text):
        results = self.classifier(text)
        return {result['label']: result['score'] for result in results[0]}


class EmotionJournal:
    def __init__(self, journal_processor, emotion_classifier):
        self.journal_processor = journal_processor
        self.emotion_classifier = emotion_classifier
        self.emotion_dict = self.create_emotion_dictionary()

    def create_emotion_dictionary(self):
        emotion_dict = {}
        for i, doc in enumerate(self.journal_processor.docs):
            chunk_text = doc.page_content
            emotions = self.emotion_classifier.classify(chunk_text)
            emotion_dict[f"Entry_{i+1}"] = {
                "text": chunk_text,
                "emotions": emotions
            }
        return emotion_dict

    def display_emotion_dictionary(self):
        for entry, data in self.emotion_dict.items():
            print(f"{entry}:")
            print(f"Text: {data['text']}")
            print(f"Emotions: {data['emotions']}")
            print("\n")


# Initialize components
journal_processor = JournalProcessor("journal.txt")
emotion_classifier = EmotionClassifier()

# Create the EmotionJournal instance
emotion_journal = EmotionJournal(journal_processor, emotion_classifier)

# Display the results
emotion_journal.display_emotion_dictionary()
