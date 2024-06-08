from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
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
    def __init__(self, file_path, chunk_size=500, chunk_overlap=0):
        self.journal_processor = JournalProcessor(file_path, chunk_size, chunk_overlap)
        self.emotion_classifier = EmotionClassifier()
        self.chat = ChatOpenAI()
        self.llm_chain = self.create_llm_chain()
        self.emotion_dict = self.create_emotion_dictionary()

    def create_llm_chain(self):
        prompt = ChatPromptTemplate(
            input_variables=["content", "messages"],
            messages=[
                HumanMessagePromptTemplate.from_template("{content}")
            ]
        )
        return LLMChain(llm=self.chat, prompt=prompt)

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

    def get_insights(self, entry_key, prompt_text):
        if entry_key in self.emotion_dict:
            journal_entry = self.emotion_dict[entry_key]['text']
            response = self.llm_chain({"content": prompt_text + journal_entry})
            return response['text']
        else:
            return "Invalid entry key."
    
    def get_insights_happy(self, entry_key, prompt_text,emotion):
        if entry_key in self.emotion_dict:
            journal_entry = self.emotion_dict[entry_key]['text']
            journal_entry_happy = self.emotion_dict[entry_key]['emotions'][emotion]
            print(journal_entry_happy)
            response = self.llm_chain({"content": prompt_text })
            return response['text']
        else:
            return "Invalid entry key."

# Initialize the EmotionJournal instance
emotion_journal = EmotionJournal("journal.txt")

# Display the emotion dictionary
#emotion_journal.display_emotion_dictionary()

# Get insights from a specific journal entry
entry_key = "Entry_1"  # Example entry key
prompt_text = "what number is this "
insights = emotion_journal.get_insights_happy(entry_key, prompt_text, 'joy')
print(f"Insights for {entry_key}: {insights}")
