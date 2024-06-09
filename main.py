from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from transformers import pipeline
import json
from dotenv import load_dotenv

from journal import journal_str

load_dotenv()

# class JournalProcessor:
#     def __init__(self, file_path, chunk_size=500, chunk_overlap=0):
#         self.file_path = file_path
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap
#         self.docs = self.load_and_split()

#     def load_and_split(self):
#         text_splitter = CharacterTextSplitter(
#             separator="-",
#             chunk_size=self.chunk_size,
#             chunk_overlap=self.chunk_overlap
#         )
#         loader = TextLoader(self.file_path)
#         return loader.load_and_split(text_splitter=text_splitter)
    

class JournalProcessor:
    def __init__(self, journal_text, chunk_size=500, chunk_overlap=0):
        self.journal_text = journal_text
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.docs = self.split_text()

    def split_text(self):
        text_splitter = CharacterTextSplitter(
            separator="-",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        return text_splitter.split_documents([Document(page_content=self.journal_text)])
    

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
            emotion_dict[f"{i+1}"] = {
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
    
    def get_insights_custom(self, entry_key, prompt_text,emotion):
        if entry_key in self.emotion_dict:
            journal_entry = self.emotion_dict[entry_key]['text']
            journal_entry_happy = self.emotion_dict[entry_key]['emotions'][emotion]
            print(journal_entry_happy)
            response = self.llm_chain({"content": str(journal_entry_happy) + prompt_text + journal_entry})
            return response['text']
        else:
            return "Invalid entry key."
    
    def get_insights_custom_top(self, emotion):
        if emotion in self.emotion_dict["1"]["emotions"]:
            response = self.llm_chain({"content": "Here are 3 diary entries, provide in one short sentence one activity what I can do from these days to have similiar " 
                                       + emotion + "in the future:" + self.get_concatenated_top_entries_text_by_emotion(emotion)})
            return response['text']
        else:
            return "Invalid entry key."
        
    def get_top_entries_by_emotion(self, emotion, top_n=3):
        entries_with_scores = [
            (entry, data['emotions'].get(emotion, 0))
            for entry, data in self.emotion_dict.items()
        ]
        # Sort entries by the emotion score in descending order
        sorted_entries = sorted(entries_with_scores, key=lambda x: x[1], reverse=True)
        top_entries = sorted_entries[:top_n]
        return top_entries
    
    def get_concatenated_top_entries_text_by_emotion(self, emotion, top_n=3):
        top_entries = self.get_top_entries_by_emotion(emotion, top_n)
        concatenated_text = " ".join([self.emotion_dict[entry]['text'] for entry, _ in top_entries])
        return concatenated_text
        
    def to_json(self):
        return json.dumps(self.emotion_dict, indent=4)
    
    def save_to_json_file(self, file_name):
        with open(file_name, 'w') as json_file:
            json.dump(self.emotion_dict, json_file, indent=4)

# Initialize the EmotionJournal instance
emotion_journal = EmotionJournal(journal_str)

# Display the emotion dictionary
#emotion_journal.display_emotion_dictionary()

# Get insights from a specific journal entry
entry_key = "1"  # Example entry key
# prompt_text = "is the value of happiness for the following journal entry, please give insights"
# insights = emotion_journal.get_insights_custom(entry_key, prompt_text, 'joy')
# emotion_journal.save_to_json_file('test')
# print(f"Insights for {entry_key}: {insights}")
print(emotion_journal.get_concatenated_top_entries_text_by_emotion('joy'))
print(emotion_journal.get_insights_custom_top('joy'))
#print(emotion_journal.emotion_dict)
