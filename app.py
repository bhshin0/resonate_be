from fastapi import FastAPI, HTTPException
from main import EmotionJournal 
import requests

app = FastAPI()

@app.post("/process_journal")
async def process_journal(journal_entry: str):
    # Send the journal entry to your friend's main.py for processing
    print(journal_entry)
    emotion_journal = EmotionJournal(journal_entry)
    
    # Return the processed JSON response from your friend
    return emotion_journal.to_json

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)