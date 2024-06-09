from flask import Flask, request, jsonify
from main import EmotionJournal
from journal import journal_str

app = Flask(__name__)

# Define the function to process the input string
def process_string(input_string):
    journ = EmotionJournal(journal_str)
    return journ.to_json()

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    input_string = data.get('input_string')
    
    if not input_string:
        return jsonify({'error': 'No input string provided'}), 400
    
    result = process_string(input_string)
    return jsonify(result)

if __name__ == '__main__':
    app.run(port=5000)