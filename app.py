from flask import Flask, request, jsonify, render_template
import pickle
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI
import faiss

# Load the vectorStore from the pickled file
with open("faiss_store_openai.pkl", "rb") as f:
    vectorStore_openAI = pickle.load(f)

# Initialize OpenAI language model
llm = OpenAI(temperature=0,  openai_api_key="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")

# Create the QA chain
chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorStore_openAI.as_retriever())

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    if request.method == 'POST':
        data = request.get_json()
        question = data.get('question')

        if not question:
            return jsonify({"error": "Question not provided"}), 400

        result = chain({"question": question}, return_only_outputs=True)
        answer = result.get("answer", "No answer found")

        return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True,port=8000)
