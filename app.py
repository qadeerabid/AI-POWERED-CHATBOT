import os
from src.utils.chatbot_utils import BuildChatbot
from src.utils.logger import logging
from src.utils.exception import Custom_exception

from flask import Flask, request, render_template, jsonify


# initializing flask app
app = Flask(__name__)

# setting up the chatbot(retriever)
utils = BuildChatbot()
chatbot = utils.initialize_chatbot()



# route for home page
@app.route('/')
def home():
    # ...removed debug print...
    return render_template('home_page.html')




@app.route('/chat', methods=["GET", "POST"])
def chat():
    # ...removed debug print...
    logging.info("/chat route called")
    try:
        data = request.get_json()
        logging.info(f"Request JSON: {data}")
        question = data.get('input', '')
        logging.info(f"User Input: {question}")

        config = {"configurable": {"session_id": "chat_1"}}
        logging.info("Invoking chatbot...")
        response = chatbot.invoke({"input": question}, config=config)
        logging.info(f"Chatbot Response: {response['answer']}")
        return jsonify({"response": response['answer']})    
    except Exception as e:
        logging.error(f"Chatbot error: {str(e)}", exc_info=True)
    # ...removed debug error return comment...
        return jsonify({"response": f"[ERROR] {str(e)}"}), 500



if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)