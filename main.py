from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.chains.llm import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import os
from dotenv import load_dotenv
from datetime import datetime
from utils import load_vectorstore

app = Flask(__name__)
CORS(app)
# Load environment variables
load_dotenv()

vectorstore = load_vectorstore(vectorstore_path="shopify_langchain_testing_vectorstore",index_name="products")
print("vectorstore loaded successfully")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0.7,
    max_tokens=50,
    timeout=None,
    max_retries=2,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

keywords = ""

memory = ConversationBufferWindowMemory(return_messages=True, k=2)


def get_keywords(input_text):
    try:

        prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """ you are system that helps to extract main keywords from the user query.
        output should be a list of keywords that are extracted from the user query without brackets.
        """,
    ),
    (
        "human",
        "{input}"
    ),
])
        chain = prompt | llm
        response = chain.invoke({"input": input_text})
        print(response.content)
        global keywords
        keywords +=response.content
        if len(keywords) > 50:
            keywords = keywords[10:]
        return response.content
        
    except Exception as e:
        print(f"Error in get_response: {str(e)}")
        raise

def convert_to_json(data):
    result = []
    for item in data:
        # Filter out unnecessary keys from metadata
        filtered_metadata = {
            key: value for key, value in item.metadata.items()
            if key not in ["seq_num", "source", "handle"]
        }
        result.append(filtered_metadata)
    return result


def get_product_search(query):
    result = vectorstore.similarity_search(query=query,k=4)
    return convert_to_json(result)

def get_response(input_text):
    try:

        prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Use the following step-by-step instructions to respond to user inputs.

        Step 1 - Identify the customer's request <specific product, recommendation, or general inquiry>.

        Step 2 - Respond in a short, direct manner <maximum 1-2 brief sentences> with only the most relevant information. and max tokens 20

        Step 3 - Avoid unnecessary details or extra recommendations unless explicitly asked.

        Step 4 - Keep the response clear, friendly, and free of special formatting.

        Remember to maintain context from the conversation history: {history}

        Deliver responses in plain text with minimal wording like a chatbots reply.
        """,
    ),
    (
        "human",
        "{input}"
    ),
])



        chain = LLMChain(
            llm=llm,
            prompt=prompt,
            memory=memory,
        )

        response = chain.invoke({"input": input_text})
        return response['text']
        
    except Exception as e:
        print(f"Error in get_response: {str(e)}")
        raise


def get_response_product_search(input_text,related_products):
    try:
        prompt = ChatPromptTemplate.from_messages([
        (
        "system",
        """You are Jackson Hardware Store's AI assistant. Your job is to:
        1. Help customers find the right tools, hardware, or equipment.
        2. Suggest relevant products based on customer needs and related items.
        3. Share key product details like brand, features, use cases, and availability.
        5. note important: Avoid using technical formatting like new line symbols, markdown symbols *, _, etc., or bullet points.
        6.  Respond in a short, direct manner <maximum 1-2 brief sentences> with only the most relevant information. and max tokens 20

        note: act as a chatbot 
        Deliver the response here in plain text without any formatting.
        chat history: {history}
        """,
    ),
    ("human", "{input}"),
])

        chain = LLMChain(
            llm=llm,
            prompt=prompt,
            memory=memory,
        )
        query = f"user query : {input_text} and related products based on user query:{str(related_products)}"
        response = chain.invoke({"input": query})
        return response['text']
        
    except Exception as e:
        print(f"Error in get_response: {str(e)}")
        raise 

def get_availability(input_text):
    try:
        prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """

        You are Jackson Hardware Store's AI assistant. Your job is to:
        
        extract the price and availability of the products asked by the user from the related products list: {input}

        note respond 0$ if not mentioned.

         Respond in a short, direct manner <maximum 1-2 brief sentences> with only the most relevant information. and max tokens 20

        example: user: "I need a hammer." and related products based on user query: <<"title": "Hammer", "price": "$10", "inventory_quantity": 5>, <"title": "Screwdriver", "price": "$5", "inventory_quantity": 10>>
        output: "The hammer is available for $10 and we have 5 in stock."
        """,
    ),
    (
        "human",
        "{input}"
    ),
])



        chain = LLMChain(
            llm=llm,
            prompt=prompt,
            memory=memory,
        )

        response = chain.invoke({"input": input_text})
        return response['text']
        
    except Exception as e:
        print(f"Error in get_response: {str(e)}")
        raise


@app.route('/open-vectorstore', methods=['POST'])
def createvectorstore():
    from create_store import open_vectorstore
    open_vectorstore()
    print("vectorstore created successfully")
    return jsonify({"message": "Vectorstore created successfully"})


@app.route('/load-vectorstore', methods=['POST'])
def loadvectorstore():
    global vectorstore
    vectorstore = load_vectorstore(vectorstore_path="shopify_langchain_testing_vectorstore",index_name="products")
    print("vectorstore loaded successfully")
    return jsonify({"message": "Vectorstore loaded successfully"})

# Store chat history
chat_history = []

@app.route('/chat', methods=['POST'])
def chat():
    try:
        message = request.json
        message.update({
            'timestamp': datetime.now().isoformat(),
            'id': len(chat_history)
        })
        chat_history.append(message)


        search_keywords = get_keywords(message['content'])

        query = f"user query : {message['content']} and extracted keywords from user query:{search_keywords}"

        related_products_for_query = get_product_search(query) 

        ai_response = get_response(input_text = message['content'])
        response = {
            'content': ai_response,
            'sender': 'bot',
            'timestamp': datetime.now().isoformat(),
            'id': len(chat_history),
            'related_products_for_query':related_products_for_query
        }

        chat_history.append(response)
        
        return jsonify(response)
    
    except Exception as e:
        error_response = {
            'error_response': str(e),
            'content': "I apologize, but I encountered an error. Please try again.",
            'sender': 'bot',
            # 'error': True,
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(error_response), 500
    
@app.route('/check-availability', methods=['POST'])
def check_availability():
    try:
        message = request.json
        message.update({
            'timestamp': datetime.now().isoformat(),
            'id': len(chat_history)
        })
        chat_history.append(message)
        

        search_keywords = get_keywords(message['content'])

        squery = f"user query : {message['content']} and extracted keywords from user query:{search_keywords}"

        related_products_for_query = get_product_search(squery) 

        query = f"user query : {message['content']} and related products based on user query:{str(related_products_for_query)}"

        ai_response = get_availability(input_text = query)
        response = {
            'content': ai_response,
            'sender': 'bot',
            'timestamp': datetime.now().isoformat(),
            'id': len(chat_history),
            'related_products_for_query':related_products_for_query
        }

        chat_history.append(response)
        
        return jsonify(response)
    
    except Exception as e:
        error_response = {
            'error_response': str(e),
            'content': "I apologize, but I encountered an error. Please try again.",
            'sender': 'bot',
            # 'error': True,
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(error_response), 500
    
@app.route('/chat-product-search', methods=['POST'])
def chat_product_search():
    try:
        message = request.json
        message.update({
            'timestamp': datetime.now().isoformat(),
            'id': len(chat_history)
        })
        chat_history.append(message)


        search_keywords = get_keywords(message['content'])
        print(keywords)
        squery = f"user query : {message['content']} and extracted keywords from user query:{keywords}"

        related_products_for_query = get_product_search(squery)

        ai_response = get_response_product_search(input_text = message['content'], related_products = related_products_for_query)
        
        response = {
            'content': ai_response,
            'sender': 'bot',
            'timestamp': datetime.now().isoformat(),
            'id': len(chat_history),
            'related_products_for_query':related_products_for_query

        }
        chat_history.append(response)
        
        return jsonify(response)
    
    except Exception as e:
        error_response = {
            "error_response" : str(e),
            'content': "I apologize, but I encountered an error. Please try again.",
            'sender': 'bot',
            'error': True,
            'timestamp': datetime.now().isoformat()
        }
        return jsonify(error_response), 500
    

@app.route('/chat-history', methods=['GET'])
def get_chat_history():
    return jsonify(chat_history)

@app.route('/clear-chat', methods=['POST'])
def clear_chat():
    try:
        chat_history.clear()
        memory.clear()
        return jsonify({"message": "Chat history cleared successfully"})
    except Exception as e:
        return jsonify({"error": "Failed to clear chat history"}), 500
    
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "working"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(debug=True, port=port, host='0.0.0.0')
