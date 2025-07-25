from langchain.text_splitter import CharacterTextSplitter
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain.vectorstores import FAISS # vector db meta, chroma db
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import PromptTemplate 
from langchain.schema.runnable import RunnableLambda
from dotenv import load_dotenv
import os

class ChatBot:
    def __init__(self):
        load_dotenv("../.env")

        file_path = './it_sector.txt'
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=4)
        docs = text_splitter.split_text(text)

        embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v2:0",
        )
        # store the docs chunks into a vector database
        vectorstore = FAISS.from_texts(texts=docs, embedding=embeddings)

        llm = ChatBedrock(
            model_id="mistral.mistral-large-2402-v1:0",
            model_kwargs={"temperature": 0.5}
        )

        template = """
        You are an IT support chatbot. If the user greets you, respond politely and let them know you are here to assist with IT support questions.
        If they ask a question about the IT sector, use the following context to answer the question.
        If the question is out of context, respond politely that you are an IT support chatbot and can only assist with IT-related questions.
        If you don't know the answer, just say you don't know.
        You should respond with short and concise answers, no longer than 2 sentences.

        Context: {context}
        Question: {question}
        Answer:
        """

        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        def print_prompt(prompt_str):
            print("Prompt sent to LLM:\n", prompt_str)
            return prompt_str
        
        
        def retrieve_context(question):
            docs1 = vectorstore.similarity_search(question, k=4)
            # return only 4 chunks as a single string
            return "\n".join([doc.page_content for doc in docs1])

        self.rag_chain = (
            {"context": RunnableLambda(retrieve_context), "question": RunnablePassthrough()}
            | prompt
            | RunnableLambda(print_prompt)
            | llm
            | StrOutputParser()
        )
