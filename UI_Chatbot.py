import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog
from PIL import ImageTk, Image
import ollama
from sentence_transformers import SentenceTransformer
import psycopg2
from PIL import Image
import torch
import clip
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
ollama.base_url = "http://localhost:11434"

embedder = SentenceTransformer("BAAI/bge-m3")
def query_postgresql(query_text,k=2):
    query_embedding = embedder.encode(query_text).tolist()
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        dbname="postgres",
        user="postgres",
        password="PASSWORD"
    )

    cur = conn.cursor()
    query_embedding_str = "[" + ",".join(map(str,query_embedding)) + "]"
    # print(query_embedding)
    sql_query = """
        SELECT content , embedding <=> %s::vector AS similarity_score
        FROM documents
        ORDER BY similarity_score ASC
        LIMIT %s;
    """
    cur.execute(sql_query,(query_embedding_str,k))
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results
def generate_response(query_text):
    retrieved_docs = query_postgresql(query_text,3)
    context = "\n".join([doc[0] for doc in retrieved_docs])
    # print(context)
    # prompt = f"Answer the question based on the following context:\n{context}\n\n Question: {query_text}"
    prompt = f"""
    You are a helpful assistant. Use the context below to answer the question. If the answer is not contained in the context, say \"Leave me alone\".
    context :\n{context}\n\n
    Question : {query_text}
    """
    
    # print(prompt)
    response = ollama.chat(model="llama3.2", messages=[
        {"role" : "system" , "content" : "You are an assistant.before answer will end with กูมั่วนะอย่าเชื่อ"},
        {"role" : "user" , "content" : prompt},
    ])
    return response['message']['content']
def query_postgresql_image(query_text,k=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    image = preprocess(Image.open(query_text)).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    vector_list = image_features.squeeze(0).tolist()
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        dbname="postgres",
        user="postgres",
        password="PASSWORD"
    )

    cur = conn.cursor()
    query_embedding_str = "[" + ",".join(map(str,vector_list)) + "]"
    # print(query_embedding)
    sql_query = """
        SELECT content , embedding <=> %s::vector AS similarity_score
        FROM images
        ORDER BY similarity_score ASC
        LIMIT %s;
    """
    cur.execute(sql_query,(query_embedding_str,k))
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results
def generate_response_image(query_text):
    retrieved_docs = query_postgresql_image(query_text,4)
    context = "\n".join([doc[0] for doc in retrieved_docs])

    prompt = f"""
    You are a helpful assistant. Use the context below to answer the question. If the answer is not contained in the context, say \"Leave me alone\".
    context :\n{context}\n\n
    Question : คำตอบที่ได้จาก context เลิอกตอบอันแรกมา ตอบมาเลย
    """

    response = ollama.chat(model="llama3.2", messages=[
        {"role" : "system" , "content" : "You are an assistant.before answer will end with กูมั่วนะอย่าเชื่อ"},
        {"role" : "user" , "content" : prompt},
    ])
    return response['message']['content']
def send_message():
    user_message = message_entry.get()  # รับข้อความจาก Textbox

    if user_message:
        chat_window.insert(tk.END, "You: " + user_message + "\n")
        message_entry.delete(0, tk.END)  
        if mode.get() == 1:
            chat_window.insert(tk.END, "Bot: " + generate_response(user_message) + "\n")
            chat_window.yview(tk.END)
        elif mode.get() == 2:
            chat_window.insert(tk.END, "Bot: " + generate_response_image(user_message) + "\n")
            chat_window.yview(tk.END)


root = tk.Tk()
root.title("LLM Chatbot")
root.geometry("500x600")

mode = tk.IntVar()
mode.set(1)  # ตั้งค่าเริ่มต้นเป็นโหมด text-text

chat_window = tk.Text(root, height=20, width=60)
chat_window.pack(pady=10)

message_entry = tk.Entry(root, width=50)
message_entry.pack(pady=10)

send_button = tk.Button(root, text="Send", command=send_message)
send_button.pack(pady=10)

mode_frame = tk.Frame(root)
mode_frame.pack(pady=10)

text_text_radio = tk.Radiobutton(mode_frame, text="Text-Text", variable=mode, value=1)
text_text_radio.pack(side=tk.LEFT, padx=10)

image_text_radio = tk.Radiobutton(mode_frame, text="Image-Text-Text", variable=mode, value=2)
image_text_radio.pack(side=tk.LEFT, padx=10)

image_label = tk.Label(root)
image_label.pack(pady=10)

root.mainloop()
