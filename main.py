import torch
import customtkinter as ctk
import nltk
import requests
import torch
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bs4 import BeautifulSoup

def extract_article(url):
    
    try:
        response = requests.get(url)
    except requests.exceptions.RequestException:
        url_entry.delete(0, "end")
        url_entry.configure(placeholder_text="Please check the URL and try again.")
        return
    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup.find_all(['title', 'script', 'style', 'nav', 'footer', 'aside']):
        tag.decompose()

    paragraphs = soup.find_all('p')
    text = "\n".join(paragraph.get_text().strip() for paragraph in paragraphs)

    end_tags = ["Photo: ", "Disclaimer: This content", "Image via", "Image: ", "Photo Courtesy: "]
    for tag in end_tags:
        end_position = text.find(tag)
        if end_position != -1:
            break
            
    text = text[:end_position]

    return text

def analyze_article(article):
    article_text = article
    try:
        article_list = sent_tokenize(article_text)
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
        article_list = sent_tokenize(article_text)

    tensor_cumulative_scores = torch.tensor([0.0, 0.0, 0.0])
    for sentence in article_list:
        inputs = tokenizer(sentence, return_tensors="pt")

        with torch.no_grad():
            logits = model(**inputs).logits

        tensor_cumulative_scores = tensor_cumulative_scores + logits[0]
    
    tensor_cumulative_scores = torch.softmax(tensor_cumulative_scores, dim=0)
    tensor_cumulative_scores = torch.round((tensor_cumulative_scores).to(torch.float64), decimals=2)

    tensor_cumulative_scores = tensor_cumulative_scores.tolist()
    negative_progress_bar.set(tensor_cumulative_scores[0])
    negative_label.configure(text="Negative: " + str(tensor_cumulative_scores[0] * 100) + "%")
    neutral_progress_bar.set(tensor_cumulative_scores[1])
    neutral_label.configure(text="Neutral: " + str(tensor_cumulative_scores[1] * 100) + "%")
    positive_progress_bar.set(tensor_cumulative_scores[2])
    positive_label.configure(text="Positive: " + str(tensor_cumulative_scores[2] * 100) + "%")
    url_entry.delete(0, "end")

def button_event():
    button.configure(text="Loading..", state="disabled")
    app.update()
    url = url_entry.get()
    article = extract_article(url)
    if article is None:
        button.configure(text="Analyze", state="normal")
        return
    analyze_article(article)
    button.configure(text="Analyze", state="normal")

model = AutoModelForSequenceClassification.from_pretrained("Mysticz0/finance-pro-model-v1.0")
tokenizer = AutoTokenizer.from_pretrained("Mysticz0/finance-pro-model-v1.0")

app = ctk.CTk()
app.title("Financial Sentiment Analyzer")
app.geometry("930x235")
app.iconbitmap("logo.ico")
app.configure(fg_color="#23272d")
app.configure(bg_color="#23272d")
app.configure(text_color="black")
app.resizable(False, False)
app.configure(button_color="white")
app.configure(button_hover_color="lightgray")
app.configure(button_active_color="gray")
app.configure(button_text_color="black")
app.configure(button_text_hover_color="black")
app.configure(button_text_active_color="black")

url_entry = ctk.CTkEntry(
    app, 
    placeholder_text="Enter the URL of the article", 
    width=750, 
    height=50, 
    fg_color="white", 
    border_color="white", 
    corner_radius=25,
    text_color="black"
)

button = ctk.CTkButton(
    app,
    text="Analyze",
    width=100,
    height=50,
    fg_color="white",
    border_color="white",
    corner_radius=25,
    text_color="black",
    command=button_event
)

negative_progress_bar = ctk.CTkProgressBar(
    app,
    width=750,
    height=25,
    fg_color="white",
    border_width=2,
    corner_radius=25,
    progress_color="red",
)

neutral_progress_bar = ctk.CTkProgressBar(
    app,
    width=750,
    height=25,
    fg_color="white",
    border_width=2,
    corner_radius=25,
    progress_color="yellow",
)

positive_progress_bar = ctk.CTkProgressBar(
    app,
    width=750,
    height=25,
    fg_color="white",
    border_width=2,
    corner_radius=25,
    progress_color="green"
)

negative_label = ctk.CTkLabel(
    app,
    text="Negative",
    font=("Arial", 20),
    text_color="white"
)

neutral_label = ctk.CTkLabel(
    app,
    text="Neutral",
    font=("Arial", 20),
    text_color="white"
)

positive_label = ctk.CTkLabel(
    app,
    text="Positive",
    font=("Arial", 20),
    text_color="white"
)

negative_progress_bar.set(0)
neutral_progress_bar.set(0)
positive_progress_bar.set(0)

url_entry.grid(row=0, column=0, padx=10, pady=10)
button.grid(row=0, column=1, padx=10, pady=10)

negative_progress_bar.grid(row=1, column=0, padx=10, pady=10)
negative_label.grid(row=1, column=1, padx=10, pady=10)
neutral_progress_bar.grid(row=2, column=0, padx=10, pady=10)
neutral_label.grid(row=2, column=1, padx=10, pady=10)
positive_progress_bar.grid(row=3, column=0, padx=10, pady=10)
positive_label.grid(row=3, column=1, padx=10, pady=10)

app.mainloop()