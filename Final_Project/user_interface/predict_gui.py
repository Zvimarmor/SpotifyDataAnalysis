import sys
import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox

# Load genre list
with open("Final_Project/user_interface/genre_list.json") as f:
    genre_list = json.load(f)

sys.path.append(os.path.dirname(__file__))
from predict_popularity import predict_popularity_range

def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("MP3 Files", "*.mp3")])
    if file_path:
        entry_path.delete(0, tk.END)
        entry_path.insert(0, file_path)

def predict():
    path = entry_path.get()
    selected_indices = genre_listbox.curselection()
    if not path:
        messagebox.showwarning("Missing file", "Please choose an MP3 file.")
        return
    if not selected_indices:
        messagebox.showwarning("Missing genre", "Please select a genre from the list.")
        return
    genre = genre_listbox.get(selected_indices[0])
    try:
        result = predict_popularity_range(path, genre)
        result_label.config(text=result)
        messagebox.showinfo("Prediction", result)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to predict: {str(e)}")

# GUI setup
root = tk.Tk()
root.title("Spotify Popularity Predictor")
root.geometry("320x550")
root.resizable(False, False)
root.configure(bg="white")

# Header
tk.Label(root, text="browse for mp3 song file",
         font=("Helvetica", 11), bg="white").pack(pady=(15, 5))

entry_path = tk.Entry(root, width=40)
entry_path.pack(pady=(0, 5))

tk.Button(root, text="Browse", command=browse_file).pack()

# Genre selection
tk.Label(root, text="select the genre of the song",
         font=("Helvetica", 11), bg="white").pack(pady=(20, 5))

# Listbox with scrollbar
genre_frame = tk.Frame(root, bg="white")
genre_frame.pack(pady=(0, 10))

scrollbar = tk.Scrollbar(genre_frame, orient="vertical")
scrollbar.pack(side="right", fill="y")

genre_listbox = tk.Listbox(genre_frame, height=10, width=32, exportselection=False,
                           yscrollcommand=scrollbar.set)
for genre in sorted(genre_list):
    genre_listbox.insert(tk.END, genre)
genre_listbox.pack(side="left", fill="y")
scrollbar.config(command=genre_listbox.yview)

# Predict Button
tk.Button(root, text="predict popularity!", command=predict,
          font=("Helvetica", 12), bg="#00cc99", fg="white").pack(pady=25)

# Prediction result
result_label = tk.Label(root, text="", font=("Helvetica", 11), bg="white", fg="black")
result_label.pack()

# Footer
footer_text = (
    "Created by Zvi Marmor, Shaked Hartal, and Shai Abu\n"
    "as part of the \"A Needle in a Data Haystack\" course at the\n"
    "Hebrew University of Jerusalem."
)
tk.Label(root, text=footer_text, font=("Helvetica", 8),
         bg="white", fg="gray", justify="center", wraplength=300).pack(side="bottom", pady=20)

root.mainloop()
