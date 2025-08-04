import sys
import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

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
    genre = genre_var.get()
    if not path:
        messagebox.showwarning("Missing file", "Please choose an MP3 file.")
        return
    if genre == "" or genre == "-- Select Genre --":
        messagebox.showwarning("Missing genre", "Please choose a genre from the list.")
        return
    try:
        result = predict_popularity_range(path, genre)
        result_label.config(text=result)
        messagebox.showinfo("Prediction", result)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to predict: {str(e)}")

# GUI setup
root = tk.Tk()
root.title("🎵 Spotify Popularity Predictor")
root.geometry("380x550")
root.configure(bg="#1e1e1e")

# Header
tk.Label(root, text="🎧 Predict Your Song's Success!",
         font=("Helvetica", 16, "bold"), bg="#1e1e1e", fg="#00FFB2").pack(pady=20)

# MP3 File Selection
tk.Label(root, text="Choose MP3 file:", bg="#1e1e1e", fg="white", font=("Helvetica", 11)).pack(pady=(10, 0))
entry_path = tk.Entry(root, width=40)
entry_path.pack(pady=5)
tk.Button(root, text="Browse", command=browse_file).pack()

# Genre Selection
tk.Label(root, text="Choose Genre:", bg="#1e1e1e", fg="white", font=("Helvetica", 11)).pack(pady=(20, 0))
genre_var = tk.StringVar(value="-- Select Genre --")
genre_dropdown = ttk.Combobox(root, textvariable=genre_var,
                              values=sorted(genre_list), state="readonly", width=35)
genre_dropdown.pack(pady=5)

# Predict Button
tk.Button(root, text="🔮 Predict Popularity", command=predict,
          font=("Helvetica", 12), bg="#00FFB2", fg="black", padx=10, pady=5).pack(pady=30)

# Result Display
result_label = tk.Label(root, text="", bg="#1e1e1e", fg="white", font=("Helvetica", 12))
result_label.pack()

# Footer
footer_text = "Created by Zvi Marmor, Shaked Hartal, and Shai Abu\nas part of 'A Needle in a Data Haystack' @ Hebrew University"
tk.Label(root, text=footer_text, font=("Helvetica", 9),
         bg="#1e1e1e", fg="gray", justify="center").pack(side="bottom", pady=20)

root.mainloop()
