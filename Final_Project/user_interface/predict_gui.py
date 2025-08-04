import tkinter as tk
from tkinter import filedialog, messagebox
from .predict_popularity.py import predict_popularity_range


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
    try:
        result = predict_popularity_range(path, genre)
        messagebox.showinfo("Prediction", result)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to predict: {str(e)}")

# GUI setup
root = tk.Tk()
root.title("Spotify Popularity Predictor")

tk.Label(root, text="MP3 File Path:").grid(row=0, column=0, padx=10, pady=5, sticky='w')
entry_path = tk.Entry(root, width=50)
entry_path.grid(row=0, column=1, padx=10, pady=5)
tk.Button(root, text="Browse", command=browse_file).grid(row=0, column=2, padx=10, pady=5)

tk.Label(root, text="Genre:").grid(row=1, column=0, padx=10, pady=5, sticky='w')
genre_var = tk.StringVar(value="pop")
tk.Entry(root, textvariable=genre_var).grid(row=1, column=1, padx=10, pady=5)

tk.Button(root, text="Predict Popularity", command=predict).grid(row=2, column=1, pady=15)

root.mainloop()
