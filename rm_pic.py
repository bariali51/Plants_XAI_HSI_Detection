import os
from pathlib import Path

media_path = Path(r"C:\Users\bari\Desktop\projer-main\Plants_XAI_HSI_Detection-main\media")
extensions = {".png", ".jpg", ".jpeg"}

deleted_count = 0
for file_path in media_path.rglob("*"):
    if file_path.is_file() and file_path.suffix.lower() in extensions:
        file_path.unlink()
        print(f"Deleted: {file_path}")
        deleted_count += 1

print(f"Done. Deleted {deleted_count} pictures.")