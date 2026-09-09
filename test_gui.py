import customtkinter as ctk
import sys
print("ctk version:", ctk.__version__)
try:
    app = ctk.CTk()
    print("app initialized successfully")
    app.withdraw()
except Exception as e:
    print("Error initializing CTk:", e)
    sys.exit(1)
