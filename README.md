# FYP_01

Final Year Project - Prevention-Focused Cyberbullying Detection Discord Bot Using Neural Networks

## Description
This project is a Discord bot that detects cyberbullying in text messages using neural networks. The bot is designed to be prevention-focused, meaning it aims to identify and mitigate potential instances of cyberbullying before they escalate.

## Setup Instructions

Server setup is required to run the bot. Follow these steps to set up the server and run the bot:

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Create a virtual environment (optional but recommended) refer to [ENV.md](ENV.md) for more details.
4. Create a .env file in the root directory and add environment variables. Refer to .env.example for more details.
5. Install the required packages using `pip install -r requirements.txt`.
6. Make sure all the models are in the `model` folder.
7. Install `Tesseract-OCR` folder using the `tesseract.exe` file in the `model` folder.
8. Run the bot using `python app.py` or `fastapi dev app.py`.
9. Visit 127.0.0.1:8000/docs to access the web interface to check endpoints.

### Folder Structure
```
.
├── data
├── model
│   ├── Tesseract-OCR
│   ├── count_vectorizer.pkl
│   ├── logistic_regression_model.pkl
│   └── tesseract.exe
├── notebooks
├── python files
│   ├── A2T.py
│   ├── I2T.py
│   └──T2T.py
├── app.py
├── requirements.txt
├── README.md
├── ENV.md
└── .gitignore
```

### Environment Variables

- [Hugging Face Token](https://huggingface.co/settings/tokens)
