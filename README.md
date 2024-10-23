# Polirizer Backend

## 🌟 Project Overview

Polirizer Backend is a Flask-based API service that provides text summarization capabilities using various AI models. It's designed to work with a frontend application, offering features like user management, activity tracking, and text-to-speech conversion.

## 🚀 Key Features

- Text summarization using multiple AI models (Claude, GPT, Gemini, Mistral)
- Web scraping for URL-based summarization
- User authentication and management
- Activity logging and tracking
- Text-to-speech conversion
- Database integration with SQLAlchemy
- API rate limiting and security measures

## 🛠️ Technology Stack

- **Framework**: Flask
- **Database**: SQLAlchemy with PostgreSQL (Heroku Postgres)
- **AI Models**: Anthropic Claude, OpenAI GPT, Google Gemini, Mistral AI
- **Authentication**: JWT (JSON Web Tokens)
- **Security**: Flask-Talisman, CORS
- **Deployment**: Heroku

## 📁 File Structure

- `app.py`: Main application file containing all routes and core logic
- `requirements.txt`: List of Python dependencies
- `Procfile`: Heroku configuration file
- `runtime.txt`: Specifies the Python version for Heroku
- `.env`: (Not in repo) Environment variables file

## 🔑 Key Components

### 1. `app.py`

This is the heart of the application, containing:
- Flask app initialization
- Database models (User, UserActivity)
- API routes for summarization, user management, and TTS
- Integration with AI models
- Security configurations

### 2. `requirements.txt`

Lists all Python packages required to run the application. Key dependencies include:
- Flask and its extensions (CORS, Limiter, SQLAlchemy, Talisman)
- AI model libraries (anthropic, openai, google-generativeai, mistralai)
- Database and ORM tools (SQLAlchemy, psycopg2-binary)
- Utility libraries (requests, beautifulsoup4, cryptography)

### 3. `Procfile`

This file is crucial for Heroku deployment. It tells Heroku how to run the application:
web: gunicorn app:app
- `web`: Runs the main Flask application using Gunicorn

### 4. `runtime.txt`

Specifies the exact Python version for Heroku:
python-3.9.16


This ensures consistency between development and production environments.

## 🚦 API Endpoints

- `/summarize`: POST - Generate text summaries
- `/get_free_summaries_count`: POST - Retrieve remaining free summaries for a user
- `/get_tts_endpoint`: GET - Get the TTS service endpoint
- `/tts`: POST - Convert text to speech
- `/get_user_info`: POST - Retrieve user information
- `/get_or_create_token`: POST - Authenticate and get user token

## 🔒 Security Features

- JWT-based authentication
- API rate limiting
- Content Security Policy (CSP) with Flask-Talisman
- CORS configuration
- Encryption for sensitive data storage

## 🔧 Setup and Deployment

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables:
   Create a `.env` file in the root directory with the following variables:
   ```
   CLAUDE_API_KEY=your_claude_api_key
   GEMINI_API_KEY=your_gemini_api_key
   GPT_API_KEY=your_gpt_api_key
   MISTRAL_API_KEY=your_mistral_api_key
   ENCRYPTION_KEY=your_encryption_key
   DATABASE_URL=your_database_url
   TTS_ENDPOINT_URL=your_tts_endpoint_url
   API_KEY=your_api_key
   ```
   Replace `your_*_key` with actual API keys and values.

4. Local run: `flask run`
5. Heroku deployment:
   - Create a Heroku app
   - Set config vars in Heroku dashboard (use the same variables as in the `.env` file)
   - Push to Heroku: `git push heroku main`

## 🔍 Monitoring and Logging

- Utilizes Flask's built-in logging
- Heroku logs accessible via Heroku CLI: `heroku logs --tail`

## 🚧 Future Improvements

- Implement more robust error handling
- Add unit and integration tests
- Enhance scalability for high traffic
- Implement caching mechanisms
