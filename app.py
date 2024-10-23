import sys
import os
import traceback
import logging
from flask import Flask, request, jsonify, send_file, Response 
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from openai import OpenAI
import google.generativeai as genai
import re
import openai
import anthropic
import uuid
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from marshmallow import Schema, fields, ValidationError, validate, validates_schema
from flask_talisman import Talisman
import bleach
from google.api_core import exceptions as google_exceptions
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
import pkg_resources
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import unicodedata
import time
import io
from functools import wraps
import secrets
import jwt
from flask.logging import create_logger
from datetime import datetime, timedelta, timezone
from cryptography.fernet import Fernet
from flask_migrate import Migrate
from sqlalchemy.exc import IntegrityError

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        auth_header = request.headers.get('Authorization')
        app.logger.info(f"Auth header: {auth_header}")
        
        if auth_header:
            try:
                token = auth_header.split(" ")[1]
            except IndexError:
                app.logger.warning("Authorization header is malformed")
                return jsonify({'message': 'Token is missing'}), 401
        
        if not token:
            app.logger.warning("Token is missing in the request")
            app.logger.debug(f"Headers: {request.headers}")
            return jsonify({'message': 'Token is missing'}), 401
        
        try:
            app.logger.info(f"Decoding token: {token}")
            data = jwt.decode(token, str(app.config['ENCRYPTION_KEY']), algorithms=["HS256"])
            user_id = data.get('user_id') or data.get('sub')
            if not user_id:
                raise ValueError("User ID not found in token")
            current_user = User.query.filter_by(id=user_id).first()
            if not current_user:
                app.logger.warning(f"User not found for token: {token}")
                return jsonify({'message': 'Invalid token'}), 401
        except jwt.ExpiredSignatureError:
            app.logger.warning(f"Expired token: {token}")
            return jsonify({'message': 'Token has expired'}), 401
        except jwt.InvalidTokenError as e:
            app.logger.warning(f"Invalid token: {token}. Error: {str(e)}")
            return jsonify({'message': 'Invalid token'}), 401
        except Exception as e:
            app.logger.error(f"Error decoding token: {str(e)}")
            return jsonify({'message': 'Invalid token'}), 401
        
        return f(current_user, *args, **kwargs)
    return decorated

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting application...")
logger.info("Creating Flask app...")

# Load environment variables
load_dotenv()

# Create Flask app
app = Flask(__name__)

# Set up API keys
app.config['CLAUDE_API_KEY'] = os.getenv('CLAUDE_API_KEY')
app.config['GEMINI_API_KEY'] = os.getenv('GEMINI_API_KEY')
app.config['GPT_API_KEY'] = os.getenv('GPT_API_KEY')
app.config['MISTRAL_API_KEY'] = os.getenv('MISTRAL_API_KEY')

# Verify API keys are set
if not app.config['CLAUDE_API_KEY']:
    app.logger.error("CLAUDE_API_KEY is not set in the environment variables")
if not app.config['GEMINI_API_KEY']:
    app.logger.error("GEMINI_API_KEY is not set in the environment variables")
if not app.config['GPT_API_KEY']:
    app.logger.error("GPT_API_KEY is not set in the environment variables")
if not app.config['MISTRAL_API_KEY']:
    app.logger.error("MISTRAL_API_KEY is not set in the environment variables")

# Near the top of your file, after creating the Flask app
ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY')
if not ENCRYPTION_KEY:
    raise ValueError("ENCRYPTION_KEY environment variable is not set")
app.config['ENCRYPTION_KEY'] = ENCRYPTION_KEY

cipher_suite = Fernet(ENCRYPTION_KEY.encode())

# Database configuration
database_url = os.environ.get('DATABASE_URL')
if database_url and database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)
if not database_url:
    database_url = 'sqlite:////tmp/app.db'

app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

try:
    from flask_migrate import Migrate
    migrate = Migrate(app, db)
except ImportError:
    logger.error("Flask-Migrate not installed. Database migrations will not be available.")
    # Optionally, you can raise an error or handle it as needed

# Define User model
class User(db.Model):
    id = db.Column(db.String(36), primary_key=True)
    fingerprint = db.Column(db.String(255), nullable=False, default='unknown')
    last_ip = db.Column(db.String(45), nullable=False, default='unknown')
    summaries_left = db.Column(db.Integer, default=10)
    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    last_token_refresh = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    activities = db.relationship('UserActivity', backref='user', lazy=True)

class UserActivity(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey('user.id'), nullable=False)
    session_id = db.Column(db.String(36), nullable=True)
    model_selected = db.Column(db.String(50), nullable=False)
    searched_for = db.Column(db.String(10), nullable=False)
    searched_data = db.Column(db.Text, nullable=False)
    scrape_data = db.Column(db.Text)
    request_time = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now(timezone.utc))
    tts_used = db.Column(db.Boolean, default=False)
    copied_summary = db.Column(db.Boolean, default=False)
    summary_exported = db.Column(db.Boolean, default=False)

# Define constants
HUMAN_PROMPT = "\n\nHuman: "
AI_PROMPT = "\n\nAssistant: "

# Set up CORS
CORS(app, resources={
    r"/*": {
        "origins": [
            "chrome-extension://your_extension_id"
        ]
    }
})

# Set up rate limiting
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["50 per day", "5 per hour"],
    storage_uri="memory://"
)

# Add this near the top of the file, where other environment variables are loaded
TTS_ENDPOINT_URL = os.getenv('TTS_ENDPOINT_URL', 'your_server_url/tts')

# Set up API keys
SERVER_API_KEYS = {
    'gpt-4o-mini': os.getenv('GPT_API_KEY'),
    'claude-3-5-sonnet-20240620': os.getenv('CLAUDE_API_KEY'),
    'gemini-1.5-flash-8b': os.getenv('GEMINI_API_KEY'),
    'mistral-small-latest': os.getenv('MISTRAL_API_KEY')
}

def check_api_keys():
    missing_keys = [model for model, key in SERVER_API_KEYS.items() if not key]
    if missing_keys:
        app.logger.error(f"Missing API keys for models: {', '.join(missing_keys)}")
        # You might want to disable free summaries or take other actions if keys are missing

# Call check_api_keys
check_api_keys()

def create_tables():
    with app.app_context():
        try:
            db.create_all()
            logger.info("Database tables created successfully")
        except Exception as e:
            if "already exists" in str(e):
                logger.info("Tables already exist, continuing...")
            else:
                logger.error(f"Error creating database tables: {str(e)}")
                raise

# Call create_tables
create_tables()

class SummarizeSchema(Schema):
    input = fields.String(required=True)
    model = fields.String(required=True)
    user_id = fields.String(required=True)
    is_url = fields.Boolean(required=True)

    @validates_schema
    def validate_input(self, data, **kwargs):
        if not data['is_url'] and (len(data['input']) < 100 or len(data['input']) > 50000):
            raise ValidationError('Input length must be between 100 and 50000 characters for non-URL input.')

# Generate a random API key
API_KEY = os.getenv('API_KEY', secrets.token_urlsafe(32))

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        provided_key = request.headers.get('X-API-Key')
        if provided_key and provided_key == API_KEY:
            return f(*args, **kwargs)
        else:
            return jsonify({"error": "Invalid API key"}), 401
    return decorated

@app.route('/', methods=['GET'])
def root():
    return jsonify({"message": "Welcome to the summarizer API. Use /summarize for summarization requests."}), 200

def scrape_content(input_text):
    if input_text.startswith('http://') or input_text.startswith('https://'):
        try:
            response = requests.get(input_text)
            soup = BeautifulSoup(response.text, 'html.parser')
            # Extract text from p, h1, h2, h3, h4, h5, h6 tags
            text = ' '.join([tag.get_text() for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])])
            return text
        except Exception as e:
            app.logger.error(f"Error scraping URL: {str(e)}")
            return None
    else:
        return input_text  # Return the original text if it's not a URL

def format_summary(summary):
    # Ensure the summary is in UTF-8
    summary = summary.encode('utf-8', errors='ignore').decode('utf-8')
    sections = re.split(r'\n(?=\d\.)', summary)
    formatted_sections = []
    
    for section in sections:
        if section.strip():
            # Extract the category title
            title_match = re.match(r'(\d\.\s*[^:]+):', section)
            if title_match:
                title = title_match.group(1)
                content = section[len(title_match.group(0)):].strip()
                
                # Format the content as a list
                points = re.findall(r'[-•🔹]\s*(.+)', content)
                formatted_points = ''.join([f'<li>{point.strip()}</li>' for point in points])
                
                formatted_section = f'<h3>{title}</h3><ul>{formatted_points}</ul>'
                formatted_sections.append(formatted_section)
    
    if not formatted_sections:
        return f'<div class="summary-content">{summary}</div>'
    
    return '<div class="summary-content">' + ''.join(formatted_sections) + '</div>'


def generate_token(user_id):
    try:
        payload = {
            'exp': datetime.now(timezone.utc) + timedelta(days=30),
            'iat': datetime.now(timezone.utc),
            'sub': user_id,
            'user_id': user_id
        }
        return jwt.encode(payload, str(app.config['ENCRYPTION_KEY']), algorithm='HS256')
    except Exception as e:
        app.logger.error(f"Error generating token: {str(e)}")
        return None


@app.route('/verify_token', methods=['POST'])
@token_required
def verify_token(current_user):
    return jsonify({
        'message': 'Token is valid',
        'user_id': current_user.id
    })  

def generate_summary(text, model):
    try:
        app.logger.info(f"Generating summary with model: {model}")
        
        if model == 'claude-3-5-sonnet-20240620':
            client = anthropic.Anthropic(api_key=app.config['CLAUDE_API_KEY'])
            response = client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=1000,
                messages=[
                    {
                        "role": "user",
                        "content": f"""Summarize this privacy policy based on the following categories:
1. What personal data is collected
2. How the data is used
3. Data sharing practices
4. User rights and controls
5. Important policy changes or unique clauses

Present the summary as a bulleted list with clear, concise points. Use emoji indicators for each point. Start each category with the category name on a new line.

Privacy Policy:
{text}"""
                    }
                ]
            )
            summary = response.content[0].text if response.content else None

        elif model == 'gpt-4o-mini':
            client = OpenAI(api_key=app.config['GPT_API_KEY'])
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"""Summarize this privacy policy based on the following categories:
1. What personal data is collected
2. How the data is used
3. Data sharing practices
4. User rights and controls
5. Important policy changes or unique clauses

Present the summary as a bulleted list with clear, concise points. Use emoji indicators for each point. Start each category with the category name on a new line.

Privacy Policy:
{text}"""}],
                max_tokens=1000
            )
            summary = response.choices[0].message.content if response.choices else None

        elif model == 'gemini-1.5-flash-8b':
            genai.configure(api_key=app.config['GEMINI_API_KEY'])
            model = genai.GenerativeModel('gemini-1.5-flash-8b')
            response = model.generate_content(f"""Summarize this privacy policy based on the following categories:
1. What personal data is collected
2. How the data is used
3. Data sharing practices
4. User rights and controls
5. Important policy changes or unique clauses

Present the summary as a bulleted list with clear, concise points. Use emoji indicators for each point. Start each category with the category name on a new line.

Privacy Policy:
{text}""")
            summary = response.text if hasattr(response, 'text') else None

        elif model == 'mistral-small-latest':
            client = MistralClient(api_key=app.config['MISTRAL_API_KEY'])
            messages = [
                ChatMessage(role="user", content=f"""Summarize this privacy policy based on the following categories:
1. What personal data is collected
2. How the data is used
3. Data sharing practices
4. User rights and controls
5. Important policy changes or unique clauses

Present the summary as a bulleted list with clear, concise points. Use emoji indicators for each point. Start each category with the category name on a new line.

Privacy Policy:
{text}""")
            ]
            response = client.chat(
                model="mistral-small-latest",
                messages=messages,
                max_tokens=1000
            )
            summary = response.choices[0].message.content if response.choices else None

        else:
            raise ValueError(f"Unsupported model: {model}")

        app.logger.info(f"Raw API response: {response}")
        app.logger.info(f"Extracted summary: {summary}")

        if summary and len(summary) >= 50:
            return {'summary': summary}
        else:
            raise ValueError("Generated summary is too short or empty")

    except Exception as e:
        app.logger.error(f"Error generating summary with {model}: {str(e)}")
        app.logger.error(f"Full traceback: {traceback.format_exc()}")
        return {'error': str(e)}

@app.route('/summarize', methods=['POST'])
@token_required
def summarize(current_user):
    start_time = time.time()
    data = request.json
    app.logger.info(f"Received summarize request: {data}")  # Log the entire request data

    input_text = data.get('input')
    model = data.get('model')
    is_url = data.get('is_url', False)
    api_key = data.get('api_key')
    session_id = data.get('session_id')

    app.logger.info(f"Summarize request - Model: {model}, API Key: {'User provided' if api_key else 'None'}, Is URL: {is_url}")
    
    try:
        # Validate input
        if not input_text:
            app.logger.warning("Missing input text in the request")
            return jsonify({'error': 'Missing input text'}), 400
        if not model:
            app.logger.warning("Missing model in the request")
            return jsonify({'error': 'Missing model'}), 400

        app.logger.info(f"Received request - Input: {input_text[:50]}..., Model: {model}, Is URL: {is_url}")

        # Determine which API key to use
        if current_user.summaries_left > 0:
            api_key = SERVER_API_KEYS.get(model)
            if not api_key:
                app.logger.error(f"Server-side API key not found for model: {model}")
                return jsonify({'error': 'Server configuration error'}), 500
            app.logger.info("Using server-side API key")
        elif api_key:
            app.logger.info("Using user-provided API key")
        else:
            app.logger.warning(f"User {current_user.id} has no free summaries left and no valid API key provided")
            return jsonify({'error': 'No free summaries left and no valid API key provided'}), 403

        # If it's a URL, we need to scrape it
        if is_url:
            app.logger.info(f"Attempting to scrape URL: {input_text}")
            scraped_content = scrape_website_content(input_text)
            if not scraped_content:
                return jsonify({"error": "Failed to scrape website content. Please try pasting the text directly."}), 400
            input_text = scraped_content
            app.logger.info(f"Successfully scraped content. Length: {len(input_text)}")
            
            # For scraped content, we'll allow a much larger size, but still limit it
            if len(input_text) > 500000:  # Adjust this limit as needed
                app.logger.warning(f"Scraped content too long: {len(input_text)} characters")
                return jsonify({'error': 'Scraped content too long. Please try a different URL or enter text manually.'}), 400
        else:
            # Validate input length for user-entered text
            if len(input_text) < 100:
                app.logger.warning(f"Input text too short: {len(input_text)} characters")
                return jsonify({'error': 'Input text too short. Please provide at least 100 characters.'}), 400
            if len(input_text) > 50000:
                app.logger.warning(f"Input text too long: {len(input_text)} characters")
                return jsonify({'error': 'Input text too long. Please provide no more than 50,000 characters.'}), 400

        # Log before starting summarization
        app.logger.info(f"Starting summarization with model: {model}")

        # Perform summarization
        try:
            app.logger.info(f"Generating summary using model: {model}")
            summary_result = generate_summary(input_text, model)
            if 'error' in summary_result:
                app.logger.error(f"Error in generate_summary: {summary_result['error']}")
                return jsonify({'error': summary_result['error']}), 500
            
            summary = summary_result['summary']
            formatted_summary = format_summary(summary)
            
            # Log after summarization
            app.logger.info(f"Summarization completed. Summary length: {len(formatted_summary)}")

            # Decrease the number of free summaries only if using server-side key
            if current_user.summaries_left > 0:
                current_user.summaries_left -= 1
            
            searched_data = request.json.get('searched_data') or request.json.get('input', '')
            encrypted_searched_data = encrypt_data(searched_data)
            # Record user activity
            end_time = time.time()
            execution_time = round(end_time - start_time, 3)
            session_id = str(uuid.uuid4())
            user_activity = UserActivity(
                user_id=current_user.id,
                session_id=session_id,
                model_selected=model,
                searched_for='url' if is_url else 'text',
                searched_data=encrypt_data(searched_data),
                scrape_data=encrypt_data(input_text) if is_url else None,
                request_time=execution_time,
                created_at=datetime.now(timezone.utc)
            )
            db.session.add(user_activity)
            db.session.commit()

            app.logger.info(f"Summarization process completed. Execution time: {execution_time:.2f} seconds")
            app.logger.info(f"Summary generated successfully for user: {current_user.id}. Summaries left: {current_user.summaries_left}")
            return jsonify({
                'summary': formatted_summary,
                'execution_time': execution_time,
                'free_summaries_left': current_user.summaries_left,
                'activity_id': user_activity.id
            })
        except Exception as e:
            app.logger.error(f"Error generating summary: {str(e)}")
            return jsonify({'error': 'Failed to generate summary'}), 500

    except Exception as e:
        app.logger.error(f"Error in summarize: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/user_activity', methods=['POST'])
@token_required
def user_activity(current_user):
    try:
        data = request.json
        app.logger.info(f"Received user activity data for user {current_user.id}")
        
        session_id = data.get('session_id')
        activity = UserActivity.query.filter_by(session_id=session_id).first()
        
        if activity:
            # Update existing activity
            if data.get('tts_used') is not None:
                activity.tts_used = data.get('tts_used')
            if data.get('copied_summary') is not None:
                activity.copied_summary = data.get('copied_summary')
            if data.get('summary_exported') is not None:
                activity.summary_exported = data.get('summary_exported')
            
            # Update other fields if they weren't set initially
            if not activity.model_selected and data.get('model_selected'):
                activity.model_selected = data.get('model_selected')
            if not activity.searched_for and data.get('searched_for'):
                activity.searched_for = data.get('searched_for')
            if not activity.searched_data and data.get('searched_data'):
                activity.searched_data = encrypt_data(data.get('searched_data', ''))
            if not activity.scrape_data and data.get('scrape_data'):
                activity.scrape_data = encrypt_data(data.get('scrape_data', ''))
            if not activity.request_time and data.get('request_time'):
                activity.request_time = round(float(data.get('request_time', 0)), 3)
            
            app.logger.info(f"Updated existing activity for session {session_id}")
        else:
            # Create new activity
            activity = UserActivity(
                user_id=current_user.id,
                session_id=session_id,
                tts_used=data.get('tts_used'),
                copied_summary=data.get('copied_summary'),
                summary_exported=data.get('summary_exported'),
                model_selected=data.get('model_selected'),
                searched_for=data.get('searched_for'),
                searched_data=encrypt_data(data.get('searched_data', '')),
                scrape_data=encrypt_data(data.get('scrape_data', '')),
                request_time=round(float(data.get('request_time', 0)), 3)
            )
            db.session.add(activity)
            app.logger.info(f"Created new activity for session {session_id}")

        db.session.commit()
        return jsonify({"message": "User activity updated successfully"}), 200

    except Exception as e:
        app.logger.error(f"Error updating user activity for user {current_user.id}: {str(e)}")
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

def scrape_website_content(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com/'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try multiple selectors to find the main content
        content = soup.find('div', {'class': 'privacy-policy'}) or \
                  soup.find('main') or \
                  soup.find('article') or \
                  soup.body
        
        if content:
            text = content.get_text(separator=' ', strip=True)
        else:
            text = soup.get_text(separator=' ', strip=True)
        
        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            app.logger.warning(f"No content scraped from {url}")
            return None
        app.logger.info(f"Successfully scraped {len(text)} characters from {url}")
        return text
    except Exception as e:
        app.logger.error(f"Error scraping website {url}: {str(e)}")
        return None

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(f"Unhandled exception occurred: {str(e)}")
    return jsonify({"error": "An unexpected error occurred"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

csp = {
    'default-src': "'self'",
    'script-src': ["'self'", "'unsafe-inline'", "https://cdnjs.cloudflare.com"],
    'style-src': ["'self'", "'unsafe-inline'", "https://fonts.googleapis.com"],
    'font-src': ["'self'", "https://fonts.gstatic.com"],
    'img-src': ["'self'", "data:", "https:"],
    'connect-src': ["'self'", "your_server_url"],
}

Talisman(app, content_security_policy=csp, content_security_policy_nonce_in=['script-src'])

def sanitize_input(text):
    # Normalize Unicode characters
    normalized_text = unicodedata.normalize('NFKD', text)
    # Remove non-ASCII characters
    ascii_text = normalized_text.encode('ascii', 'ignore').decode('ascii')
    # Limit input length
    truncated_text = ascii_text[:5000]  # Adjust the limit as needed
    return bleach.clean(truncated_text)

@app.route('/new-user', methods=['POST'])
def new_user():
    try:
        data = request.json
        user_id = data.get('userId')
        fingerprint = data.get('fingerprint')
        ip_address = request.remote_addr

        app.logger.info(f"Received new user request - User ID: {user_id}, Fingerprint: {fingerprint}, IP: {ip_address}")

        if not fingerprint:
            app.logger.warning("Missing fingerprint in request")
            return jsonify({"error": "Fingerprint is required"}), 400

        # First, try to find user by fingerprint
        user = User.query.filter_by(fingerprint=fingerprint).first()

        if user:
            app.logger.info(f"User found by fingerprint - User ID: {user.id}")
            # Update user information
            user.last_ip = ip_address
            if user_id and user.id != user_id:
                app.logger.info(f"Updating user ID from {user.id} to {user_id}")
                user.id = user_id
            db.session.commit()
            return jsonify({"success": True, "user_id": user.id, "new_user": False, "summaries_left": user.summaries_left}), 200

        # If not found by fingerprint, check by user_id
        if user_id:
            user = User.query.get(user_id)
            if user:
                app.logger.info(f"User found by user_id - User ID: {user.id}")
                # Update user information
                user.fingerprint = fingerprint
                user.last_ip = ip_address
                db.session.commit()
                return jsonify({"success": True, "user_id": user.id, "new_user": False, "summaries_left": user.summaries_left}), 200

        # If still not found, create new user
        new_user_id = user_id or str(uuid.uuid4())
        new_user = User(id=new_user_id, fingerprint=fingerprint, last_ip=ip_address, summaries_left=10)
        db.session.add(new_user)
        db.session.commit()

        app.logger.info(f"New user created successfully - User ID: {new_user_id}")
        return jsonify({"success": True, "user_id": new_user_id, "new_user": True, "summaries_left": 10}), 201

    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error processing user request: {str(e)}")
        return jsonify({"error": "An error occurred while processing the user"}), 500

@app.route('/get_free_summaries_count', methods=['POST'])
@token_required
def get_free_summaries_count(current_user):
    logger.info(f"Fetching free summaries count for user: {current_user.id}")
    return jsonify({"free_summaries_left": current_user.summaries_left})


@app.route('/get_tts_endpoint', methods=['GET'])
def get_tts_endpoint():
    return jsonify({"tts_endpoint": TTS_ENDPOINT_URL})

@app.route('/tts', methods=['POST'])
def text_to_speech():
    try:
        app.logger.info("TTS request received")
        data = request.json
        text = data.get('text', '')
        voice = data.get('voice', 'nova')

        app.logger.info(f"TTS request - Text length: {len(text)}, Voice: {voice}")

        if not text or len(text) > 4096:
            app.logger.warning(f"Invalid text provided. Length: {len(text)}")
            return jsonify({"error": "Invalid text provided"}), 400

        client = OpenAI(api_key=os.getenv('GPT_API_KEY'))
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text
        )

        app.logger.info("TTS audio generated successfully")

        audio_data = response.content
        custom_response = Response(audio_data, mimetype="audio/mpeg")
        custom_response.headers.set('Content-Disposition', 'attachment', filename='speech.mp3')

        return custom_response

    except Exception as e:
        app.logger.error(f"Error in TTS: {str(e)}")
        return jsonify({"error": str(e)}), 500

# app.logger = create_logger(app)
# app.logger.setLevel(logging.INFO)

# @app.before_request
# def log_request_info():
#     app.logger.info('Headers: %s', request.headers)
#     app.logger.info('Body: %s', request.get_data())

# @app.after_request
# def log_response_info(response):
#     app.logger.info('Response: %s', response.get_data())
#     return response

@app.route('/get_user_info', methods=['POST'])
@token_required
def get_user_info(current_user):
    # Check if the user was just created (new user)
    new_user = current_user.created_at == current_user.last_token_refresh

    return jsonify({
        'user_id': current_user.id,
        'summaries_left': current_user.summaries_left,
        'created_at': current_user.created_at.isoformat() if current_user.created_at else None,
        'new_user': new_user
    })


@app.route('/get_or_create_token', methods=['POST'])
def get_or_create_token():
    try:
        data = request.json
        user_id = data.get('user_id')
        fingerprint = data.get('fingerprint')
        
        if not user_id and not fingerprint:
            return jsonify({'error': 'Either User ID or fingerprint is required'}), 400
        
        current_time = datetime.now(timezone.utc)
        new_user = False

        try:
            if fingerprint:
                user = User.query.filter_by(fingerprint=fingerprint).first()
            if not user and user_id:
                user = User.query.get(user_id)
            
            if not user:
                if not user_id:
                    user_id = str(uuid.uuid4())
                user = User(id=user_id, fingerprint=fingerprint, last_token_refresh=current_time)
                db.session.add(user)
                db.session.flush()
                new_user = True
            else:
                # Update user information
                if fingerprint and user.fingerprint != fingerprint:
                    user.fingerprint = fingerprint
                if user_id and user.id != user_id:
                    user.id = user_id
                user.last_token_refresh = current_time

            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            # If IntegrityError occurs, the user was created by another concurrent request
            user = User.query.filter((User.id == user_id) | (User.fingerprint == fingerprint)).first()
            if user:
                user.last_token_refresh = current_time
                db.session.commit()
            else:
                return jsonify({'error': 'Failed to create or retrieve user'}), 500
        
        token = generate_token(user.id)
        return jsonify({'token': token, 'new_user': new_user, 'user_id': user.id})
    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error in get_or_create_token: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/scrape', methods=['POST'])
@token_required
def scrape_website(current_user):  # Add current_user parameter
    url = request.json.get('url')
    if not url:
        return jsonify({'error': 'URL is required'}), 400

    try:
        app.logger.info(f"Attempting to scrape URL: {url}")
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        text = ' '.join([p.get_text() for p in soup.find_all('p')])
        
        app.logger.info(f"Successfully scraped {len(text)} characters from {url}")
        return jsonify({'content': text})
    except Exception as e:
        app.logger.error(f"Error scraping website {url}: {str(e)}")
        return jsonify({'error': str(e)}), 500

def encrypt_data(data):
    if data is None:
        return None
    return cipher_suite.encrypt(data.encode()).decode()

def decrypt_data(encrypted_data):
    if encrypted_data is None:
        return None
    return cipher_suite.decrypt(encrypted_data.encode()).decode()

if __name__ == '__main__':
    logger.info("Starting Flask application in debug mode...")
    app.run()
