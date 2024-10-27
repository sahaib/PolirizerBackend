"""
Summarizer API Backend
---------------------
A Flask application that provides text summarization services with caching support.

Key Features:
- Text and URL summarization
- Multiple AI model support (Claude, GPT, Gemini, Mistral)
- Memcached integration for content caching
- JWT-based authentication
- Rate limiting
- Error handling and logging

Environment Variables Required:
- MEMCACHIER_SERVERS: Comma-separated list of cache servers
- MEMCACHIER_USERNAME: Cache authentication username
- MEMCACHIER_PASSWORD: Cache authentication password
- ENCRYPTION_KEY: JWT encryption key
- Various AI model API keys (ANTHORIPIC_API_KEY, OPENAI_API_KEY, etc.)
"""

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
from flask.logging import default_handler
from flask import g
import ipaddress
import json
import hashlib
import bmemcached


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

def init_cache_client():
    """Initialize and verify cache client"""
    try:
        servers = os.environ.get('MEMCACHIER_SERVERS', '')
        username = os.environ.get('MEMCACHIER_USERNAME', '')
        password = os.environ.get('MEMCACHIER_PASSWORD', '')
        
        app.logger.info(f"Initializing cache client with servers: {servers}")
        
        if not all([servers, username, password]):
            app.logger.error("Missing cache credentials")
            return None
            
        # Create client using bmemcached instead of pylibmc
        client = bmemcached.Client(
            servers.split(','),
            username=username,
            password=password
        )
        
        # Test connection
        try:
            test_key = f"init_test_{uuid.uuid4()}"
            client.set(test_key, "test", time=60)
            result = client.get(test_key)
            client.delete(test_key)
            
            if result == "test":
                app.logger.info("Cache client initialized and verified successfully")
                return client
            else:
                raise Exception("Cache verification failed")
                
        except Exception as e:
            app.logger.error(f"Cache test failed: {str(e)}")
            return None
            
    except Exception as e:
        app.logger.error(f"Cache initialization failed: {str(e)}")
        return None

# Initialize cache client ONCE at startup
cache_client = init_cache_client()


if not cache_client:
    app.logger.error("⚠️ Failed to initialize cache client - application may not function correctly")
    # Log environment variables (safely)
    app.logger.error(f"MEMCACHIER_SERVERS configured: {'Yes' if os.environ.get('MEMCACHIER_SERVERS') else 'No'}")
    app.logger.error(f"MEMCACHIER_USERNAME configured: {'Yes' if os.environ.get('MEMCACHIER_USERNAME') else 'No'}")
    app.logger.error(f"MEMCACHIER_PASSWORD configured: {'Yes' if os.environ.get('MEMCACHIER_PASSWORD') else 'No'}")

@app.before_request
def start_timer():
    g.start = time.time()
    app.logger.debug(f"start_timer called for path: {request.path}")

# Set up API keys
app.config['ANTHORIPIC_API_KEY'] = os.getenv('ANTHORIPIC_API_KEY')
app.config['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
app.config['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
app.config['MISTRAL_API_KEY'] = os.getenv('MISTRAL_API_KEY')

# Verify API keys are set
if not app.config['ANTHORIPIC_API_KEY']:
    app.logger.error("ANTHORIPIC_API_KEY is not set in the environment variables")
if not app.config['GOOGLE_API_KEY']:
    app.logger.error("GOOGLE_API_KEY is not set in the environment variables")
if not app.config['OPENAI_API_KEY']:
    app.logger.error("OPENAI_API_KEY is not set in the environment variables")
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
    id = db.Column(db.String(36), primary_key=True, index=True)
    fingerprint = db.Column(db.String(255), nullable=False, default='unknown')
    last_ip = db.Column(db.String(45), nullable=False, default='unknown')
    summaries_left = db.Column(db.Integer, default=10)
    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    last_token_refresh = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    activities = db.relationship('UserActivity', backref='user', lazy=True)

class UserActivity(db.Model):
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey('user.id'), nullable=False, index=True)
    session_id = db.Column(db.String(36), nullable=True, index=True)
    model_selected = db.Column(db.String(50), nullable=False)
    searched_for = db.Column(db.String(10), nullable=False)
    searched_data = db.Column(db.Text, nullable=False)
    scrape_data = db.Column(db.Text)
    request_time = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now(timezone.utc))
    tts_used = db.Column(db.Boolean, default=False)
    copied_summary = db.Column(db.Boolean, default=False)
    summary_exported = db.Column(db.Boolean, default=False)
    api_key_type = db.Column(db.String(20))

# Define constants
HUMAN_PROMPT = "\n\nHuman: "
AI_PROMPT = "\n\nAssistant: "

# Set up CORS
CORS(app, resources={
    r"/*": {
        "origins": [
            "your_server_url",
            "chrome-extension://your_extension_id"
        ]
    }
})

# Set up rate limiting
def get_remote_address():
    cf_connecting_ip = request.headers.get('CF-Connecting-IP')
    if cf_connecting_ip:
        return cf_connecting_ip
    return request.remote_addr
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["1000 per day", "300 per hour", "100 per minute"],
    storage_uri="memory://"
)

# Add this near the top of the file, where other environment variables are loaded
TTS_ENDPOINT_URL = os.getenv('TTS_ENDPOINT_URL', 'https://your_server_url/tts')

# Set up API keys
SERVER_API_KEYS = {
    'gpt-4o-mini': os.getenv('OPENAI_API_KEY'),
    'claude-3-5-sonnet-20240620': os.getenv('ANTHORIPIC_API_KEY'),
    'gemini-1.5-flash-8b': os.getenv('GOOGLE_API_KEY'),
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
    """Format summary with consistent encoding and styling"""
    if not summary:
        return ''
    
    # Handle bidirectional Unicode and encoding issues
    try:
        # Step 1: Remove any bidirectional Unicode control characters
        bidi_controls = {
            '\u200E': '',  # Left-to-Right Mark
            '\u200F': '',  # Right-to-Left Mark
            '\u202A': '',  # Left-to-Right Embedding
            '\u202B': '',  # Right-to-Left Embedding
            '\u202C': '',  # Pop Directional Formatting
            '\u202D': '',  # Left-to-Right Override
            '\u202E': '',  # Right-to-Left Override
            '\u2066': '',  # Left-to-Right Isolate
            '\u2067': '',  # Right-to-Left Isolate
            '\u2068': '',  # First Strong Isolate
            '\u2069': ''   # Pop Directional Isolate
        }
        
        # Convert to string and remove BOM
        if not isinstance(summary, str):
            summary = str(summary)
        summary = summary.replace('\ufeff', '')
        
        # Remove bidirectional control characters
        for char, replacement in bidi_controls.items():
            summary = summary.replace(char, replacement)
        
        # Normalize to NFC form (canonical decomposition followed by canonical composition)
        summary = unicodedata.normalize('NFC', summary)
        
        # Handle common UTF-8 encoding issues
        summary = summary.encode('ascii', errors='ignore').decode('ascii')
        
        # Clean markdown headers (using ASCII-safe characters)
        summary = re.sub(r'^###\s*(.+?)$', r'<h3>\1</h3>', summary, flags=re.MULTILINE)
        
    except Exception as e:
        app.logger.error(f"Encoding cleanup failed: {str(e)}")
        # Fallback to strict ASCII
        summary = ''.join(c for c in str(summary) if ord(c) < 128)
    
    sections = re.split(r'\n(?=\d\.)', summary)
    formatted_sections = []
    
    for section in sections:
        if section.strip():
            # Extract the category title
            title_match = re.match(r'(\d\.\s*[^:]+):', section)
            if title_match:
                title = title_match.group(1)
                content = section[len(title_match.group(0)):].strip()
                
                # Improved bullet point detection for main points and sub-points
                points = []
                current_point = []
                
                for line in content.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Check if line starts with bullet or dash
                    if re.match(r'^[•-]', line):
                        if current_point:
                            points.append('\n'.join(current_point))
                            current_point = []
                        current_point.append(line.lstrip('•- '))
                    # Check if line is indented (sub-point)
                    elif line.startswith('    ') or line.startswith('\t'):
                        if current_point:
                            current_point.append(f"<ul><li>{line.strip()}</li></ul>")
                    else:
                        if current_point:
                            points.append('\n'.join(current_point))
                            current_point = []
                        current_point.append(line)
                
                if current_point:
                    points.append('\n'.join(current_point))
                
                formatted_points = ''.join([f'<li>{point}</li>' for point in points if point.strip()])
                formatted_section = f'<h3>{title}</h3><ul>{formatted_points}</ul>'
                formatted_sections.append(formatted_section)
            else:
                # Handle sections without numbers
                formatted_sections.append(f'<p>{section.strip()}</p>')
    
    # If no sections were formatted, return the original content in a div
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
@limiter.limit("30 per minute")
def verify_token(current_user):
    return jsonify({
        'message': 'Token is valid',
        'user_id': current_user.id
    })  

def generate_summary(text, model):
    try:
        app.logger.info(f"Generating summary with model: {model}")
        
        if model == 'claude-3-5-sonnet-20240620':
            client = anthropic.Anthropic(api_key=app.config['ANTHORIPIC_API_KEY'])
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
            client = OpenAI(api_key=app.config['OPENAI_API_KEY'])
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
            genai.configure(api_key=app.config['GOOGLE_API_KEY'])
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



def handle_memcached_error(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            app.logger.error(f"Memcached error in {func.__name__}: {str(e)}")
            # Attempt to reinitialize client
            global cache_client
            cache_client = init_cache_client()
            return None
    return wrapper
# cache helper functions
def cache_content_data(key, data):
    """
    Store content data in cache with retry logic.
    
    Args:
        key: Cache key for storing the data
        data: Dictionary containing content and metadata
        
    Returns:
        tuple: (success: bool, error_details: dict)
        
    Cache Data Structure:
        - content: The actual content to cache
        - last_updated: Content's last update timestamp
        - cached_at: Cache storage timestamp
        - is_plain_text: Boolean flag for content type
        - url: Original URL if applicable
        - cache_version: Schema version for future compatibility
    """
    try:
        if not cache_client:
            app.logger.error("Cache client not initialized")
            return False, {}
            
        cache_data = {
            'content': data.get('content'),
            'last_updated': data.get('last_updated'),
            'cached_at': datetime.now(timezone.utc).isoformat(),
            'is_plain_text': data.get('is_plain_text', False),
            'url': data.get('url'),
            'cache_version': '1.1'
        }
        
        # Add retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                success = cache_client.set(key, json.dumps(cache_data), time=7776000)
                if success:
                    app.logger.info(f"Cache storage successful for key {key}")
                    return True, {}
            except Exception as e:
                app.logger.warning(f"Cache attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                
        return False, {}
        
    except Exception as e:
        app.logger.error(f"Cache error for key {key}: {str(e)}")
        return False, {}

def get_cached_content(key, url=None):
    """Retrieve cached content with smart update checking"""
    try:
        cached_data = cache_client.get(key)
        if cached_data:
            data = json.loads(cached_data)
            cached_at = datetime.fromisoformat(data.get('cached_at', ''))
            cache_age = (datetime.now(timezone.utc) - cached_at).days
            
            app.logger.info(f"Cache hit for key: {key}, Age: {cache_age} days")
            app.logger.debug(f"Cache metadata - URL: {data.get('url')}, Version: {data.get('cache_version')}")
            
            if url:
                should_check = (
                    cache_age > 90 or
                    (cache_age > 7 and has_recent_update(url, cached_at))
                )
                
                if should_check:
                    app.logger.info(f"Checking for updates - URL: {url}, Cache Age: {cache_age} days")
                    new_content = scrape_website_content(url)
                    if new_content:
                        new_last_updated = extract_last_updated_date(new_content)
                        
                        content_changed = new_content != data.get('content')
                        date_updated = new_last_updated and new_last_updated > data.get('last_updated')
                        
                        app.logger.info(f"Update check results - Content Changed: {content_changed}, Date Updated: {date_updated}")
                        
                        if content_changed or date_updated or cache_age > 90:
                            app.logger.info(f"Updating cache - Reason: {'Content Changed' if content_changed else 'Date Updated' if date_updated else 'Cache Expired'}")
                            new_data = {
                                'content': new_content,
                                'last_updated': new_last_updated,
                                'cached_at': datetime.now(timezone.utc).isoformat(),
                                'is_plain_text': False,
                                'url': url,
                                'previous_cache_date': data.get('cached_at')
                            }
                            cache_result = cache_content_data(key, new_data)
                            app.logger.info(f"Cache update status: {'Success' if cache_result[0] else 'Failed'}")
                            return new_data
            
            app.logger.info(f"Using existing cached content - Age: {cache_age} days")
            return data
        else:
            app.logger.info(f"Cache miss for key: {key}")
            return None
    except Exception as e:
        app.logger.error(f"Cache retrieval error for key {key}: {str(e)}")
        return None
    
    
def has_recent_update(url, cached_at):
    """Check if the page has a recent update date"""
    try:
        # Quick check of the page header/metadata without full scrape
        response = requests.head(url, allow_redirects=True)
        last_modified = response.headers.get('last-modified')
        if last_modified:
            last_modified_date = datetime.strptime(last_modified, '%a, %d %b %Y %H:%M:%S GMT')
            return last_modified_date > cached_at
        return False
    except Exception as e:
        app.logger.error(f"Error checking for recent updates: {str(e)}")
        return False

def get_cache_key(url_or_text, is_url=True):
    """Generate a unique and consistent cache key"""
    if is_url:
        # Normalize URL by removing trailing slashes and query parameters
        normalized_url = re.sub(r'\?.*$', '', url_or_text.rstrip('/').lower())
        return f"privacy_policy_url_{hash(normalized_url)}"
    return f"privacy_policy_text_{hash(url_or_text)}"

def force_cache_refresh(key):
    """Force refresh of cached content"""
    try:
        cache_client.delete(key)
        app.logger.info(f"Forced cache refresh for key: {key}")
        return True
    except Exception as e:
        app.logger.error(f"Cache refresh error: {str(e)}")
        return False

def get_summary(text: str, model: str, api_key: str = None) -> dict:
    """Get summary without caching for all models"""
    try:
        # Set API key in config
        if api_key:
            if model == 'claude-3-5-sonnet-20240620':
                app.config['ANTHORIPIC_API_KEY'] = api_key
            elif model == 'gpt-4o-mini':
                app.config['OPENAI_API_KEY'] = api_key
            elif model == 'gemini-1.5-flash-8b':
                app.config['GOOGLE_API_KEY'] = api_key
            elif model == 'mistral-small-latest':
                app.config['MISTRAL_API_KEY'] = api_key

        summary_result = generate_summary(text, model)
        if 'error' in summary_result:
            raise Exception(summary_result['error'])
            
        # Format summary and return without caching
        formatted_summary = format_summary(summary_result['summary'])
        
        return {
            'summary': formatted_summary,
            'cached': False
        }
        
    except Exception as e:
        app.logger.error(f"❌ Summary generation failed: {str(e)}")
        raise

@app.route('/summarize', methods=['POST'])
@token_required
@limiter.limit("50 per minute")
def summarize(current_user):
    start_time = time.time()
    
    # Keep all existing input validation code
    if not request.is_json:
        app.logger.error("Request is not JSON")
        return jsonify({"error": "Request must be JSON"}), 400
        
    data = request.get_json(silent=True)
    if not data:
        app.logger.error("Empty or invalid JSON data received")
        return jsonify({"error": "Invalid or empty JSON data"}), 400

    app.logger.info(f"Received summarize request: {data}")

    # Keep existing value extraction
    input_text = data.get('input')
    model = data.get('model')
    is_url = data.get('is_url', False)
    api_key = data.get('api_key')
    session_id = data.get('session_id')

    # Keep existing validation
    if not input_text or not model:
        app.logger.warning("Missing required fields")
        return jsonify({"error": "Input text and model selection are required"}), 400

    app.logger.info(f"Summarize request - Model: {model}, API Key: {'User provided' if api_key else 'None'}, Is URL: {is_url}")
    
    try:
        # Keep API key determination logic
        if current_user.summaries_left > 0:
            api_key = SERVER_API_KEYS.get(model)
            api_key_type = 'SERVER_KEY'
            if not api_key:
                app.logger.error(f"Server-side API key not found for model: {model}")
                return jsonify({'error': 'Server configuration error'}), 500
            app.logger.info("Using server-side API key")
        elif api_key:
            api_key_type = 'USER_KEY'
            app.logger.info("Using user-provided API key")
        else:
            app.logger.warning(f"User {current_user.id} has no free summaries left and no valid API key provided")
            return jsonify({'error': 'No free summaries left and no valid API key provided'}), 403

        # Process input with content caching
        content_cache_key = get_cache_key(input_text, is_url=is_url)
        cached_data = get_cached_content(content_cache_key)
        
        if is_url:
            if cached_data:
                app.logger.info(f"🎯 Cache hit for URL: {input_text}")
                input_text = cached_data.get('content')
                last_updated = cached_data.get('last_updated')
                cache_age = (datetime.now(timezone.utc) - datetime.fromisoformat(cached_data.get('cached_at'))).days
                app.logger.info(f"📅 Using cached content. Age: {cache_age} days")
            else:
                app.logger.info(f"🔍 Cache miss for URL: {input_text}")
                scraped_content = scrape_website_content(input_text)
                if not scraped_content:
                    return jsonify({"error": "Failed to scrape website content. Please try pasting the text directly."}), 400
                
                input_text = scraped_content
                last_updated = extract_last_updated_date(input_text)
                
                cache_content_data(content_cache_key, {
                    'content': input_text,
                    'last_updated': last_updated,
                    'is_plain_text': False,
                    'url': input_text,
                    'cached_at': datetime.now(timezone.utc).isoformat()
                })
            
            if len(input_text) > 500000:
                return jsonify({'error': 'Scraped content too long. Please try a different URL or enter text manually.'}), 400
        else:
            # Handle plain text with length validation
            if len(input_text) < 100:
                return jsonify({'error': 'Input text too short. Please provide at least 100 characters.'}), 400
            if len(input_text) > 50000:
                return jsonify({'error': 'Input text too long. Please provide no more than 50,000 characters.'}), 400

            # Cache plain text if not cached
            if not cached_data:
                cache_content_data(content_cache_key, {
                    'content': input_text,
                    'last_updated': None,
                    'is_plain_text': True,
                    'cached_at': datetime.now(timezone.utc).isoformat()
                })

        # Generate fresh summary (no caching)
        app.logger.info(f"🤖 Starting summarization with model: {model}")
        try:
            summary_result = get_summary(input_text, model, api_key)
            if isinstance(summary_result, dict) and 'error' in summary_result:
                app.logger.error(f"❌ Error in get_summary: {summary_result['error']}")
                return jsonify({'error': summary_result['error']}), 500
            
            formatted_summary = format_summary(summary_result['summary'])
            
            # Keep existing user activity tracking
            if current_user.summaries_left > 0:
                current_user.summaries_left -= 1
            
            searched_data = request.json.get('searched_data') or request.json.get('input', '')
            encrypted_searched_data = encrypt_data(searched_data)
            
            execution_time = round(time.time() - start_time, 3)
            user_activity = UserActivity(
                user_id=current_user.id,
                session_id=session_id or str(uuid.uuid4()),
                model_selected=model,
                searched_for='url' if is_url else 'text',
                searched_data=encrypt_data(searched_data),
                scrape_data=encrypt_data(input_text) if is_url else None,
                request_time=execution_time,
                created_at=datetime.now(timezone.utc),
                api_key_type=api_key_type
            )
            db.session.add(user_activity)
            db.session.commit()

            response_data = {
                'summary': formatted_summary,
                'execution_time': execution_time,
                'free_summaries_left': current_user.summaries_left,
                'activity_id': user_activity.id,
                'cached': False,  # Summary is never cached
                'content_cached': True if cached_data else False  # Indicate if input content was cached
            }

            app.logger.info("✅ Summarization completed successfully")
            return jsonify(response_data)

        except Exception as e:
            app.logger.error(f"❌ Error generating summary: {str(e)}")
            return jsonify({'error': 'Failed to generate summary'}), 500

    except Exception as e:
        app.logger.error(f"❌ Error in summarize: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/user_activity', methods=['POST'])
@token_required
@limiter.limit("100 per minute")
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
    """Scrape website content with caching"""
    cache_key = f"scrape_{hashlib.md5(url.encode()).hexdigest()}"
    
    # Try cache first
    cached_content = cache_get(cache_key)
    if cached_content:
        app.logger.info(f"📄 Cache HIT: Using cached content for URL: {url}")
        return cached_content
        
    app.logger.info(f"🌐 Cache MISS: Scraping new content for URL: {url}")
    start_time = time.time()
    
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
            
        scrape_time = time.time() - start_time
        app.logger.info(f"⏱️ Scraping completed in {scrape_time:.3f}s")
        
        # Cache the result
        cache_set(cache_key, text, timeout=7776000)  # Cache for 90 days
        
        app.logger.info(f"Successfully scraped {len(text)} characters from {url}")
        return text
        
    except Exception as e:
        app.logger.error(f"Error scraping website {url}: {str(e)}")
        return None

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

def handle_cloudflare_error(e):
    app.logger.error(f"Cloudflare error: {str(e)}")
    response = jsonify({
        'error': 'Service temporarily unavailable',
        'cf_ray': request.headers.get('CF-Ray', 'unknown')
    })
    response.headers['Cache-Control'] = 'no-store'
    return response, 503

@app.errorhandler(500)
def internal_server_error(e):
    cf_ray = request.headers.get('CF-Ray', 'unknown')
    app.logger.error(f"Internal error (CF-Ray: {cf_ray}): {str(e)}")
    return handle_cloudflare_error(e)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

csp = {
    'default-src': "'self'",
    'script-src': ["'self'", "'unsafe-inline'", "https://cdnjs.cloudflare.com"],
    'style-src': ["'self'", "'unsafe-inline'", "https://fonts.googleapis.com"],
    'font-src': ["'self'", "https://fonts.gstatic.com"],
    'img-src': ["'self'", "data:", "https:"],
    'connect-src': ["'self'", "https://your_server_url"],
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
@limiter.limit("30 per minute")
def get_free_summaries_count(current_user):
    logger.info(f"Fetching free summaries count for user: {current_user.id}")
    return jsonify({"free_summaries_left": current_user.summaries_left})


# Update the TTS endpoint to handle potential Cloudflare redirects
@app.route('/tts', methods=['POST'])
@limiter.limit("30 per minute")
def text_to_speech():
    app.logger.info(f"TTS request headers: {request.headers}")
    try:
        app.logger.info("TTS request received")
        data = request.json
        text = data.get('text', '')
        voice = data.get('voice', 'nova')

        app.logger.info(f"TTS request - Text length: {len(text)}, Voice: {voice}")

        if not text or len(text) > 4096:
            app.logger.warning(f"Invalid text provided. Length: {len(text)}")
            return jsonify({"error": "Invalid text provided"}), 400

        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text
        )

        app.logger.info("TTS audio generated successfully")

        audio_data = response.content
        custom_response = Response(audio_data, mimetype="audio/mpeg")
        custom_response.headers.set('Content-Disposition', 'attachment', filename='speech.mp3')
        
        # Add Cache-Control header to prevent Cloudflare caching
        custom_response.headers.set('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')

        return custom_response

    except Exception as e:
        app.logger.error(f"Error in TTS: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Update the get_tts_endpoint function to include a version number
@app.route('/get_tts_endpoint', methods=['GET'])
def get_tts_endpoint():
    version = "1.0"  # Increment this when you make changes to the TTS endpoint
    return jsonify({"tts_endpoint": f"{TTS_ENDPOINT_URL}?v={version}"})

# app.logger = create_logger(app)
# app.logger.setLevel(logging.INFO)

@app.route('/get_user_info', methods=['POST'])
@token_required
@limiter.limit("50 per minute")
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
@limiter.limit("20 per minute")
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
@limiter.limit("50 per minute")
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

try:
    cloudflare_response = requests.get('https://api.cloudflare.com/client/v4/ips')
    if cloudflare_response.status_code == 200:
        CLOUDFLARE_IPS = cloudflare_response.json()['result']
    else:
        app.logger.error(f"Failed to fetch Cloudflare IPs. Status code: {cloudflare_response.status_code}")
        CLOUDFLARE_IPS = {'ipv4_cidrs': [], 'ipv6_cidrs': []}
except Exception as e:
    app.logger.error(f"Error fetching Cloudflare IPs: {str(e)}")
    CLOUDFLARE_IPS = {'ipv4_cidrs': [], 'ipv6_cidrs': []}

@app.before_request
def fix_client_ip():
    cf_connecting_ip = request.headers.get('CF-Connecting-IP')
    if cf_connecting_ip:
        try:
            client_ip = ipaddress.ip_address(request.remote_addr)
            if any(client_ip in ipaddress.ip_network(cidr) for cidr in CLOUDFLARE_IPS.get('ipv4_cidrs', []) + CLOUDFLARE_IPS.get('ipv6_cidrs', [])):
                request.remote_addr = cf_connecting_ip
        except ValueError:
            app.logger.warning(f"Invalid IP address: {request.remote_addr}")

class RequestFormatter(logging.Formatter):
    def format(self, record):
        record.url = request.url
        record.remote_addr = request.remote_addr
        return super().format(record)

formatter = RequestFormatter(
    '[%(asctime)s] %(remote_addr)s requested %(url)s\n'
    '%(levelname)s in %(module)s: %(message)s'
)
default_handler.setFormatter(formatter)


@app.before_request
def start_timer():
    g.start = time.time()
    app.logger.debug(f"start_timer called for path: {request.path}")

@app.after_request
def log_request(response):
    if request.path == '/favicon.ico':
        return response
    elif request.path.startswith('/static'):
        return response

    if not hasattr(g, 'start'):
        app.logger.warning(f"g.start not set for path: {request.path}")

    now = time.time()
    start_time = getattr(g, 'start', now)  # Use now as a fallback if g.start is not set
    duration = round(now - start_time, 2)
    ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    host = request.host.split(':', 1)[0]
    args = dict(request.args)

    log_params = [
        ('method', request.method),
        ('path', request.path),
        ('status', response.status_code),
        ('duration', duration),
        ('ip', ip),
        ('host', host),
        ('params', args)
    ]

    request_id = request.headers.get('X-Request-ID')
    if request_id:
        log_params.append(('request_id', request_id))

    parts = []
    for name, value in log_params:
        part = f"{name}={value}"
        parts.append(part)
    line = " ".join(parts)

    app.logger.info(line)

    return response

def extract_last_updated_date(content):
    """Try to extract last updated date from privacy policy content"""
    try:
        # Common patterns for last updated dates
        patterns = [
            r'Last (?:updated|modified|revised)(?:\s*:|\s+on)?\s*([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
            r'Effective (?:date|as of)(?:\s*:|\s+on)?\s*([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
            r'Updated:?\s*([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
            r'Revision date:?\s*([A-Za-z]+\s+\d{1,2},?\s+\d{4})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                date_str = match.group(1)
                try:
                    # Parse the date string
                    date_obj = datetime.strptime(date_str, '%B %d, %Y')
                    return date_obj.isoformat()
                except ValueError:
                    continue
        
        return None
    except Exception as e:
        app.logger.error(f"Error extracting last updated date: {str(e)}")
        return None

def generate_cache_key(data):
    """Generate a unique cache key based on request content"""
    content = data.get('content', '')
    url = data.get('url', '')
    
    # Create a unique hash of the content/url
    hash_input = f"{content}{url}".encode('utf-8')
    content_hash = hashlib.sha256(hash_input).hexdigest()
    
    return f"summary:{content_hash}"

@app.after_request
def add_cache_headers(response):
    """Add appropriate cache headers based on request type"""
    if request.method == 'GET' and 'static' in request.path:
        # Cache static content
        response.headers['Cache-Control'] = 'public, max-age=31536000'
        response.headers['Cloudflare-CDN-Cache-Control'] = 'public, max-age=31536000'
    elif request.method == 'POST' and request.path == '/summarize':
        # Allow caching for summaries but revalidate
        response.headers['Cache-Control'] = 'public, max-age=7776000, must-revalidate'
        response.headers['Cloudflare-CDN-Cache-Control'] = 'public, max-age=7776000'
    else:
        # No caching for other dynamic content
        response.headers['Cache-Control'] = 'no-store, private'
        response.headers['Cloudflare-CDN-Cache-Control'] = 'no-cache'
    
    # Add debug headers
    response.headers['X-Cache-Status'] = 'HIT' if g.get('cache_hit') else 'MISS'
    response.headers['X-Cache-Key'] = g.get('cache_key', 'none')
    
    return response

def test_memcached_connection():
    """Test Memcached connection and configuration"""
    try:
        # Get environment variables
        servers = os.environ.get('MEMCACHIER_SERVERS', '').split(',')
        username = os.environ.get('MEMCACHIER_USERNAME', '')
        password = os.environ.get('MEMCACHIER_PASSWORD', '')
        
        app.logger.info(f"Memcached Servers: {servers}")
        app.logger.info(f"Username configured: {'Yes' if username else 'No'}")
        app.logger.info(f"Password configured: {'Yes' if password else 'No'}")
        
        # Try to establish connection
        client = bmemcached.Client(
            servers,
            username=username,
            password=password
        )
        
        # Test set/get operations
        test_key = "test_connection"
        test_value = "working"
        
        set_result = client.set(test_key, test_value)
        app.logger.info(f"Test set operation: {'Success' if set_result else 'Failed'}")
        
        get_result = client.get(test_key)
        app.logger.info(f"Test get operation: {'Success' if get_result == test_value else 'Failed'}")
        
        return True, "Memcached connection test successful"
        
    except Exception as e:
        error_msg = f"Memcached connection test failed: {str(e)}"
        app.logger.error(error_msg)
        return False, error_msg

# Add this to your app initialization
@app.before_first_request
def initialize_cache():
    """Initialize and test cache connection before first request"""
    success, message = test_memcached_connection()
    if not success:
        app.logger.error(f"Cache initialization failed: {message}")
    else:
        app.logger.info("Cache initialization successful")

def _check_cache_status():
    """Internal function to check cache status"""
    try:
        if not cache_client:
            return {
                "status": "error",
                "message": "Cache client not initialized"
            }, False
            
        # Test basic operations
        test_key = f"status_check_{uuid.uuid4()}"
        cache_client.set(test_key, "test", time=60)
        result = cache_client.get(test_key)
        cache_client.delete(test_key)
        
        # Get server stats
        stats = cache_client.get_stats()
        
        return {
            "status": "healthy" if result == "test" else "unhealthy",
            "stats": stats[0][1] if stats else {},
            "servers": os.environ.get('MEMCACHIER_SERVERS', '').split(','),
            "operations_test": "passed" if result == "test" else "failed"
        }, result == "test"
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "servers": os.environ.get('MEMCACHIER_SERVERS', '').split(',')
        }, False

# Protected endpoint for external access
@app.route('/api/cache/status', methods=['GET'])
@token_required
def cache_status(current_user):
    """
    Protected endpoint to check cache system health.
    
    Requirements:
        - Valid JWT token in Authorization header
        - Admin user privileges
        
    Returns:
        JSON response with cache status and health information
        
    Response Codes:
        200: Cache is healthy
        403: Unauthorized (non-admin user)
        500: Cache is unhealthy or error occurred
    """
    try:
        if not current_user.is_admin:  # Optional: restrict to admin users
            return jsonify({"message": "Unauthorized"}), 403
            
        status, is_healthy = _check_cache_status()
        
        # Add request context
        status.update({
            "request_id": request.headers.get('X-Request-ID', 'none'),
            "user": current_user.email,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return jsonify(status), 200 if is_healthy else 500
        
    except Exception as e:
        app.logger.error(f"Cache status check failed: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500
    

@app.before_request
def check_cache_health():
    """Verify cache health before each request"""
    global cache_client
    if not cache_client:
        app.logger.warning("Cache client not initialized, attempting to reconnect...")
        cache_client = init_cache_client()
    elif not hasattr(g, 'cache_checked'):
        try:
            test_key = f"health_check_{uuid.uuid4()}"
            cache_client.set(test_key, "test", time=1)
            cache_client.delete(test_key)
            g.cache_checked = True
        except Exception as e:
            app.logger.error(f"Cache health check failed: {str(e)}")
            cache_client = init_cache_client()
            

def cache_set(key: str, value: any, timeout: int = 3600):
    """Helper function to set cache with error handling"""
    try:
        return cache_client.set(key, value, time=timeout)
    except Exception as e:
        app.logger.error(f"Cache set error for key {key}: {str(e)}")
        return False

def cache_get(key: str):
    """Helper function to get cache with error handling"""
    try:
        return cache_client.get(key)
    except Exception as e:
        app.logger.error(f"Cache get error for key {key}: {str(e)}")
        return None

def cache_delete(key: str):
    """Helper function to delete cache with error handling"""
    try:
        return cache_client.delete(key)
    except Exception as e:
        app.logger.error(f"Cache delete error for key {key}: {str(e)}")
        return False

if __name__ == '__main__':
    logger.info("Starting Flask application in debug mode...")
    app.run()
