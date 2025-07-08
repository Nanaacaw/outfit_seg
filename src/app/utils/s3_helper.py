import os
import boto3
import tempfile
import logging
import httpx
import cv2
from urllib.parse import urlparse
from botocore.config import Config
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from typing import Optional, Union, List
logger = logging.getLogger(__name__)

load_dotenv()

# Constants
TIMEOUT = 30  # seconds
IMAGE_EXTENSIONS = {
    'jpeg': 'jpg',
    'jpg': 'jpg', 
    'png': 'png',
    'gif': 'gif',
    'webp': 'webp',
    'bmp': 'bmp',
    'tiff': 'tiff'
}

# S3 Configuration
S3_ACCESS_KEY = os.getenv('S3_ACCESS_KEY') 
S3_SECRET_KEY = os.getenv('S3_SECRET_KEY')
S3_ENDPOINT = os.getenv('S3_ENDPOINT')

# Initialize S3 client with custom endpoint
s3_client = boto3.client(
    's3',
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    endpoint_url=S3_ENDPOINT,
    config=Config(
        s3={'addressing_style': 'virtual'}
    )
)

def parse_s3_url(s3_url):
    """
    Parse an S3 URL into bucket name and object key.
    
    Args:
        s3_url (str): S3 URL in format s3://bucket-name/path/to/object
        
    Returns:
        tuple: (bucket_name, object_key)
    """
    parsed = urlparse(s3_url)
    if parsed.scheme != 's3':
        raise ValueError(f"Invalid S3 URL scheme: {parsed.scheme}. URL must start with 's3://'")
    
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')
    
    return bucket, key

def generate_presigned_url(bucket_name, object_key, expiration=3600):
    """
    Generate a presigned URL for an S3 object.
    
    Args:
        bucket_name (str): Name of the S3 bucket
        object_key (str): Key of the object in the bucket
        expiration (int): Time in seconds until the presigned URL expires (default: 1 hour)
        
    Returns:
        str: Presigned URL for the object
    """
    try:
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': object_key},
            ExpiresIn=expiration
        )
        return url
    except Exception as e:
        logger.error(f"Error generating presigned URL: {str(e)}")
        raise Exception(f"Failed to generate presigned URL: {str(e)}")

def download_from_s3(url, local_dir=None):
    """
    Download a file from S3 or public URL to a local temporary file.
    
    Args:
        url (str): S3 URL in format s3://bucket-name/path/to/object 
                  or public URL (http:// or https://)
        local_dir (str, optional): Directory to save the file. If None, uses a temp directory.
        
    Returns:
        str: Path to the downloaded file
    """
    try:
        logger.info(f"Starting download: {url}")
        # Create a temporary file if no local directory is specified
        if local_dir is None:
            temp_dir = tempfile.mkdtemp()
            local_dir = temp_dir
        else:
            os.makedirs(local_dir, exist_ok=True)
        
        # Handle S3 URLs
        if url.startswith('s3://'):
            bucket, key = parse_s3_url(url)
            # Get the filename from the key
            filename = os.path.basename(key)
            local_path = os.path.join(local_dir, filename)
            
            logger.info(f"Downloading {url} to {local_path}")
            
            # Download the file from S3
            s3_client.download_file(bucket, key, local_path)
            logger.info(f"Downloaded from S3: {url} to {local_path}")
            
        # Handle public URLs (http:// or https://)
        elif url.startswith('http://') or url.startswith('https://'):
            # For Pinterest short URLs or other redirects, follow the redirects to get the actual image URL
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Referer': 'https://www.google.com/'
            }
            
            # First make a HEAD request to check content type and get real URL after redirects
            try:
                with httpx.Client(follow_redirects=True, timeout=TIMEOUT) as client:
                    head_response = client.head(url, headers=headers)
                    # Get the final URL after all redirects
                    final_url = str(head_response.url)
                    content_type = head_response.headers.get('content-type', '')
                    
                    # If it's not an image type, try to find an image URL on the page
                    if not content_type.startswith('image/'):
                        # For Pinterest, we need to scrape the page to get the actual image URL
                        if 'pin.it' in url or 'pinterest' in url:
                            logger.info(f"Pinterest URL detected: {url}, fetching HTML to extract image")
                            get_response = client.get(final_url, headers=headers)

                            soup = BeautifulSoup(get_response.text, 'html.parser')
                            
                            # Try to find the main image URL
                            og_image = soup.find('meta', property='og:image')
                            if og_image and og_image.get('content'):
                                final_url = str(og_image.get('content'))
                                logger.info(f"Extracted Pinterest image URL: {final_url}")
                            else:
                                # Try other image finding strategies
                                all_images = soup.find_all('img')
                                for img in all_images:
                                    img_src = img.get('src')
                                    img_class = img.get('class')
                                    if img_src and img_class and 'mainImage' in img_class:
                                        final_url = str(img_src)
                                        break
                                
                                if final_url == str(head_response.url):
                                    raise ValueError(f"Could not find image URL from Pinterest page: {url}")
                        else:
                            raise ValueError(f"URL does not point to an image: {url}, Content-Type: {content_type}")
                    
                    # Get the filename from the URL
                    parsed_url = urlparse(final_url)
                    filename = os.path.basename(parsed_url.path)
                    
                    # If filename is empty or doesn't have an extension, create a default one
                    if not filename or '.' not in filename:
                        # Try to get extension from Content-Type
                        if content_type.startswith('image/'):
                            ext = content_type.split('/')[1].split(';')[0].strip()
                            if ext in IMAGE_EXTENSIONS:
                                ext = IMAGE_EXTENSIONS[ext]
                            filename = f"image_from_url_{hash(url) % 10000}.{ext}"
                        else:
                            filename = f"image_from_url_{hash(url) % 10000}.jpg"
                    
                    local_path = os.path.join(local_dir, filename)
                    
                    logger.info(f"Downloading image from {final_url} to {local_path}")
                    
                    # Download the file
                    response = client.get(final_url, headers=headers)
                    response.raise_for_status()
                    
                    with open(local_path, 'wb') as f:
                        f.write(response.content)
                    
                    # Verify the downloaded file is a valid image
                    img = cv2.imread(local_path)
                    if img is None:
                        raise ValueError(f"Downloaded file is not a valid image: {url}")
                        
            except ImportError:
                logger.warning("BeautifulSoup not installed, can't extract Pinterest images. Install with: pip install beautifulsoup4")
                raise ValueError(f"Can't extract Pinterest images without BeautifulSoup: {url}")
                
        else:
            raise ValueError(f"Unsupported URL format: {url}. Must start with 's3://', 'http://' or 'https://'")
        
        return local_path
    
    except Exception as e:
        logger.error(f"Error downloading from URL: {url} - {str(e)}")
        raise Exception(f"Failed to download from URL: {str(e)}")