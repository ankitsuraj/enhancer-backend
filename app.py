#!/usr/bin/env python3
"""
Professional AI Image Enhancer Web Application
Complete Backend server with Flask and advanced image processing
"""

from flask import Flask, request, render_template, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import os
import io
import base64
from werkzeug.utils import secure_filename
import uuid
import time
from datetime import datetime
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'

# Create necessary directories
directories = ['uploads', 'enhanced', 'static/temp']
for directory in directories:
    os.makedirs(directory, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'tiff', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class ProfessionalImageEnhancer:
    """Advanced image enhancement with professional algorithms"""
    
    def __init__(self):
        self.enhancement_params = {
            'upscale_factor': 2,
            'denoise_strength': 15,
            'sharpen_amount': 1.8,
            'contrast_factor': 1.3,
            'brightness_factor': 1.1,
            'color_factor': 1.2,
            'gamma_correction': 1.1
        }
        
    def load_and_prepare_image(self, image_path):
        """Load and prepare image for processing"""
        try:
            # Load with PIL
            pil_image = Image.open(image_path)
            
            # Handle EXIF orientation
            pil_image = ImageOps.exif_transpose(pil_image)
            
            # Convert to RGB if necessary
            if pil_image.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', pil_image.size, (255, 255, 255))
                if pil_image.mode == 'P':
                    pil_image = pil_image.convert('RGBA')
                background.paste(pil_image, mask=pil_image.split()[-1] if pil_image.mode in ('RGBA', 'LA') else None)
                pil_image = background
            elif pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Load with OpenCV
            cv_image = cv2.imread(image_path)
            if cv_image is not None:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            return pil_image, cv_image
            
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            return None, None
    
    def super_resolution_upscale(self, cv_image, factor=2):
        """Advanced upscaling with edge preservation"""
        height, width = cv_image.shape[:2]
        new_width = int(width * factor)
        new_height = int(height * factor)
        
        # Use INTER_LANCZOS4 for best quality upscaling
        upscaled = cv2.resize(cv_image, (new_width, new_height), 
                             interpolation=cv2.INTER_LANCZOS4)
        
        # Apply additional sharpening after upscaling
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(upscaled, -1, kernel * 0.1)
        
        # Blend original upscaled with sharpened version
        result = cv2.addWeighted(upscaled, 0.7, sharpened, 0.3, 0)
        
        return result
    
    def advanced_denoising(self, cv_image, strength=15):
        """Multi-stage denoising pipeline"""
        # Stage 1: Non-local means denoising
        denoised = cv2.fastNlMeansDenoisingColored(cv_image, None, strength, strength, 7, 21)
        
        # Stage 2: Bilateral filtering for edge preservation
        bilateral = cv2.bilateralFilter(denoised, 9, 80, 80)
        
        # Stage 3: Morphological operations for fine details
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opening = cv2.morphologyEx(bilateral, cv2.MORPH_OPENING, kernel, iterations=1)
        
        return opening
    
    def smart_contrast_enhancement(self, cv_image):
        """Intelligent contrast enhancement using CLAHE and histogram equalization"""
        # Convert to LAB color space
        lab = cv2.cvtColor(cv_image, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Additional histogram equalization for better distribution
        yuv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        final = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        
        return final
    
    def professional_sharpening(self, pil_image, amount=1.8):
        """Professional unsharp mask sharpening"""
        # High-quality unsharp mask
        sharpened = pil_image.filter(ImageFilter.UnsharpMask(
            radius=3, percent=int(amount * 100), threshold=5))
        
        # Additional edge enhancement
        edge_enhance = sharpened.filter(ImageFilter.EDGE_ENHANCE_MORE)
        
        # Blend original with enhanced
        from PIL import ImageChops
        blended = ImageChops.blend(sharpened, edge_enhance, 0.3)
        
        return blended
    
    def color_grading(self, pil_image):
        """Professional color grading and correction"""
        # Enhance color saturation
        color_enhancer = ImageEnhance.Color(pil_image)
        enhanced = color_enhancer.enhance(self.enhancement_params['color_factor'])
        
        # Adjust brightness
        brightness_enhancer = ImageEnhance.Brightness(enhanced)
        enhanced = brightness_enhancer.enhance(self.enhancement_params['brightness_factor'])
        
        # Enhance contrast
        contrast_enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = contrast_enhancer.enhance(self.enhancement_params['contrast_factor'])
        
        return enhanced
    
    def gamma_correction(self, cv_image, gamma=1.1):
        """Apply gamma correction for better exposure"""
        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")
        
        # Apply gamma correction using the lookup table
        return cv2.LUT(cv_image, table)
    
    def enhance_image(self, image_path, progress_callback=None):
        """Main enhancement pipeline with progress tracking"""
        try:
            start_time = time.time()
            
            # Step 1: Load image
            if progress_callback:
                progress_callback("Loading and preparing image...", 10)
            
            pil_original, cv_original = self.load_and_prepare_image(image_path)
            if pil_original is None:
                return None, "Failed to load image"
            
            original_size = pil_original.size
            
            # Step 2: Super-resolution upscaling
            if progress_callback:
                progress_callback("Applying super-resolution upscaling...", 25)
            
            cv_enhanced = self.super_resolution_upscale(
                cv_original, self.enhancement_params['upscale_factor'])
            
            # Step 3: Advanced denoising
            if progress_callback:
                progress_callback("Removing noise and artifacts...", 40)
            
            cv_enhanced = self.advanced_denoising(
                cv_enhanced, self.enhancement_params['denoise_strength'])
            
            # Step 4: Smart contrast enhancement
            if progress_callback:
                progress_callback("Enhancing contrast and details...", 55)
            
            cv_enhanced = self.smart_contrast_enhancement(cv_enhanced)
            
            # Step 5: Gamma correction
            if progress_callback:
                progress_callback("Applying gamma correction...", 70)
            
            cv_enhanced = self.gamma_correction(
                cv_enhanced, self.enhancement_params['gamma_correction'])
            
            # Convert back to PIL for final processing
            pil_enhanced = Image.fromarray(cv_enhanced)
            
            # Step 6: Professional sharpening
            if progress_callback:
                progress_callback("Applying professional sharpening...", 85)
            
            pil_enhanced = self.professional_sharpening(
                pil_enhanced, self.enhancement_params['sharpen_amount'])
            
            # Step 7: Color grading
            if progress_callback:
                progress_callback("Final color grading and optimization...", 95)
            
            pil_enhanced = self.color_grading(pil_enhanced)
            
            # Final processing time
            processing_time = round(time.time() - start_time, 2)
            
            if progress_callback:
                progress_callback("Enhancement complete!", 100)
            
            # Prepare result info
            result_info = {
                'original_size': original_size,
                'enhanced_size': pil_enhanced.size,
                'processing_time': processing_time,
                'upscale_factor': self.enhancement_params['upscale_factor'],
                'enhancements_applied': [
                    'Super-resolution upscaling',
                    'Advanced noise reduction',
                    'Smart contrast enhancement',
                    'Gamma correction',
                    'Professional sharpening',
                    'Color grading'
                ]
            }
            
            return pil_enhanced, result_info
            
        except Exception as e:
            logger.error(f"Enhancement error: {str(e)}")
            return None, f"Enhancement failed: {str(e)}"

# Initialize enhancer
enhancer = ProfessionalImageEnhancer()

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/api/enhance', methods=['POST'])
def enhance_image_api():
    """API endpoint for image enhancement"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
        
        # Generate unique filename
        unique_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = secure_filename(f"{timestamp}_{unique_id}_{file.filename}")
        file_path = os.path.join('uploads', filename)
        
        # Save original file
        file.save(file_path)
        
        # Get file info
        file_size = os.path.getsize(file_path)
        
        # Progress tracking (you can implement WebSocket for real-time updates)
        def progress_callback(message, percent):
            logger.info(f"Progress: {percent}% - {message}")
        
        # Enhance image
        enhanced_image, result_info = enhancer.enhance_image(file_path, progress_callback)
        
        if enhanced_image is None:
            os.remove(file_path)
            return jsonify({'success': False, 'error': result_info}), 500
        
        # Save enhanced image
        enhanced_filename = f"enhanced_{filename}"
        enhanced_path = os.path.join('enhanced', enhanced_filename)
        enhanced_image.save(enhanced_path, 'JPEG', quality=95, optimize=True)
        
        # Convert images to base64
        def image_to_base64(img_path):
            with open(img_path, 'rb') as img_file:
                return base64.b64encode(img_file.read()).decode()
        
        original_b64 = image_to_base64(file_path)
        enhanced_b64 = image_to_base64(enhanced_path)
        
        # Prepare response
        response_data = {
            'success': True,
            'original_image': original_b64,
            'enhanced_image': enhanced_b64,
            'original_size': f"{result_info['original_size'][0]}x{result_info['original_size'][1]}",
            'enhanced_size': f"{result_info['enhanced_size'][0]}x{result_info['enhanced_size'][1]}",
            'file_size': f"{round(file_size / (1024*1024), 2)} MB",
            'enhanced_file_size': f"{round(os.path.getsize(enhanced_path) / (1024*1024), 2)} MB",
            'processing_time': result_info['processing_time'],
            'upscale_factor': result_info['upscale_factor'],
            'enhancements': result_info['enhancements_applied'],
            'enhanced_filename': enhanced_filename
        }
        
        # Cleanup original file (keep enhanced for download)
        os.remove(file_path)
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500

@app.route('/api/download/<filename>')
def download_enhanced(filename):
    """Download enhanced image"""
    try:
        file_path = os.path.join('enhanced', filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(file_path, as_attachment=True, 
                        download_name=f"enhanced_{filename}")
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 50MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found.'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error.'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    print("🚀 Starting Professional AI Image Enhancer...")
    print(f"📡 Server running on http://localhost:{port}")
    print("🖼️ Ready to enhance images!")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
