from __future__ import annotations

import os
import io
from typing import Optional
from PIL import Image


def pdf_page_to_image(pdf_path: str, page_number: int, dpi: int = 200) -> Optional[bytes]:
    """
    Convert a specific PDF page to an image.
    
    Args:
        pdf_path: Path to the PDF file
        page_number: Page number (0-indexed)
        dpi: Resolution for the image conversion
        
    Returns:
        Image data as bytes (PNG format) or None if conversion fails
    """
    import fitz  # PyMuPDF - now included as default dependency
    
    try:
        # Open the PDF document
        doc = fitz.open(pdf_path)
        
        # Check if page number is valid
        if page_number >= len(doc) or page_number < 0:
            return None
        
        # Get the specific page
        page = doc[page_number]
        
        # Convert page to image
        mat = fitz.Matrix(dpi/72, dpi/72)  # 72 is the default DPI
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PIL Image
        img_data = pix.tobytes("png")
        
        # Close the document
        doc.close()
        
        return img_data
        
    except Exception as e:
        print(f"Error converting PDF page to image: {e}")
        return None


def pdf_page_to_base64_image(pdf_path: str, page_number: int, dpi: int = 200) -> Optional[str]:
    """
    Convert a specific PDF page to a base64-encoded image.
    
    Args:
        pdf_path: Path to the PDF file
        page_number: Page number (0-indexed)
        dpi: Resolution for the image conversion
        
    Returns:
        Base64-encoded image string or None if conversion fails
    """
    import base64
    
    img_data = pdf_page_to_image(pdf_path, page_number, dpi)
    if img_data is None:
        return None
    
    # Convert to base64
    base64_string = base64.b64encode(img_data).decode('utf-8')
    return f"data:image/png;base64,{base64_string}"
