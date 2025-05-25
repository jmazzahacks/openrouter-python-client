"""
Images endpoint implementation.

This module provides the endpoint handler for image generation API,
supporting various image creation techniques.

Exported:
- ImagesEndpoint: Handler for images endpoint
"""

import os
import re
from typing import Dict, Optional, Union, Any, BinaryIO

from ..auth import AuthManager
from ..http import HTTPManager
from .base import BaseEndpoint


class ImagesEndpoint(BaseEndpoint):
    """
    Handler for the images API endpoint.
    
    Provides methods for generating and manipulating images.
    """
    
    def __init__(self, auth_manager: AuthManager, http_manager: HTTPManager):
        """
        Initialize the images endpoint handler.
        
        Args:
            auth_manager (AuthManager): Authentication manager.
            http_manager (HTTPManager): HTTP communication manager.
        """
        # Call parent initializer with 'images' as endpoint_path
        super().__init__(auth_manager, http_manager, "images")
        
        # Log initialization of images endpoint
        self.logger.debug("Initialized images endpoint handler")
    
    def create(self, 
               prompt: str,
               model: Optional[str] = None,
               n: Optional[int] = None,
               size: Optional[str] = None,
               response_format: Optional[str] = None,
               quality: Optional[str] = None,
               style: Optional[str] = None,
               **kwargs) -> Dict[str, Any]:
        """
        Create images from a text prompt.
        
        Args:
            prompt (str): Text description of the desired image.
            model (Optional[str]): Model to use for generation.
            n (Optional[int]): Number of images to generate.
            size (Optional[str]): Size of the image (e.g., "1024x1024").
            response_format (Optional[str]): Format for response ("url" or "b64_json").
            quality (Optional[str]): Quality of the image ("standard" or "hd").
            style (Optional[str]): Style preset for the image.
            **kwargs: Additional parameters to pass to the API.
            
        Returns:
            Dict[str, Any]: API response with generated images.
            
        Raises:
            APIError: If the API request fails.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        # Prepare request data with prompt and parameters
        data = {"prompt": prompt}
        
        if model is not None:
            if not isinstance(model, str) or not model.strip():
                raise ValueError("Model cannot be empty")
            data["model"] = model
        if n is not None:
            if not isinstance(n, int) or n <= 0:
                raise ValueError("n must be a positive integer")
            data["n"] = n
        if size is not None:
            if not isinstance(size, str) or re.match(r"^\d+x\d+$", size) is None:
                raise ValueError("Size cannot be empty")
            data["size"] = size
        if response_format is not None:
            if not isinstance(response_format, str) or response_format not in ["url", "b64_json"]:
                raise ValueError("Response format cannot be empty")
            data["response_format"] = response_format
        if quality is not None:
            if not isinstance(quality, str) or quality not in ["standard", "hd"]:
                raise ValueError("Quality must be a string and must be either 'standard' or 'hd'")
            data["quality"] = quality
        if style is not None:
            if not isinstance(style, str) or not style.strip():
                raise ValueError("Style cannot be empty")
            data["style"] = style
            
        # Add any additional kwargs to data
        data.update(kwargs)
        
        # Get authentication headers
        headers = self._get_headers()
        
        # Make POST request to images/generations endpoint
        response = self.http_manager.post(
            self._get_endpoint_url("generations"),
            headers=headers,
            json=data
        )
        
        # Return parsed JSON response
        return response.json()
    
    def edit(self,
             image: Union[str, BinaryIO],
             prompt: str,
             mask: Optional[Union[str, BinaryIO]] = None,
             model: Optional[str] = None,
             n: Optional[int] = None,
             size: Optional[str] = None,
             response_format: Optional[str] = None,
             **kwargs) -> Dict[str, Any]:
        """
        Create edited version of an image.
        
        Args:
            image (Union[str, BinaryIO]): Image file path or file-like object to edit.
            prompt (str): Text description of the desired edit.
            mask (Optional[Union[str, BinaryIO]]): Mask image path or file-like object.
            model (Optional[str]): Model to use for generation.
            n (Optional[int]): Number of images to generate.
            size (Optional[str]): Size of the image (e.g., "1024x1024").
            response_format (Optional[str]): Format for response ("url" or "b64_json").
            **kwargs: Additional parameters to pass to the API.
            
        Returns:
            Dict[str, Any]: API response with generated images.
            
        Raises:
            APIError: If the API request fails.
        """
        # Prepare multipart form data with image, mask, prompt, and parameters
        files = {}
        form_data = {"prompt": prompt}
        
        # Process the main image
        image_obj = None
        mask_obj = None
        
        try:
            # Process the main image file
            image_obj = self._process_image_input(image)
            files["image"] = image_obj
            
            # Process the mask image if provided
            if mask is not None:
                mask_obj = self._process_image_input(mask)
                files["mask"] = mask_obj
            
            # Add additional parameters
            if model is not None:
                form_data["model"] = model
            if n is not None:
                form_data["n"] = n
            if size is not None:
                form_data["size"] = size
            if response_format is not None:
                form_data["response_format"] = response_format
                
            # Add any additional kwargs to form_data
            form_data.update(kwargs)
            
            # Get authentication headers
            headers = self._get_headers()
            # Remove Content-Type as it will be set by the multipart request
            if "Content-Type" in headers:
                del headers["Content-Type"]
            
            # Make POST request to images/edits endpoint
            response = self.http_manager.post(
                self._get_endpoint_url("edits"),
                headers=headers,
                data=form_data,
                files=files
            )
            
            # Return parsed JSON response
            return response.json()
        finally:
            # Close any file objects that we opened
            if isinstance(image, str) and image_obj is not None:
                image_obj.close()
            if isinstance(mask, str) and mask_obj is not None:
                mask_obj.close()
    
    def variations(self,
                   image: Union[str, BinaryIO],
                   model: Optional[str] = None,
                   n: Optional[int] = None,
                   size: Optional[str] = None,
                   response_format: Optional[str] = None,
                   **kwargs) -> Dict[str, Any]:
        """
        Create variations of an existing image.
        
        Args:
            image (Union[str, BinaryIO]): Image file path or file-like object.
            model (Optional[str]): Model to use for generation.
            n (Optional[int]): Number of variations to generate.
            size (Optional[str]): Size of the variations (e.g., "1024x1024").
            response_format (Optional[str]): Format for response ("url" or "b64_json").
            **kwargs: Additional parameters to pass to the API.
            
        Returns:
            Dict[str, Any]: API response with generated image variations.
            
        Raises:
            APIError: If the API request fails.
        """
        # Prepare multipart form data with image and parameters
        files = {}
        form_data = {}
        
        # Process the image file
        image_obj = None
        
        try:
            # Process the image file
            image_obj = self._process_image_input(image)
            files["image"] = image_obj
            
            # Add additional parameters
            if model is not None:
                form_data["model"] = model
            if n is not None:
                form_data["n"] = n
            if size is not None:
                form_data["size"] = size
            if response_format is not None:
                form_data["response_format"] = response_format
                
            # Add any additional kwargs to form_data
            form_data.update(kwargs)
            
            # Get authentication headers
            headers = self._get_headers()
            # Remove Content-Type as it will be set by the multipart request
            if "Content-Type" in headers:
                del headers["Content-Type"]
            
            # Make POST request to images/variations endpoint
            response = self.http_manager.post(
                self._get_endpoint_url("variations"),
                headers=headers,
                data=form_data,
                files=files
            )
            
            # Return parsed JSON response
            return response.json()
        finally:
            # Close file object if we opened it
            if isinstance(image, str) and image_obj is not None:
                image_obj.close()
    
    def _process_image_input(self, image: Union[str, BinaryIO]) -> BinaryIO:
        """
        Process image input to ensure it's in the correct format.
        
        Args:
            image (Union[str, BinaryIO]): Image file path or file-like object.
            
        Returns:
            BinaryIO: File-like object containing the image data.
            
        Raises:
            ValueError: If the image file cannot be read.
        """
        # If image is a string (path)
        if isinstance(image, str):
            # Check if file exists
            if not os.path.isfile(image):
                raise ValueError(f"Image file does not exist: {image}")
            
            # Open file in binary mode and return file object
            return open(image, 'rb')
        
        # Else if image is file-like object
        elif hasattr(image, 'read'):
            # Ensure it's in binary mode if possible
            if hasattr(image, 'mode') and 'b' not in image.mode:
                self.logger.warning("Image file is not opened in binary mode. This may cause issues.")
            
            # Return the image object
            return image
        
        # Else raise ValueError about unsupported image input type
        else:
            raise ValueError("Unsupported image input type. Must be a file path or file-like object.")
