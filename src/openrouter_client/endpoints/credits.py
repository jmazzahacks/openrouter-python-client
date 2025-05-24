"""
Credits endpoint implementation.

This module provides the endpoint handler for credits management API,
allowing checking balance and history, and purchasing credits.

Exported:
- CreditsEndpoint: Handler for credits endpoint
"""

from typing import Dict, Optional, Union, Any
from datetime import datetime

from ..auth import AuthManager
from ..http import HTTPManager
from .base import BaseEndpoint


class CreditsEndpoint(BaseEndpoint):
    """
    Handler for the credits API endpoint.
    
    Provides methods for managing credits and payment.
    """
    
    def __init__(self, auth_manager: AuthManager, http_manager: HTTPManager):
        """
        Initialize the credits endpoint handler.
        
        Args:
            auth_manager (AuthManager): Authentication manager.
            http_manager (HTTPManager): HTTP communication manager.
        """
        # Call parent initializer with 'credits' as endpoint_path
        super().__init__(auth_manager, http_manager, "credits")
        
        # Log initialization of credits endpoint
        self.logger.debug("Initialized credits endpoint handler")
    
    def get(self) -> Dict[str, Any]:
        """
        Get current credit balance and information.
        
        Returns:
            Dict[str, Any]: Credit information including balance.
            
        Raises:
            APIError: If the API request fails.
        """
        # Get authentication headers (requires provisioning API key)
        headers = self._get_headers(require_provisioning=True)
        
        # Make GET request to credits endpoint
        response = self.http_manager.get(
            self._get_endpoint_url(),
            headers=headers
        )
        
        # Return parsed JSON response
        return response.json()
    
    def history(self,
                limit: Optional[int] = None,
                offset: Optional[int] = None,
                start_date: Optional[Union[str, datetime]] = None,
                end_date: Optional[Union[str, datetime]] = None,
                **kwargs) -> Dict[str, Any]:
        """
        Get credit usage history.
        
        Args:
            limit (Optional[int]): Maximum number of items to return.
            offset (Optional[int]): Number of items to skip for pagination.
            start_date (Optional[Union[str, datetime]]): Start date for filtering.
            end_date (Optional[Union[str, datetime]]): End date for filtering.
            **kwargs: Additional parameters to pass to the API.
            
        Returns:
            Dict[str, Any]: Credit usage history.
            
        Raises:
            APIError: If the API request fails.
        """
        # Prepare query parameters from function arguments
        params = {}
        if limit is not None:
            params['limit'] = limit
        if offset is not None:
            params['offset'] = offset
            
        # If start_date or end_date is datetime object:
        #     Convert to ISO format string
        if start_date is not None:
            if isinstance(start_date, datetime):
                params['start_date'] = start_date.isoformat()
            else:
                params['start_date'] = start_date
                
        if end_date is not None:
            if isinstance(end_date, datetime):
                params['end_date'] = end_date.isoformat()
            else:
                params['end_date'] = end_date
                
        # Add any additional kwargs to params
        params.update(kwargs)
        
        # Get authentication headers (requires provisioning API key)
        headers = self._get_headers(require_provisioning=True)
        
        # Make GET request to credits/history endpoint
        response = self.http_manager.get(
            self._get_endpoint_url('history'),
            headers=headers,
            params=params
        )
        
        # Return parsed JSON response
        return response.json()
    
    def purchase(self, amount: int, payment_method: Optional[str] = None) -> Dict[str, Any]:
        """
        Purchase additional credits.
        
        Args:
            amount (int): Number of credits to purchase.
            payment_method (Optional[str]): Payment method ID if using saved method.
            
        Returns:
            Dict[str, Any]: Purchase confirmation or payment URL.
            
        Raises:
            APIError: If the API request fails.
        """
        # Prepare request data with amount and payment_method if provided
        data = {"amount": amount}
        if payment_method is not None:
            data["payment_method"] = payment_method
            
        # Get authentication headers (requires provisioning API key)
        headers = self._get_headers(require_provisioning=True)
        
        # Make POST request to credits/purchase endpoint
        response = self.http_manager.post(
            self._get_endpoint_url('purchase'),
            headers=headers,
            json=data
        )
        
        # Return parsed JSON response
        return response.json()
    
    def payment_methods(self) -> Dict[str, Any]:
        """
        List available payment methods.
        
        Returns:
            Dict[str, Any]: Available payment methods.
            
        Raises:
            APIError: If the API request fails.
        """
        # Get authentication headers (requires provisioning API key)
        headers = self._get_headers(require_provisioning=True)
        
        # Make GET request to credits/payment-methods endpoint
        response = self.http_manager.get(
            self._get_endpoint_url('payment-methods'),
            headers=headers
        )
        
        # Return parsed JSON response
        return response.json()
    
    def add_payment_method(self, payment_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a new payment method.
        
        Args:
            payment_info (Dict[str, Any]): Payment method details.
            
        Returns:
            Dict[str, Any]: Added payment method confirmation.
            
        Raises:
            APIError: If the API request fails.
        """
        # Get authentication headers (requires provisioning API key)
        headers = self._get_headers(require_provisioning=True)
        
        # Make POST request to credits/payment-methods endpoint with payment_info
        response = self.http_manager.post(
            self._get_endpoint_url('payment-methods'),
            headers=headers,
            json=payment_info
        )
        
        # Return parsed JSON response
        return response.json()
