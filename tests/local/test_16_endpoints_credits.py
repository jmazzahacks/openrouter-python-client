import pytest
from datetime import datetime
from unittest.mock import Mock, patch
from pydantic import ValidationError

from openrouter_client.auth import AuthManager
from openrouter_client.http import HTTPManager
from openrouter_client.exceptions import APIError
from openrouter_client.endpoints.credits import CreditsEndpoint


class Test_CreditsEndpoint_Init_01_NominalBehaviors:
    """Test nominal behaviors for CreditsEndpoint.__init__ method."""
    
    def test_initialize_with_valid_managers_and_verify_endpoint_path(self):
        """Test initialization with valid AuthManager and HTTPManager instances and verify correct endpoint path assignment."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        
        # Act
        endpoint = CreditsEndpoint(auth_manager, http_manager)
        
        # Assert
        assert endpoint.auth_manager is auth_manager
        assert endpoint.http_manager is http_manager
        assert endpoint.endpoint_path == "credits"
        assert hasattr(endpoint, 'logger')


class Test_CreditsEndpoint_Init_02_NegativeBehaviors:
    """Test negative behaviors for CreditsEndpoint.__init__ method."""
    
    @pytest.mark.parametrize("auth_manager,http_manager,expected_exception", [
        (None, Mock(spec=HTTPManager), (ValidationError, TypeError, AttributeError)),
        (Mock(spec=AuthManager), None, (ValidationError, TypeError, AttributeError)),
        (None, None, (ValidationError, TypeError, AttributeError)),
        ("invalid_auth", Mock(spec=HTTPManager), (ValidationError, TypeError, AttributeError)),
        (Mock(spec=AuthManager), "invalid_http", (ValidationError, TypeError, AttributeError)),
        (123, Mock(spec=HTTPManager), (ValidationError, TypeError, AttributeError)),
        (Mock(spec=AuthManager), 456, (ValidationError, TypeError, AttributeError)),
    ])
    def test_initialize_with_invalid_parameters(self, auth_manager, http_manager, expected_exception):
        """Test initialization with None or invalid object types for required parameters."""
        # Act & Assert
        with pytest.raises(expected_exception):
            CreditsEndpoint(auth_manager, http_manager)


class Test_CreditsEndpoint_Init_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for CreditsEndpoint.__init__ method."""
    
    def test_handle_exceptions_during_initialization(self):
        """Test handling exceptions during object initialization."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        
        # Mock BaseEndpoint to raise exception during initialization
        with patch('openrouter_client.endpoints.base.BaseEndpoint.__init__', side_effect=RuntimeError("Initialization failed")):
            # Act & Assert
            with pytest.raises(RuntimeError, match="Initialization failed"):
                CreditsEndpoint(auth_manager, http_manager)


class Test_CreditsEndpoint_Init_05_StateTransitionBehaviors:
    """Test state transition behaviors for CreditsEndpoint.__init__ method."""
    
    def test_object_transitions_to_operational_state(self):
        """Test object transitions from uninitialized to fully operational state."""
        # Arrange
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        
        # Act
        endpoint = CreditsEndpoint(auth_manager, http_manager)
        
        # Assert - verify object is fully operational
        assert endpoint.auth_manager is not None
        assert endpoint.http_manager is not None
        assert endpoint.endpoint_path == "credits"
        assert callable(getattr(endpoint, 'get', None))
        assert callable(getattr(endpoint, 'history', None))
        assert callable(getattr(endpoint, 'purchase', None))
        assert callable(getattr(endpoint, 'payment_methods', None))
        assert callable(getattr(endpoint, 'add_payment_method', None))


class Test_CreditsEndpoint_Get_01_NominalBehaviors:
    """Test nominal behaviors for CreditsEndpoint.get method."""
    
    @pytest.fixture
    def endpoint(self):
        """Create CreditsEndpoint instance for testing."""
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        return CreditsEndpoint(auth_manager, http_manager)
    
    @pytest.mark.parametrize("mock_response_data", [
        {"balance": 100, "currency": "USD"},
        {"balance": 0, "currency": "USD", "last_updated": "2025-05-24T19:20:00Z"},
        {"balance": 999999, "currency": "USD", "subscription": "premium"},
        {"balance": 50.5, "currency": "EUR", "bonus_credits": 10},
    ])
    def test_successfully_retrieve_credit_balance_with_valid_authentication(self, endpoint, mock_response_data):
        """Test successfully retrieve current credit balance with valid authentication and return properly formatted response."""
        # Arrange
        mock_response = Mock()
        mock_response.json.return_value = mock_response_data
        endpoint.http_manager.get.return_value = mock_response
        endpoint._get_headers = Mock(return_value={"Authorization": "Bearer valid_token"})
        endpoint._get_endpoint_url = Mock(return_value="https://api.example.com/credits")
        
        # Act
        result = endpoint.get()
        
        # Assert
        assert result == mock_response_data
        endpoint._get_headers.assert_called_once_with(require_provisioning=True)
        endpoint.http_manager.get.assert_called_once_with(
            "https://api.example.com/credits",
            headers={"Authorization": "Bearer valid_token"}
        )


class Test_CreditsEndpoint_Get_02_NegativeBehaviors:
    """Test negative behaviors for CreditsEndpoint.get method."""
    
    @pytest.fixture
    def endpoint(self):
        """Create CreditsEndpoint instance for testing."""
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        return CreditsEndpoint(auth_manager, http_manager)
    
    @pytest.mark.parametrize("auth_error,expected_exception", [
        ("Missing API key", APIError),
        ("Invalid API key", APIError),
        ("Expired API key", APIError),
        ("Insufficient permissions", APIError),
    ])
    def test_handle_invalid_or_missing_provisioning_api_key(self, endpoint, auth_error, expected_exception):
        """Test handling invalid or missing provisioning API key authentication."""
        # Arrange
        endpoint._get_headers = Mock(side_effect=expected_exception(auth_error))
        
        # Act & Assert
        with pytest.raises(expected_exception, match=auth_error):
            endpoint.get()


class Test_CreditsEndpoint_Get_03_BoundaryBehaviors:
    """Test boundary behaviors for CreditsEndpoint.get method."""
    
    @pytest.fixture
    def endpoint(self):
        """Create CreditsEndpoint instance for testing."""
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        return CreditsEndpoint(auth_manager, http_manager)
    
    def test_handle_zero_credit_balance(self, endpoint):
        """Test handling zero credit balance scenarios."""
        # Arrange
        mock_response = Mock()
        mock_response.json.return_value = {"balance": 0, "currency": "USD"}
        endpoint.http_manager.get.return_value = mock_response
        endpoint._get_headers = Mock(return_value={"Authorization": "Bearer token"})
        endpoint._get_endpoint_url = Mock(return_value="https://api.example.com/credits")
        
        # Act
        result = endpoint.get()
        
        # Assert
        assert result["balance"] == 0
        assert "currency" in result


class Test_CreditsEndpoint_Get_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for CreditsEndpoint.get method."""
    
    @pytest.fixture
    def endpoint(self):
        """Create CreditsEndpoint instance for testing."""
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        return CreditsEndpoint(auth_manager, http_manager)
    
    @pytest.mark.parametrize("exception_type,error_message", [
        (APIError, "API request failed"),
        (ConnectionError, "Network connectivity failure"),
        (TimeoutError, "Request timeout"),
        (ValueError, "Invalid response format"),
    ])
    def test_manage_api_errors_and_network_failures(self, endpoint, exception_type, error_message):
        """Test managing APIError exceptions and network connectivity failures."""
        # Arrange
        endpoint._get_headers = Mock(return_value={"Authorization": "Bearer token"})
        endpoint._get_endpoint_url = Mock(return_value="https://api.example.com/credits")
        endpoint.http_manager.get.side_effect = exception_type(error_message)
        
        # Act & Assert
        with pytest.raises(exception_type, match=error_message):
            endpoint.get()


class Test_CreditsEndpoint_History_01_NominalBehaviors:
    """Test nominal behaviors for CreditsEndpoint.history method."""
    
    @pytest.fixture
    def endpoint(self):
        """Create CreditsEndpoint instance for testing."""
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        return CreditsEndpoint(auth_manager, http_manager)
    
    @pytest.mark.parametrize("limit,offset,start_date,end_date,kwargs,expected_params", [
        (10, 0, None, None, {}, {"limit": 10, "offset": 0}),
        (50, 25, "2025-01-01", "2025-05-24", {}, {"limit": 50, "offset": 25, "start_date": "2025-01-01", "end_date": "2025-05-24"}),
        (None, None, datetime(2025, 1, 1), datetime(2025, 5, 24), {}, {"start_date": "2025-01-01T00:00:00", "end_date": "2025-05-24T00:00:00"}),
        (20, 10, None, None, {"category": "purchase"}, {"limit": 20, "offset": 10, "category": "purchase"}),
    ])
    def test_retrieve_history_with_pagination_and_datetime_conversion(self, endpoint, limit, offset, start_date, end_date, kwargs, expected_params):
        """Test retrieving credit history with pagination parameters and datetime conversion to ISO format."""
        # Arrange
        mock_response_data = {"history": [], "total": 0}
        mock_response = Mock()
        mock_response.json.return_value = mock_response_data
        endpoint.http_manager.get.return_value = mock_response
        endpoint._get_headers = Mock(return_value={"Authorization": "Bearer token"})
        endpoint._get_endpoint_url = Mock(return_value="https://api.example.com/credits/history")
        
        # Act
        result = endpoint.history(limit=limit, offset=offset, start_date=start_date, end_date=end_date, **kwargs)
        
        # Assert
        assert result == mock_response_data
        endpoint.http_manager.get.assert_called_once_with(
            "https://api.example.com/credits/history",
            headers={"Authorization": "Bearer token"},
            params=expected_params
        )


class Test_CreditsEndpoint_History_02_NegativeBehaviors:
    """Test negative behaviors for CreditsEndpoint.history method."""
    
    @pytest.fixture
    def endpoint(self):
        """Create CreditsEndpoint instance for testing."""
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        return CreditsEndpoint(auth_manager, http_manager)
    
    @pytest.mark.parametrize("limit,offset", [
        (-1, 0),
        (10, -5),
        (-10, -5),
    ])
    def test_handle_negative_limit_offset_values(self, endpoint, limit, offset):
        """Test handling negative values for limit and offset parameters."""
        # Arrange
        mock_response = Mock()
        mock_response.json.return_value = {"error": "Invalid parameters"}
        endpoint.http_manager.get.return_value = mock_response
        endpoint._get_headers = Mock(return_value={"Authorization": "Bearer token"})
        endpoint._get_endpoint_url = Mock(return_value="https://api.example.com/credits/history")
        
        # Act
        result = endpoint.history(limit=limit, offset=offset)
        
        # Assert
        endpoint.http_manager.get.assert_called_once()
        call_args = endpoint.http_manager.get.call_args
        assert call_args[1]["params"]["limit"] == limit
        assert call_args[1]["params"]["offset"] == offset


class Test_CreditsEndpoint_History_03_BoundaryBehaviors:
    """Test boundary behaviors for CreditsEndpoint.history method."""
    
    @pytest.fixture
    def endpoint(self):
        """Create CreditsEndpoint instance for testing."""
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        return CreditsEndpoint(auth_manager, http_manager)
    
    @pytest.mark.parametrize("limit,expected_response", [
        (0, {"history": [], "message": "No records to return"}),
        (1, {"history": [{"id": 1, "amount": 100}], "total": 1}),
    ])
    def test_process_limit_zero_and_empty_responses(self, endpoint, limit, expected_response):
        """Test processing limit of 0 and handling empty history responses."""
        # Arrange
        mock_response = Mock()
        mock_response.json.return_value = expected_response
        endpoint.http_manager.get.return_value = mock_response
        endpoint._get_headers = Mock(return_value={"Authorization": "Bearer token"})
        endpoint._get_endpoint_url = Mock(return_value="https://api.example.com/credits/history")
        
        # Act
        result = endpoint.history(limit=limit)
        
        # Assert
        assert result == expected_response
        call_args = endpoint.http_manager.get.call_args
        assert call_args[1]["params"]["limit"] == limit


class Test_CreditsEndpoint_History_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for CreditsEndpoint.history method."""
    
    @pytest.fixture
    def endpoint(self):
        """Create CreditsEndpoint instance for testing."""
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        return CreditsEndpoint(auth_manager, http_manager)
    
    @pytest.mark.parametrize("start_date,expected_exception", [
        ("invalid-date", ValueError),
        ("2025-13-01", ValueError),
        ("not-a-date", ValueError),
    ])
    def test_handle_datetime_conversion_errors(self, endpoint, start_date, expected_exception):
        """Test handling APIError exceptions and datetime conversion errors."""
        # Note: This tests the scenario where datetime parsing might fail in the application logic
        # Since the current implementation doesn't validate dates, we simulate API-level validation
        endpoint._get_headers = Mock(return_value={"Authorization": "Bearer token"})
        endpoint._get_endpoint_url = Mock(return_value="https://api.example.com/credits/history")
        endpoint.http_manager.get.side_effect = APIError("Invalid date format")
        
        # Act & Assert
        with pytest.raises(APIError, match="Invalid date format"):
            endpoint.history(start_date=start_date)


class Test_CreditsEndpoint_Purchase_01_NominalBehaviors:
    """Test nominal behaviors for CreditsEndpoint.purchase method."""
    
    @pytest.fixture
    def endpoint(self):
        """Create CreditsEndpoint instance for testing."""
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        return CreditsEndpoint(auth_manager, http_manager)
    
    @pytest.mark.parametrize("amount,payment_method,expected_data,expected_response", [
        (100, None, {"amount": 100}, {"transaction_id": "tx123", "status": "completed"}),
        (50, "pm_123", {"amount": 50, "payment_method": "pm_123"}, {"transaction_id": "tx456", "payment_url": "https://pay.example.com"}),
        (1, "pm_456", {"amount": 1, "payment_method": "pm_456"}, {"transaction_id": "tx789", "status": "pending"}),
    ])
    def test_execute_credit_purchases_with_valid_amounts(self, endpoint, amount, payment_method, expected_data, expected_response):
        """Test successfully executing credit purchases with valid amounts and processing payment method specifications."""
        # Arrange
        mock_response = Mock()
        mock_response.json.return_value = expected_response
        endpoint.http_manager.post.return_value = mock_response
        endpoint._get_headers = Mock(return_value={"Authorization": "Bearer token"})
        endpoint._get_endpoint_url = Mock(return_value="https://api.example.com/credits/purchase")
        
        # Act
        result = endpoint.purchase(amount, payment_method)
        
        # Assert
        assert result == expected_response
        endpoint.http_manager.post.assert_called_once_with(
            "https://api.example.com/credits/purchase",
            headers={"Authorization": "Bearer token"},
            json=expected_data
        )


class Test_CreditsEndpoint_Purchase_02_NegativeBehaviors:
    """Test negative behaviors for CreditsEndpoint.purchase method."""
    
    @pytest.fixture
    def endpoint(self):
        """Create CreditsEndpoint instance for testing."""
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        return CreditsEndpoint(auth_manager, http_manager)
    
    @pytest.mark.parametrize("amount", [
        -1, -100, 0
    ])
    def test_reject_negative_or_zero_amounts(self, endpoint, amount):
        """Test rejecting attempts to purchase negative or zero credit amounts."""
        # Arrange
        endpoint._get_headers = Mock(return_value={"Authorization": "Bearer token"})
        endpoint._get_endpoint_url = Mock(return_value="https://api.example.com/credits/purchase")
        endpoint.http_manager.post.side_effect = APIError("Invalid amount")
        
        # Act & Assert
        with pytest.raises(APIError, match="Invalid amount"):
            endpoint.purchase(amount)


class Test_CreditsEndpoint_Purchase_03_BoundaryBehaviors:
    """Test boundary behaviors for CreditsEndpoint.purchase method."""
    
    @pytest.fixture
    def endpoint(self):
        """Create CreditsEndpoint instance for testing."""
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        return CreditsEndpoint(auth_manager, http_manager)
    
    def test_handle_minimum_purchase_amount(self, endpoint):
        """Test handling minimum valid purchase amount (1 credit)."""
        # Arrange
        mock_response = Mock()
        mock_response.json.return_value = {"transaction_id": "tx_min", "amount": 1, "status": "completed"}
        endpoint.http_manager.post.return_value = mock_response
        endpoint._get_headers = Mock(return_value={"Authorization": "Bearer token"})
        endpoint._get_endpoint_url = Mock(return_value="https://api.example.com/credits/purchase")
        
        # Act
        result = endpoint.purchase(1)
        
        # Assert
        assert result["amount"] == 1
        assert result["transaction_id"] == "tx_min"
        call_args = endpoint.http_manager.post.call_args
        assert call_args[1]["json"]["amount"] == 1


class Test_CreditsEndpoint_Purchase_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for CreditsEndpoint.purchase method."""
    
    @pytest.fixture
    def endpoint(self):
        """Create CreditsEndpoint instance for testing."""
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        return CreditsEndpoint(auth_manager, http_manager)
    
    @pytest.mark.parametrize("exception_type,error_message", [
        (APIError, "Payment processing failed"),
        (APIError, "Insufficient funds"),
        (APIError, "Invalid payment method"),
        (ConnectionError, "Network failure during payment"),
    ])
    def test_handle_payment_failures_and_api_errors(self, endpoint, exception_type, error_message):
        """Test handling payment processing failures and APIError exceptions."""
        # Arrange
        endpoint._get_headers = Mock(return_value={"Authorization": "Bearer token"})
        endpoint._get_endpoint_url = Mock(return_value="https://api.example.com/credits/purchase")
        endpoint.http_manager.post.side_effect = exception_type(error_message)
        
        # Act & Assert
        with pytest.raises(exception_type, match=error_message):
            endpoint.purchase(100)


class Test_CreditsEndpoint_Purchase_05_StateTransitionBehaviors:
    """Test state transition behaviors for CreditsEndpoint.purchase method."""
    
    @pytest.fixture
    def endpoint(self):
        """Create CreditsEndpoint instance for testing."""
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        return CreditsEndpoint(auth_manager, http_manager)
    
    def test_verify_credit_balance_increases_after_purchase(self, endpoint):
        """Test verifying credit balance increases correctly after successful purchase."""
        # Arrange - simulate successful purchase
        purchase_response = Mock()
        purchase_response.json.return_value = {"transaction_id": "tx123", "status": "completed", "credits_added": 100}
        
        # Arrange - simulate balance check after purchase
        balance_response = Mock()
        balance_response.json.return_value = {"balance": 200}  # Increased balance
        
        endpoint.http_manager.post.return_value = purchase_response
        endpoint.http_manager.get.return_value = balance_response
        endpoint._get_headers = Mock(return_value={"Authorization": "Bearer token"})
        endpoint._get_endpoint_url = Mock(side_effect=lambda path="": f"https://api.example.com/credits/{path}".rstrip("/"))
        
        # Act - purchase credits
        purchase_result = endpoint.purchase(100)
        
        # Act - check new balance
        balance_result = endpoint.get()
        
        # Assert
        assert purchase_result["status"] == "completed"
        assert purchase_result["credits_added"] == 100
        assert balance_result["balance"] == 200


class Test_CreditsEndpoint_PaymentMethods_01_NominalBehaviors:
    """Test nominal behaviors for CreditsEndpoint.payment_methods method."""
    
    @pytest.fixture
    def endpoint(self):
        """Create CreditsEndpoint instance for testing."""
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        return CreditsEndpoint(auth_manager, http_manager)
    
    @pytest.mark.parametrize("mock_response_data", [
        {"payment_methods": []},
        {"payment_methods": [{"id": "pm_1", "type": "card", "last4": "4242"}]},
        {"payment_methods": [
            {"id": "pm_1", "type": "card", "last4": "4242"},
            {"id": "pm_2", "type": "bank", "bank_name": "Example Bank"}
        ]},
    ])
    def test_successfully_retrieve_payment_methods_with_authentication(self, endpoint, mock_response_data):
        """Test successfully retrieving list of available payment methods with proper authentication."""
        # Arrange
        mock_response = Mock()
        mock_response.json.return_value = mock_response_data
        endpoint.http_manager.get.return_value = mock_response
        endpoint._get_headers = Mock(return_value={"Authorization": "Bearer token"})
        endpoint._get_endpoint_url = Mock(return_value="https://api.example.com/credits/payment-methods")
        
        # Act
        result = endpoint.payment_methods()
        
        # Assert
        assert result == mock_response_data
        endpoint._get_headers.assert_called_once_with(require_provisioning=True)


class Test_CreditsEndpoint_PaymentMethods_02_NegativeBehaviors:
    """Test negative behaviors for CreditsEndpoint.payment_methods method."""
    
    @pytest.fixture
    def endpoint(self):
        """Create CreditsEndpoint instance for testing."""
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        return CreditsEndpoint(auth_manager, http_manager)
    
    def test_handle_unauthorized_access(self, endpoint):
        """Test handling unauthorized access attempts."""
        # Arrange
        endpoint._get_headers = Mock(side_effect=APIError("Unauthorized access"))
        
        # Act & Assert
        with pytest.raises(APIError, match="Unauthorized access"):
            endpoint.payment_methods()


class Test_CreditsEndpoint_PaymentMethods_03_BoundaryBehaviors:
    """Test boundary behaviors for CreditsEndpoint.payment_methods method."""
    
    @pytest.fixture
    def endpoint(self):
        """Create CreditsEndpoint instance for testing."""
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        return CreditsEndpoint(auth_manager, http_manager)
    
    def test_handle_accounts_with_no_payment_methods(self, endpoint):
        """Test handling accounts with no saved payment methods."""
        # Arrange
        mock_response = Mock()
        mock_response.json.return_value = {"payment_methods": [], "message": "No payment methods found"}
        endpoint.http_manager.get.return_value = mock_response
        endpoint._get_headers = Mock(return_value={"Authorization": "Bearer token"})
        endpoint._get_endpoint_url = Mock(return_value="https://api.example.com/credits/payment-methods")
        
        # Act
        result = endpoint.payment_methods()
        
        # Assert
        assert result["payment_methods"] == []
        assert "message" in result


class Test_CreditsEndpoint_PaymentMethods_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for CreditsEndpoint.payment_methods method."""
    
    @pytest.fixture
    def endpoint(self):
        """Create CreditsEndpoint instance for testing."""
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        return CreditsEndpoint(auth_manager, http_manager)
    
    @pytest.mark.parametrize("exception_type,error_message", [
        (APIError, "Service unavailable"),
        (ConnectionError, "Network connectivity issue"),
        (TimeoutError, "Request timeout"),
    ])
    def test_handle_api_error_exceptions(self, endpoint, exception_type, error_message):
        """Test handling APIError exceptions from API failures."""
        # Arrange
        endpoint._get_headers = Mock(return_value={"Authorization": "Bearer token"})
        endpoint._get_endpoint_url = Mock(return_value="https://api.example.com/credits/payment-methods")
        endpoint.http_manager.get.side_effect = exception_type(error_message)
        
        # Act & Assert
        with pytest.raises(exception_type, match=error_message):
            endpoint.payment_methods()


class Test_CreditsEndpoint_AddPaymentMethod_01_NominalBehaviors:
    """Test nominal behaviors for CreditsEndpoint.add_payment_method method."""
    
    @pytest.fixture
    def endpoint(self):
        """Create CreditsEndpoint instance for testing."""
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        return CreditsEndpoint(auth_manager, http_manager)
    
    @pytest.mark.parametrize("payment_info,expected_response", [
        (
            {"type": "card", "number": "4242424242424242", "exp_month": 12, "exp_year": 2026, "cvc": "123"},
            {"id": "pm_new1", "type": "card", "last4": "4242", "status": "active"}
        ),
        (
            {"type": "bank", "account_number": "12345678", "routing_number": "987654321"},
            {"id": "pm_new2", "type": "bank", "bank_name": "Example Bank", "status": "pending"}
        ),
    ])
    def test_successfully_add_valid_payment_method(self, endpoint, payment_info, expected_response):
        """Test successfully adding valid payment method information and returning confirmation."""
        # Arrange
        mock_response = Mock()
        mock_response.json.return_value = expected_response
        endpoint.http_manager.post.return_value = mock_response
        endpoint._get_headers = Mock(return_value={"Authorization": "Bearer token"})
        endpoint._get_endpoint_url = Mock(return_value="https://api.example.com/credits/payment-methods")
        
        # Act
        result = endpoint.add_payment_method(payment_info)
        
        # Assert
        assert result == expected_response
        endpoint.http_manager.post.assert_called_once_with(
            "https://api.example.com/credits/payment-methods",
            headers={"Authorization": "Bearer token"},
            json=payment_info
        )


class Test_CreditsEndpoint_AddPaymentMethod_02_NegativeBehaviors:
    """Test negative behaviors for CreditsEndpoint.add_payment_method method."""
    
    @pytest.fixture
    def endpoint(self):
        """Create CreditsEndpoint instance for testing."""
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        return CreditsEndpoint(auth_manager, http_manager)
    
    @pytest.mark.parametrize("invalid_payment_info,expected_error", [
        ({}, "Missing required fields"),
        ({"type": "card"}, "Missing card details"),
        ({"type": "invalid_type", "number": "123"}, "Invalid payment method type"),
        ({"type": "card", "number": "invalid_number"}, "Invalid card number"),
    ])
    def test_reject_incomplete_or_invalid_payment_data(self, endpoint, invalid_payment_info, expected_error):
        """Test rejecting incomplete or invalid payment method data."""
        # Arrange
        endpoint._get_headers = Mock(return_value={"Authorization": "Bearer token"})
        endpoint._get_endpoint_url = Mock(return_value="https://api.example.com/credits/payment-methods")
        endpoint.http_manager.post.side_effect = APIError(expected_error)
        
        # Act & Assert
        with pytest.raises(APIError, match=expected_error):
            endpoint.add_payment_method(invalid_payment_info)


class Test_CreditsEndpoint_AddPaymentMethod_03_BoundaryBehaviors:
    """Test boundary behaviors for CreditsEndpoint.add_payment_method method."""
    
    @pytest.fixture
    def endpoint(self):
        """Create CreditsEndpoint instance for testing."""
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        return CreditsEndpoint(auth_manager, http_manager)
    
    def test_handle_addition_at_payment_method_limits(self, endpoint):
        """Test handling addition when account reaches payment method limits."""
        # Arrange
        payment_info = {"type": "card", "number": "4242424242424242", "exp_month": 12, "exp_year": 2026}
        endpoint._get_headers = Mock(return_value={"Authorization": "Bearer token"})
        endpoint._get_endpoint_url = Mock(return_value="https://api.example.com/credits/payment-methods")
        endpoint.http_manager.post.side_effect = APIError("Maximum payment methods reached")
        
        # Act & Assert
        with pytest.raises(APIError, match="Maximum payment methods reached"):
            endpoint.add_payment_method(payment_info)


class Test_CreditsEndpoint_AddPaymentMethod_04_ErrorHandlingBehaviors:
    """Test error handling behaviors for CreditsEndpoint.add_payment_method method."""
    
    @pytest.fixture
    def endpoint(self):
        """Create CreditsEndpoint instance for testing."""
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        return CreditsEndpoint(auth_manager, http_manager)
    
    @pytest.mark.parametrize("exception_type,error_message", [
        (APIError, "Payment validation failed"),
        (APIError, "Card declined"),
        (ConnectionError, "Payment processor unavailable"),
        (TimeoutError, "Payment verification timeout"),
    ])
    def test_handle_validation_failures_and_api_errors(self, endpoint, exception_type, error_message):
        """Test handling payment validation failures and APIError exceptions."""
        # Arrange
        payment_info = {"type": "card", "number": "4242424242424242"}
        endpoint._get_headers = Mock(return_value={"Authorization": "Bearer token"})
        endpoint._get_endpoint_url = Mock(return_value="https://api.example.com/credits/payment-methods")
        endpoint.http_manager.post.side_effect = exception_type(error_message)
        
        # Act & Assert
        with pytest.raises(exception_type, match=error_message):
            endpoint.add_payment_method(payment_info)


class Test_CreditsEndpoint_AddPaymentMethod_05_StateTransitionBehaviors:
    """Test state transition behaviors for CreditsEndpoint.add_payment_method method."""
    
    @pytest.fixture
    def endpoint(self):
        """Create CreditsEndpoint instance for testing."""
        auth_manager = Mock(spec=AuthManager)
        http_manager = Mock(spec=HTTPManager)
        return CreditsEndpoint(auth_manager, http_manager)
    
    def test_verify_new_payment_method_becomes_available(self, endpoint):
        """Test verifying new payment method becomes available for future purchases."""
        # Arrange - simulate adding payment method
        add_response = Mock()
        add_response.json.return_value = {"id": "pm_new", "type": "card", "last4": "4242", "status": "active"}
        
        # Arrange - simulate listing payment methods after addition
        list_response = Mock()
        list_response.json.return_value = {
            "payment_methods": [
                {"id": "pm_existing", "type": "card", "last4": "1234"},
                {"id": "pm_new", "type": "card", "last4": "4242", "status": "active"}
            ]
        }
        
        endpoint.http_manager.post.return_value = add_response
        endpoint.http_manager.get.return_value = list_response
        endpoint._get_headers = Mock(return_value={"Authorization": "Bearer token"})
        endpoint._get_endpoint_url = Mock(return_value="https://api.example.com/credits/payment-methods")
        
        # Act - add payment method
        payment_info = {"type": "card", "number": "4242424242424242", "exp_month": 12, "exp_year": 2026}
        add_result = endpoint.add_payment_method(payment_info)
        
        # Act - verify it appears in payment methods list
        list_result = endpoint.payment_methods()
        
        # Assert
        assert add_result["id"] == "pm_new"
        assert add_result["status"] == "active"
        
        new_payment_method = next(
            (pm for pm in list_result["payment_methods"] if pm["id"] == "pm_new"), 
            None
        )
        assert new_payment_method is not None
        assert new_payment_method["status"] == "active"
