import pytest
from unittest.mock import Mock, patch


@pytest.fixture
def mock_requests_session():
    """Fixture to mock requests.Session."""
    with patch('requests.Session') as mock_session_class:
        mock_session = Mock()
        mock_session.headers = {}
        mock_session_class.return_value = mock_session
        yield mock_session


@pytest.fixture
def mock_pow_solver():
    """Fixture to provide a mock POWSolver."""
    mock = Mock()
    mock.solve_challenge.return_value = "mock_pow_response"
    return mock


@pytest.fixture
def sample_challenge():
    """Fixture providing a sample POW challenge dict."""
    return {
        "algorithm": "SHA3-256",
        "challenge": "test_challenge",
        "salt": "test_salt",
        "signature": "test_signature",
        "target_path": "/api/v0/chat/completion",
        "difficulty": 1000000,
        "expire_at": 1740000000
    }


@pytest.fixture
def sample_chat_session():
    """Fixture providing a sample chat session response."""
    return {
        "code": 0,
        "data": {
            "biz_data": {
                "chat_session": {
                    "id": "test_chat_id",
                    "title": "Test Chat"
                }
            }
        }
    }
