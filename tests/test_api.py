import pytest
import json
from unittest.mock import Mock, patch, call
from src.deepseek_api.api import DeepSeekAPI


class TestDeepSeekAPI:
    """Tests for the DeepSeekAPI class."""

    def test_init(self, mock_pow_solver):
        """Test initialization sets up session and headers correctly."""
        api = DeepSeekAPI("test_token", mock_pow_solver)
        assert api.session.headers["authorization"] == "Bearer test_token"
        assert api.session.headers["Content-Type"] == "application/json"
        assert api.pow_solver == mock_pow_solver

    def test_create_chat_success(self, mock_requests_session, mock_pow_solver):
        """Test create_chat returns chat data on success."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": {"biz_data": {"id": "chat123", "title": "New Chat"}}
        }
        mock_requests_session.post.return_value = mock_response

        api = DeepSeekAPI("token", mock_pow_solver)
        result = api.create_chat()

        mock_requests_session.post.assert_called_once_with(
            "https://chat.deepseek.com/api/v0/chat_session/create", "{}"
        )
        assert result == {"id": "chat123", "title": "New Chat"}

    def test_get_chat_info_success(self, mock_requests_session, mock_pow_solver, sample_chat_session):
        """Test get_chat_info returns chat info on success."""
        mock_response = Mock()
        mock_response.json.return_value = sample_chat_session
        mock_requests_session.get.return_value = mock_response

        api = DeepSeekAPI("token", mock_pow_solver)
        result = api.get_chat_info("test_chat_id")

        mock_requests_session.get.assert_called_once_with(
            "https://chat.deepseek.com/api/v0/chat/history_messages?chat_session_id=test_chat_id"
        )
        assert result == {"id": "test_chat_id", "title": "Test Chat"}

    def test_get_chat_info_error(self, mock_requests_session, mock_pow_solver):
        """Test get_chat_info raises exception on API error."""
        mock_response = Mock()
        mock_response.json.return_value = {"code": 1, "msg": "Some error"}
        mock_requests_session.get.return_value = mock_response

        api = DeepSeekAPI("token", mock_pow_solver)
        with pytest.raises(Exception, match="Failed to get chat info: Some error"):
            api.get_chat_info("bad_id")

    def test_set_pow_header(self, mock_requests_session, mock_pow_solver, sample_challenge):
        """Test _set_pow_header sends challenge request and sets header."""
        # Mock the challenge request response
        challenge_response = Mock()
        challenge_response.json.return_value = {
            "data": {"biz_data": {"challenge": sample_challenge}}
        }
        mock_requests_session.post.return_value = challenge_response

        api = DeepSeekAPI("token", mock_pow_solver)
        api._set_pow_header()

        # Verify POST to create_pow_challenge
        mock_requests_session.post.assert_called_once_with(
            "https://chat.deepseek.com/api/v0/chat/create_pow_challenge",
            json.dumps({"target_path": "/api/v0/chat/completion"})
        )
        # Verify solver was called with challenge
        mock_pow_solver.solve_challenge.assert_called_once_with(
            sample_challenge)
        # Verify header was set
        assert api.session.headers["x-ds-pow-response"] == mock_pow_solver.solve_challenge.return_value

    @patch('src.deepseek_api.api.DeepSeekAPI._set_pow_header')
    def test_complete_non_streaming(self, mock_set_header, mock_requests_session, mock_pow_solver):
        """Test complete method in non-streaming mode."""
        # Mock the streaming response (simulate SSE lines)
        mock_response = Mock()
        # Simulate two data lines then finish
        mock_response.iter_lines.return_value = [
            b'data: {"v": {"response": {"content": "Hello"}}, "p": "response/content", "o": "SET"}',
            b'data: {"v": " world", "p": "response/content", "o": "APPEND"}',
            b'event: finish'
        ]
        mock_requests_session.post.return_value = mock_response

        api = DeepSeekAPI("token", mock_pow_solver)
        result = api.complete(
            "chat_id", "Hello", parent_message_id=123, search=True, thinking=False)

        # Verify _set_pow_header called
        mock_set_header.assert_called_once()
        # Verify POST request
        expected_payload = {
            "chat_session_id": "chat_id",
            "prompt": "Hello",
            "parent_message_id": 123,
            "ref_file_ids": [],
            "search_enabled": True,
            "thinking_enabled": False
        }
        mock_requests_session.post.assert_called_once_with(
            "https://chat.deepseek.com/api/v0/chat/completion",
            json.dumps(expected_payload),
            stream=True
        )
        # Verify final result (should be dict with content)
        assert result == {"content": "Hello world"}

    @patch('src.deepseek_api.api.DeepSeekAPI._set_pow_header')
    def test_complete_stream(self, mock_set_header, mock_requests_session, mock_pow_solver):
        """Test complete_stream generator yields correct chunks."""
        # Mock streaming response with both content and thinking chunks
        mock_response = Mock()
        mock_response.iter_lines.return_value = [
            # initial full message
            b'data: {"v": {"response": {"content": "", "thinking_content": ""}}}',
            b'data: {"v": "I am ", "p": "response/thinking_content", "o": "APPEND"}',
            b'data: {"v": "thinking", "p": "response/thinking_content", "o": "APPEND"}',
            b'data: {"v": "Hello", "p": "response/content", "o": "APPEND"}',
            b'data: {"v": " world", "p": "response/content", "o": "APPEND"}',
            b'event: finish'
        ]
        mock_requests_session.post.return_value = mock_response

        api = DeepSeekAPI("token", mock_pow_solver)
        chunks = list(api.complete_stream(
            "chat_id", "Hello", search=False, thinking=True))

        # Verify chunks (updated to match actual implementation)
        expected_chunks = [
            {"type": "thinking", "content": "I am "},
            {"type": "thinking", "content": "thinking"},
            {"type": "content", "content": "Hello"},
            {"type": "content", "content": " world"},
            {"type": "message", "content": {
                "content": "Hello world", "thinking_content": "I am thinking"}}
        ]
        assert chunks == expected_chunks

    def test_handle_property_update_set(self):
        """Test _handle_property_update with SET operation."""
        api = DeepSeekAPI("token", None)  # pow_solver not needed for this test
        obj = {"a": {"b": "old"}}
        update = {"p": "a/b", "v": "new", "o": "SET"}
        result = api._handle_property_update(obj, update)
        assert result is True
        assert obj["a"]["b"] == "new"

    def test_handle_property_update_append(self):
        """Test _handle_property_update with APPEND operation."""
        api = DeepSeekAPI("token", None)
        obj = {"a": {"b": "hello"}}
        update = {"p": "a/b", "v": " world", "o": "APPEND"}
        result = api._handle_property_update(obj, update)
        assert result is True
        assert obj["a"]["b"] == "hello world"

    def test_handle_property_update_append_new_key(self):
        """Test APPEND when key doesn't exist (should create empty string)."""
        api = DeepSeekAPI("token", None)
        obj = {"a": {}}
        update = {"p": "a/b", "v": "new", "o": "APPEND"}
        result = api._handle_property_update(obj, update)
        assert result is True
        assert obj["a"]["b"] == "new"

    def test_handle_property_update_invalid_op(self):
        """Test _handle_property_update with unknown operation returns False."""
        api = DeepSeekAPI("token", None)
        obj = {"a": {"b": "old"}}
        update = {"p": "a/b", "v": "new", "o": "INVALID"}
        result = api._handle_property_update(obj, update)
        assert result is False
        assert obj["a"]["b"] == "old"  # unchanged

    def test_handle_property_update_path_not_dict(self):
        """Test _handle_property_update when path intermediate is not dict."""
        api = DeepSeekAPI("token", None)
        obj = {"a": "not a dict"}
        update = {"p": "a/b", "v": "new", "o": "SET"}
        result = api._handle_property_update(obj, update)
        assert result is False
