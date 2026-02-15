import pytest
import requests
from unittest.mock import patch, MagicMock
from src.deepseek_api.wasm_download import get_wasm_path


class TestWasmDownload:
    """Tests for the wasm_download module."""

    @patch('src.deepseek_api.wasm_download.platformdirs.user_cache_dir')
    @patch('src.deepseek_api.wasm_download.os.makedirs')
    @patch('src.deepseek_api.wasm_download.os.path.isfile')
    @patch('src.deepseek_api.wasm_download.requests.get')
    def test_download_when_file_missing(self, mock_requests_get, mock_isfile, mock_makedirs, mock_cache_dir):
        """Test that the WASM file is downloaded when not present in cache."""
        mock_cache_dir.return_value = "/fake/cache/dir"
        mock_isfile.return_value = False  # file doesn't exist

        # Mock the response from requests.get
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
        mock_requests_get.return_value = mock_response

        # Mock open
        mock_open = MagicMock()
        with patch('builtins.open', mock_open):
            result = get_wasm_path()

        # Verify directories created
        mock_makedirs.assert_called_once_with("/fake/cache/dir", exist_ok=True)

        # Verify URL requested
        mock_requests_get.assert_called_once_with(
            "https://fe-static.deepseek.com/chat/static/sha3_wasm_bg.7b9ca65ddd.wasm",
            stream=True
        )
        mock_response.raise_for_status.assert_called_once()

        # Verify file written
        mock_open.assert_called_once_with(
            "/fake/cache/dir/sha3_wasm_bg.7b9ca65ddd.wasm", "wb")
        handle = mock_open.return_value.__enter__.return_value
        handle.write.assert_any_call(b"chunk1")
        handle.write.assert_any_call(b"chunk2")

        # Verify returned path
        assert result == "/fake/cache/dir/sha3_wasm_bg.7b9ca65ddd.wasm"

    @patch('src.deepseek_api.wasm_download.platformdirs.user_cache_dir')
    @patch('src.deepseek_api.wasm_download.os.makedirs')
    @patch('src.deepseek_api.wasm_download.os.path.isfile')
    @patch('src.deepseek_api.wasm_download.requests.get')
    def test_use_existing_file(self, mock_requests_get, mock_isfile, mock_makedirs, mock_cache_dir):
        """Test that existing file is used without download."""
        mock_cache_dir.return_value = "/fake/cache/dir"
        mock_isfile.return_value = True  # file exists

        result = get_wasm_path()

        # Verify no download attempt
        mock_requests_get.assert_not_called()
        # Verify path returned
        assert result == "/fake/cache/dir/sha3_wasm_bg.7b9ca65ddd.wasm"

    @patch('src.deepseek_api.wasm_download.platformdirs.user_cache_dir')
    @patch('src.deepseek_api.wasm_download.os.makedirs')
    @patch('src.deepseek_api.wasm_download.os.path.isfile')
    @patch('src.deepseek_api.wasm_download.requests.get')
    def test_download_raises_on_http_error(self, mock_requests_get, mock_isfile, mock_makedirs, mock_cache_dir):
        """Test that download raises RuntimeError on HTTP error."""
        mock_cache_dir.return_value = "/fake/cache/dir"
        mock_isfile.return_value = False

        # Mock requests.get to raise an exception
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "404 Not Found")
        mock_requests_get.return_value = mock_response

        with pytest.raises(RuntimeError, match="Failed to download WASM file: 404 Not Found"):
            get_wasm_path()
