import pytest
import json
import base64
import numpy as np
from unittest.mock import patch, MagicMock, Mock
from src.deepseek_api.pow_solve import POWSolver


class TestPOWSolver:
    """Tests for the POWSolver class."""

    def test_init_without_path(self):
        """Test initialization when no wasm path provided."""
        with patch('src.deepseek_api.pow_solve.get_wasm_path') as mock_get_path, \
                patch('builtins.open', create=True) as mock_open, \
                patch('wasmtime.Module') as mock_module, \
                patch('wasmtime.Instance') as mock_instance, \
                patch('wasmtime.Store') as mock_store:

            mock_get_path.return_value = "/fake/path.wasm"
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            mock_file.read.return_value = b"fake_wasm_bytes"

            solver = POWSolver()

            mock_get_path.assert_called_once()
            mock_open.assert_called_once_with("/fake/path.wasm", "rb")
            # Verify wasmtime components were created
            assert solver.memory is not None
            assert solver.wasm_solve is not None
            assert solver.alloc is not None
            assert solver.add_stack is not None

    def test_init_with_path(self):
        """Test initialization with explicit wasm path."""
        with patch('builtins.open', create=True) as mock_open, \
                patch('wasmtime.Module') as mock_module, \
                patch('wasmtime.Instance') as mock_instance, \
                patch('wasmtime.Store') as mock_store:

            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            mock_file.read.return_value = b"fake_wasm_bytes"

            solver = POWSolver("/custom/path.wasm")

            mock_open.assert_called_once_with("/custom/path.wasm", "rb")

    def test_write_str_to_memory(self):
        """Test _write_str_to_memory writes string correctly and returns pointer/length."""
        with patch('src.deepseek_api.pow_solve.get_wasm_path') as mock_get_path, \
                patch('builtins.open', create=True) as mock_open, \
                patch('wasmtime.Module'), patch('wasmtime.Instance'), patch('wasmtime.Store'):

            mock_get_path.return_value = "/fake/path.wasm"
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            mock_file.read.return_value = b"fake_wasm_bytes"

            solver = POWSolver()
            solver.alloc = Mock(return_value=12345)
            # Mock memory data
            mock_memory = Mock()
            solver.memory = mock_memory
            # Create a large bytearray to accommodate pointer
            mock_data_ptr = bytearray(20000)
            mock_memory.data_ptr.return_value = mock_data_ptr
            # Simulate writing to memory by checking that we set bytes correctly
            test_str = "hello"
            ptr, length = solver._write_str_to_memory(test_str)

            solver.alloc.assert_called_once_with(
                solver.store, len(test_str), 1)
            assert ptr == 12345
            assert length == 5
            # Verify bytes were written at the correct offset
            assert mock_data_ptr[12345:12345+5] == b'hello'

    def test_solve_challenge_success(self, sample_challenge):
        """Test solve_challenge successfully computes answer and returns base64."""
        with patch('src.deepseek_api.pow_solve.get_wasm_path') as mock_get_path, \
                patch('builtins.open', create=True) as mock_open, \
                patch('wasmtime.Module'), patch('wasmtime.Instance'), patch('wasmtime.Store'):

            mock_get_path.return_value = "/fake/path.wasm"
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            mock_file.read.return_value = b"fake_wasm_bytes"

            solver = POWSolver()
            # Make add_stack return a concrete integer
            solver.add_stack = Mock(return_value=500)
            solver._write_str_to_memory = Mock(
                side_effect=[(200, 10), (300, 15)])
            solver.wasm_solve = Mock()
            # Mock memory data for reading results
            mock_memory = Mock()
            solver.memory = mock_memory
            # Create a large bytearray to accommodate out_ptr
            mem_bytes = bytearray(1024)
            # status = 1 (non-zero success) as little-endian int32 at offset 500
            mem_bytes[500:504] = (1, 0, 0, 0)
            # answer = 12345.0 as float64 little-endian at offset 508
            mem_bytes[508:516] = bytes(
                [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc8, 0x40])
            mock_memory.data_ptr.return_value = mem_bytes

            with patch('numpy.frombuffer') as mock_np_frombuffer:
                mock_np_frombuffer.return_value = np.array([12345.0])
                result_b64 = solver.solve_challenge(sample_challenge)

            # Verify calls
            solver.add_stack.assert_any_call(solver.store, -16)  # allocate
            # Verify wasm_solve called with correct args
            solver.wasm_solve.assert_called_once_with(
                solver.store,
                500,  # out_ptr from add_stack return value
                200,   # challenge_ptr
                10,    # challenge_len
                300,   # prefix_ptr
                15,    # prefix_len
                float(sample_challenge["difficulty"])
            )
            # Verify result
            result = json.loads(base64.b64decode(result_b64).decode())
            assert result["algorithm"] == sample_challenge["algorithm"]
            assert result["challenge"] == sample_challenge["challenge"]
            assert result["salt"] == sample_challenge["salt"]
            assert result["answer"] == 12345
            assert result["signature"] == sample_challenge["signature"]
            assert result["target_path"] == sample_challenge["target_path"]
            # Verify cleanup
            solver.add_stack.assert_any_call(solver.store, 16)

    def test_solve_challenge_failure_status_zero(self, sample_challenge):
        """Test solve_challenge raises assertion when status is zero."""
        with patch('src.deepseek_api.pow_solve.get_wasm_path') as mock_get_path, \
                patch('builtins.open', create=True) as mock_open, \
                patch('wasmtime.Module'), patch('wasmtime.Instance'), patch('wasmtime.Store'):

            mock_get_path.return_value = "/fake/path.wasm"
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            mock_file.read.return_value = b"fake_wasm_bytes"

            solver = POWSolver()
            solver.add_stack = Mock(return_value=500)
            solver._write_str_to_memory = Mock(return_value=(200, 10))
            solver.wasm_solve = Mock()
            # Mock memory data with status = 0
            mock_memory = Mock()
            solver.memory = mock_memory
            # all zeros (status at offset 500 will be zero)
            mem_bytes = bytearray(1024)
            mock_memory.data_ptr.return_value = mem_bytes

            with pytest.raises(AssertionError):
                solver.solve_challenge(sample_challenge)

            # Cleanup should still happen
            solver.add_stack.assert_any_call(solver.store, 16)
