import os
import requests
import sys
import platformdirs

def get_wasm_path():
    """
    Returns the local filesystem path to the DeepSeek WASM module.
    Downloads the WASM file from a remote URL if it is not already present
    in the user's cache directory.
    """
    cache_dir = platformdirs.user_cache_dir("deepseek")
    os.makedirs(cache_dir, exist_ok=True)

    wasm_filename = "sha3_wasm_bg.7b9ca65ddd.wasm"
    local_path = os.path.join(cache_dir, wasm_filename)

    # If the file already exists, return its path immediately
    if os.path.isfile(local_path):
        return local_path

    url = f"https://fe-static.deepseek.com/chat/static/{wasm_filename}"

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    except Exception as e:
        raise RuntimeError(f"Failed to download WASM file: {e}") from e

    return local_path