from pow_solve import POWSolver
import requests
import json

COMPLETION_PATH = "/api/v0/chat/completion"
POW_REQUEST = json.dumps({"target_path": COMPLETION_PATH})


class DeepSeekAPI:
    def __init__(self, token: str, pow_solver: POWSolver):
        self.session = requests.Session()
        self.session.headers["authorization"] = f"Bearer {token}"
        self.session.headers["Content-Type"] = "application/json"
        self.pow_solver = pow_solver

    def create_chat(self):
        r = self.session.post(
            "https://chat.deepseek.com/api/v0/chat_session/create", "{}")
        chat = r.json()["data"]["biz_data"]
        return chat

    def get_chat_info(self, chat_id: str):
        r = self.session.get(f"https://chat.deepseek.com/api/v0/chat/history_messages?chat_session_id={chat_id}")
        data = r.json()
        if data.get("code") != 0:
            raise Exception(f"Failed to get chat info: {data.get('msg')}")
        return data["data"]["biz_data"]["chat_session"]

    def _set_pow_header(self):
        r = self.session.post(
            "https://chat.deepseek.com/api/v0/chat/create_pow_challenge", POW_REQUEST)
        challenge = r.json()["data"]["biz_data"]["challenge"]
        self.session.headers["x-ds-pow-response"] = self.pow_solver.solve_challenge(
            challenge)

    def complete(self, chat_id: str, prompt: str, parent_message_id: int = None, search=False, thinking=False) -> str:
        self._set_pow_header()
        request = {
            "chat_session_id": chat_id,
            "prompt": prompt,
            "parent_message_id": parent_message_id,
            "ref_file_ids": [],
            "search_enabled": search,
            "thinking_enabled": thinking
        }
        r = self.session.post(
            f"https://chat.deepseek.com{COMPLETION_PATH}", json.dumps(request), stream=True)
        message = {}
        current_property = None
        for line in r.iter_lines():
            if line == b"event: finish":
                break
            if not line.startswith(b"data: "):
                continue
            data: dict = json.loads(line[6:])
            v = data.get("v")
            if v is None:
                continue
            if isinstance(v, dict):  # received the initial response
                message = v
                continue

            path: str = data.get("p")
            if path is None:  # append to current property
                data["p"] = current_property
                data["o"] = "APPEND"
            self._handle_property_update(message, data)
            current_property = data["p"]

        return message["response"]

    def complete_stream(self, chat_id: str, prompt: str, parent_message_id: int = None, search=False, thinking=False):
        """Generator that yields chunks of the streaming response.
        Each chunk is a dict with 'type' ('content' or 'thinking') and 'content' (the incremental string).
        """
        self._set_pow_header()
        request = {
            "chat_session_id": chat_id,
            "prompt": prompt,
            "parent_message_id": parent_message_id,
            "ref_file_ids": [],
            "search_enabled": search,
            "thinking_enabled": thinking
        }
        r = self.session.post(
            f"https://chat.deepseek.com{COMPLETION_PATH}", json.dumps(request), stream=True)
        message = {}
        current_property = None
        for line in r.iter_lines():
            if line == b"event: finish":
                break
            if not line.startswith(b"data: "):
                continue
            data: dict = json.loads(line[6:])
            v = data.get("v")
            if v is None:
                continue
            if isinstance(v, dict):  # initial full message
                message = v
                continue

            path: str = data.get("p")
            if path is None:
                # continuation of previous property
                path = current_property
                data["p"] = path          # ensure 'p' exists for _handle_property_update
                data["o"] = "APPEND"       # set operation to APPEND
            else:
                # new property, capture current_property
                current_property = path

            # Update internal state (optional)
            self._handle_property_update(message, data)

            # Yield incremental content
            if path == "response/content":
                yield {"type": "content", "content": v}
            elif path == "response/thinking_content":
                yield {"type": "thinking", "content": v}
        yield {"type": "message", "content": message["response"]}

    def _handle_property_update(self, obj: dict, update: dict):
        keys = update["p"].split("/")
        data = obj.copy()
        for key in keys[:-1]:
            if not isinstance(data.get(key), dict):
                return False
            data = data[key]
        last_key = keys[-1]
        match update.get("o", "SET"):
            case "SET":
                data[last_key] = update["v"]
            case "APPEND":
                # Ensure the key exists before appending
                if last_key not in data:
                    data[last_key] = ""
                data[last_key] += update["v"]
            case _:
                return False
        return True