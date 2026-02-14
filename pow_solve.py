import wasmtime
import numpy as np
import base64
import json


class POWSolver:
    def __init__(self, wasm_path: str):
        with open(wasm_path, "rb") as f:
            wat = f.read()

        engine = wasmtime.Engine()
        module = wasmtime.Module(engine, wat)
        self.store = wasmtime.Store(engine)
        instance = wasmtime.Instance(self.store, module, [])

        self.memory = instance.exports(self.store)["memory"]
        self.wasm_solve = instance.exports(self.store)["wasm_solve"]
        self.alloc = instance.exports(self.store)["__wbindgen_export_0"]
        self.add_stack = instance.exports(
            self.store)["__wbindgen_add_to_stack_pointer"]

    def _write_str_to_memory(self, data: str):
        enc = data.encode()
        ptr = self.alloc(self.store, len(enc), 1)

        mem = self.memory.data_ptr(self.store)
        for i, byte in enumerate(enc):
            mem[ptr + i] = byte
        return ptr, len(enc)

    def solve_challenge(self, challenge: dict):
        # allocate 16 bytes for output
        out_ptr = self.add_stack(self.store, -16)
        try:
            prefix = f"{challenge["salt"]}_{challenge["expire_at"]}_"
            challenge_ptr, challenge_len = self._write_str_to_memory(
                challenge["challenge"])
            prefix_ptr, prefix_len = self._write_str_to_memory(prefix)

            self.wasm_solve(self.store,
                            out_ptr,
                            challenge_ptr,
                            challenge_len,
                            prefix_ptr,
                            prefix_len,
                            float(challenge["difficulty"]))

            mem = self.memory.data_ptr(self.store)
            status = int.from_bytes(
                bytes(mem[out_ptr:out_ptr + 4]), byteorder='little', signed=True)

            assert status != 0

            value_bytes = bytes(mem[out_ptr + 8:out_ptr + 16])
            value = np.frombuffer(value_bytes, dtype=np.float64)[0]
            result = {
                "algorithm": challenge["algorithm"],
                "challenge": challenge["challenge"],
                "salt": challenge["salt"],
                "answer": int(value),
                "signature": challenge["signature"],
                "target_path": challenge["target_path"]
            }

            return base64.b64encode(json.dumps(result).encode()).decode()
        finally:
            self.add_stack(self.store, 16)  # cleanup
