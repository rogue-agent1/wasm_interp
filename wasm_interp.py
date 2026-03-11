#!/usr/bin/env python3
"""wasm_interp.py — WebAssembly interpreter (core MVP subset).

Decodes and executes Wasm binary modules: types, functions, memory,
globals, stack machine with i32/i64 ops, control flow (block, loop,
br, if), and call/return.

One file. Zero deps. Does one thing well.
"""

import struct
import sys
from dataclasses import dataclass, field
from enum import IntEnum


class ValType(IntEnum):
    I32 = 0x7F
    I64 = 0x7E
    F32 = 0x7D
    F64 = 0x7C


class Op(IntEnum):
    UNREACHABLE = 0x00; NOP = 0x01; BLOCK = 0x02; LOOP = 0x03; IF = 0x04
    ELSE = 0x05; END = 0x0B; BR = 0x0C; BR_IF = 0x0D; RETURN = 0x0F
    CALL = 0x10; DROP = 0x1A; SELECT = 0x1B
    LOCAL_GET = 0x20; LOCAL_SET = 0x21; LOCAL_TEE = 0x22
    GLOBAL_GET = 0x23; GLOBAL_SET = 0x24
    I32_LOAD = 0x28; I32_STORE = 0x36
    I32_CONST = 0x41; I64_CONST = 0x42
    I32_EQZ = 0x45; I32_EQ = 0x46; I32_NE = 0x47
    I32_LT_S = 0x48; I32_GT_S = 0x4A; I32_LE_S = 0x4C; I32_GE_S = 0x4E
    I32_ADD = 0x6A; I32_SUB = 0x6B; I32_MUL = 0x6C; I32_DIV_S = 0x6D
    I32_REM_S = 0x6F; I32_AND = 0x71; I32_OR = 0x72; I32_XOR = 0x73
    I32_SHL = 0x74; I32_SHR_S = 0x75


@dataclass
class FuncType:
    params: list[int]
    results: list[int]

@dataclass
class Function:
    type_idx: int
    locals: list[int]
    body: bytes

@dataclass
class Label:
    arity: int
    target: int  # instruction position to branch to
    is_loop: bool = False


class WasmVM:
    def __init__(self):
        self.types: list[FuncType] = []
        self.functions: list[Function] = []
        self.memory = bytearray(65536)  # 1 page
        self.globals: list[int] = []
        self.exports: dict[str, tuple[str, int]] = {}
        self.stack: list[int] = []
        self.host_funcs: dict[int, callable] = {}

    def _leb128_u(self, data: bytes, pos: int) -> tuple[int, int]:
        result, shift = 0, 0
        while True:
            b = data[pos]; pos += 1
            result |= (b & 0x7F) << shift
            if not (b & 0x80): break
            shift += 7
        return result, pos

    def _leb128_s(self, data: bytes, pos: int) -> tuple[int, int]:
        result, shift = 0, 0
        while True:
            b = data[pos]; pos += 1
            result |= (b & 0x7F) << shift
            shift += 7
            if not (b & 0x80):
                if b & 0x40: result |= -(1 << shift)
                break
        return result, pos

    def load_binary(self, data: bytes):
        if data[:4] != b'\x00asm': raise ValueError("Not a Wasm module")
        pos = 8  # skip magic + version
        while pos < len(data):
            sec_id = data[pos]; pos += 1
            sec_len, pos = self._leb128_u(data, pos)
            sec_end = pos + sec_len
            if sec_id == 1:    self._decode_types(data, pos, sec_end)
            elif sec_id == 3:  self._decode_funcs(data, pos, sec_end)
            elif sec_id == 7:  self._decode_exports(data, pos, sec_end)
            elif sec_id == 10: self._decode_code(data, pos, sec_end)
            pos = sec_end

    def _decode_types(self, data, pos, end):
        count, pos = self._leb128_u(data, pos)
        for _ in range(count):
            assert data[pos] == 0x60; pos += 1
            nparams, pos = self._leb128_u(data, pos)
            params = list(data[pos:pos+nparams]); pos += nparams
            nresults, pos = self._leb128_u(data, pos)
            results = list(data[pos:pos+nresults]); pos += nresults
            self.types.append(FuncType(params, results))

    def _decode_funcs(self, data, pos, end):
        count, pos = self._leb128_u(data, pos)
        for _ in range(count):
            idx, pos = self._leb128_u(data, pos)
            self.functions.append(Function(idx, [], b''))

    def _decode_exports(self, data, pos, end):
        count, pos = self._leb128_u(data, pos)
        for _ in range(count):
            name_len, pos = self._leb128_u(data, pos)
            name = data[pos:pos+name_len].decode(); pos += name_len
            kind = data[pos]; pos += 1
            idx, pos = self._leb128_u(data, pos)
            kinds = {0: 'func', 1: 'table', 2: 'memory', 3: 'global'}
            self.exports[name] = (kinds.get(kind, '?'), idx)

    def _decode_code(self, data, pos, end):
        count, pos = self._leb128_u(data, pos)
        for i in range(count):
            body_size, pos = self._leb128_u(data, pos)
            body_end = pos + body_size
            # Locals
            nlocal_decls, pos = self._leb128_u(data, pos)
            locals_list = []
            for _ in range(nlocal_decls):
                n, pos = self._leb128_u(data, pos)
                t = data[pos]; pos += 1
                locals_list.extend([t] * n)
            self.functions[i].locals = locals_list
            self.functions[i].body = data[pos:body_end]
            pos = body_end

    def call(self, func_idx: int, args: list[int]) -> list[int]:
        if func_idx in self.host_funcs:
            return self.host_funcs[func_idx](*args)
        func = self.functions[func_idx]
        ftype = self.types[func.type_idx]
        # Set up locals
        locals_ = list(args) + [0] * len(func.locals)
        return self._execute(func.body, locals_, ftype.results)

    def _execute(self, code: bytes, locals_: list, result_types: list) -> list[int]:
        stack: list[int] = []
        labels: list[Label] = []
        pos = 0

        def i32(v): return v & 0xFFFFFFFF
        def i32s(v):
            v = v & 0xFFFFFFFF
            return v - 0x100000000 if v >= 0x80000000 else v

        while pos < len(code):
            op = code[pos]; pos += 1
            if op == Op.NOP: pass
            elif op == Op.UNREACHABLE: raise RuntimeError("unreachable")
            elif op == Op.BLOCK:
                bt = code[pos]; pos += 1
                arity = 0 if bt == 0x40 else 1
                labels.append(Label(arity, self._find_end(code, pos), False))
            elif op == Op.LOOP:
                bt = code[pos]; pos += 1
                labels.append(Label(0, pos - 2, True))  # loop branches to start
            elif op == Op.IF:
                bt = code[pos]; pos += 1
                arity = 0 if bt == 0x40 else 1
                cond = stack.pop()
                end_pos = self._find_end(code, pos)
                else_pos = self._find_else(code, pos, end_pos)
                if cond:
                    labels.append(Label(arity, end_pos, False))
                else:
                    if else_pos is not None:
                        pos = else_pos
                        labels.append(Label(arity, end_pos, False))
                    else:
                        pos = end_pos
            elif op == Op.ELSE:
                if labels:
                    pos = labels[-1].target  # jump to end
                    labels.pop()
            elif op == Op.END:
                if labels: labels.pop()
            elif op == Op.BR:
                depth, pos = self._leb128_u(code, pos)
                if depth < len(labels):
                    label = labels[-(depth+1)]
                    labels = labels[:len(labels)-depth-1]
                    if label.is_loop: pos = label.target + 2
                    else: pos = label.target
            elif op == Op.BR_IF:
                depth, pos = self._leb128_u(code, pos)
                cond = stack.pop()
                if cond:
                    if depth < len(labels):
                        label = labels[-(depth+1)]
                        labels = labels[:len(labels)-depth-1]
                        if label.is_loop: pos = label.target + 2
                        else: pos = label.target
            elif op == Op.RETURN:
                break
            elif op == Op.CALL:
                idx, pos = self._leb128_u(code, pos)
                ftype = self.types[self.functions[idx].type_idx]
                args = [stack.pop() for _ in ftype.params][::-1]
                result = self.call(idx, args)
                stack.extend(result)
            elif op == Op.DROP: stack.pop()
            elif op == Op.SELECT:
                c = stack.pop(); b = stack.pop(); a = stack.pop()
                stack.append(a if c else b)
            elif op == Op.LOCAL_GET:
                idx, pos = self._leb128_u(code, pos); stack.append(locals_[idx])
            elif op == Op.LOCAL_SET:
                idx, pos = self._leb128_u(code, pos); locals_[idx] = stack.pop()
            elif op == Op.LOCAL_TEE:
                idx, pos = self._leb128_u(code, pos); locals_[idx] = stack[-1]
            elif op == Op.I32_CONST:
                val, pos = self._leb128_s(code, pos); stack.append(i32(val))
            elif op == Op.I64_CONST:
                val, pos = self._leb128_s(code, pos); stack.append(val)
            elif op == Op.I32_EQZ: stack.append(1 if stack.pop() == 0 else 0)
            elif op == Op.I32_EQ: b, a = stack.pop(), stack.pop(); stack.append(1 if a == b else 0)
            elif op == Op.I32_NE: b, a = stack.pop(), stack.pop(); stack.append(1 if a != b else 0)
            elif op == Op.I32_LT_S: b, a = stack.pop(), stack.pop(); stack.append(1 if i32s(a) < i32s(b) else 0)
            elif op == Op.I32_GT_S: b, a = stack.pop(), stack.pop(); stack.append(1 if i32s(a) > i32s(b) else 0)
            elif op == Op.I32_LE_S: b, a = stack.pop(), stack.pop(); stack.append(1 if i32s(a) <= i32s(b) else 0)
            elif op == Op.I32_GE_S: b, a = stack.pop(), stack.pop(); stack.append(1 if i32s(a) >= i32s(b) else 0)
            elif op == Op.I32_ADD: b, a = stack.pop(), stack.pop(); stack.append(i32(a + b))
            elif op == Op.I32_SUB: b, a = stack.pop(), stack.pop(); stack.append(i32(a - b))
            elif op == Op.I32_MUL: b, a = stack.pop(), stack.pop(); stack.append(i32(a * b))
            elif op == Op.I32_DIV_S:
                b, a = stack.pop(), stack.pop()
                if b == 0: raise RuntimeError("division by zero")
                stack.append(i32(int(i32s(a) / i32s(b))))
            elif op == Op.I32_REM_S:
                b, a = stack.pop(), stack.pop()
                if b == 0: raise RuntimeError("division by zero")
                stack.append(i32(i32s(a) % i32s(b)))
            elif op == Op.I32_AND: b, a = stack.pop(), stack.pop(); stack.append(i32(a & b))
            elif op == Op.I32_OR: b, a = stack.pop(), stack.pop(); stack.append(i32(a | b))
            elif op == Op.I32_XOR: b, a = stack.pop(), stack.pop(); stack.append(i32(a ^ b))
            elif op == Op.I32_SHL: b, a = stack.pop(), stack.pop(); stack.append(i32(a << (b & 31)))
            elif op == Op.I32_SHR_S:
                b, a = stack.pop(), stack.pop()
                stack.append(i32(i32s(a) >> (b & 31)))
            elif op == Op.I32_LOAD:
                _align = code[pos]; pos += 1
                offset, pos = self._leb128_u(code, pos)
                addr = stack.pop() + offset
                stack.append(struct.unpack_from('<I', self.memory, addr)[0])
            elif op == Op.I32_STORE:
                _align = code[pos]; pos += 1
                offset, pos = self._leb128_u(code, pos)
                val = stack.pop(); addr = stack.pop() + offset
                struct.pack_into('<I', self.memory, addr, val & 0xFFFFFFFF)
            else:
                raise RuntimeError(f"Unknown opcode: 0x{op:02x} at pos {pos-1}")

        return stack[-len(result_types):] if result_types else []

    def _find_end(self, code, pos) -> int:
        depth = 1
        while pos < len(code) and depth > 0:
            op = code[pos]; pos += 1
            if op in (0x02, 0x03, 0x04): pos += 1; depth += 1
            elif op == 0x0B: depth -= 1
            elif op == 0x41: _, pos = self._leb128_s(code, pos)
            elif op == 0x42: _, pos = self._leb128_s(code, pos)
            elif op in (0x0C, 0x0D, 0x10, 0x20, 0x21, 0x22, 0x23, 0x24):
                _, pos = self._leb128_u(code, pos)
            elif op in (0x28, 0x36): pos += 1; _, pos = self._leb128_u(code, pos)
        return pos

    def _find_else(self, code, pos, end) -> int | None:
        depth = 1
        while pos < end:
            op = code[pos]; pos += 1
            if op in (0x02, 0x03, 0x04): pos += 1; depth += 1
            elif op == 0x05 and depth == 1: return pos
            elif op == 0x0B: depth -= 1; 
            elif op == 0x41: _, pos = self._leb128_s(code, pos)
            elif op == 0x42: _, pos = self._leb128_s(code, pos)
            elif op in (0x0C, 0x0D, 0x10, 0x20, 0x21, 0x22, 0x23, 0x24):
                _, pos = self._leb128_u(code, pos)
            elif op in (0x28, 0x36): pos += 1; _, pos = self._leb128_u(code, pos)
            if depth == 0: break
        return None

    def invoke(self, name: str, args: list[int]) -> list[int]:
        if name not in self.exports:
            raise ValueError(f"Export '{name}' not found")
        kind, idx = self.exports[name]
        if kind != 'func':
            raise ValueError(f"Export '{name}' is {kind}, not func")
        return self.call(idx, args)


def build_add_module() -> bytes:
    """Build a minimal Wasm module: (func $add (param i32 i32) (result i32) local.get 0 local.get 1 i32.add)"""
    m = bytearray()
    m += b'\x00asm\x01\x00\x00\x00'  # magic + version
    # Type section: (i32, i32) -> i32
    m += bytes([1, 7, 1, 0x60, 2, 0x7F, 0x7F, 1, 0x7F])
    # Function section
    m += bytes([3, 2, 1, 0])
    # Export section: "add" -> func 0
    m += bytes([7, 7, 1, 3]) + b'add' + bytes([0, 0])
    # Code section
    body = bytes([0x20, 0, 0x20, 1, 0x6A, 0x0B])  # local.get 0, local.get 1, i32.add, end
    m += bytes([10, len(body) + 3, 1, len(body) + 1, 0]) + body
    return bytes(m)


def build_fib_module() -> bytes:
    """Build: (func $fib (param i32) (result i32) ...)"""
    m = bytearray()
    m += b'\x00asm\x01\x00\x00\x00'
    # Type: (i32) -> (i32)
    m += bytes([1, 6, 1, 0x60, 1, 0x7F, 1, 0x7F])
    # Func
    m += bytes([3, 2, 1, 0])
    # Export "fib"
    m += bytes([7, 7, 1, 3]) + b'fib' + bytes([0, 0])
    # Code: if n <= 1 return n, else fib(n-1) + fib(n-2)
    body = bytearray()
    body += bytes([0x20, 0, 0x41, 2, 0x48])  # local.get 0; i32.const 2; i32.lt_s
    body += bytes([0x04, 0x7F])  # if (result i32)
    body += bytes([0x20, 0])  # local.get 0
    body += bytes([0x05])  # else
    body += bytes([0x20, 0, 0x41, 1, 0x6B, 0x10, 0])  # fib(n-1)
    body += bytes([0x20, 0, 0x41, 2, 0x6B, 0x10, 0])  # fib(n-2)
    body += bytes([0x6A])  # i32.add
    body += bytes([0x0B])  # end if
    body += bytes([0x0B])  # end func
    m += bytes([10, len(body) + 3, 1, len(body) + 1, 0]) + body
    return bytes(m)


def demo():
    print("=== WebAssembly Interpreter ===\n")
    vm = WasmVM()
    vm.load_binary(build_add_module())
    result = vm.invoke("add", [17, 25])
    print(f"add(17, 25) = {result[0]}")

    vm2 = WasmVM()
    vm2.load_binary(build_fib_module())
    for n in range(10):
        r = vm2.invoke("fib", [n])
        print(f"  fib({n}) = {r[0]}")


if __name__ == '__main__':
    if '--test' in sys.argv:
        vm = WasmVM()
        vm.load_binary(build_add_module())
        assert vm.invoke("add", [3, 4]) == [7]
        assert vm.invoke("add", [0, 0]) == [0]
        assert vm.invoke("add", [100, 200]) == [300]
        vm2 = WasmVM()
        vm2.load_binary(build_fib_module())
        fibs = [vm2.invoke("fib", [n])[0] for n in range(8)]
        assert fibs == [0, 1, 1, 2, 3, 5, 8, 13]
        print("All tests passed ✓")
    else:
        demo()
