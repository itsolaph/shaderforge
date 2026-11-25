"""Headless GLSL renderer for Shadertoy-style ``mainImage`` shaders.

* Uses moderngl with a standalone EGL context (Mesa llvmpipe on CPU boxes,
  the real GPU driver on GPU boxes), falling back to osmesa, then the default
  backend.
* Shaders run in a *persistent worker subprocess* (one GL context, no per-shader
  process spawn). A hung shader makes the parent kill + respawn the worker.
* The wrapper uses GLSL ES 3.00 (``#version 300 es``) -- the same dialect WebGL2
  speaks -- so a shader that scores in training also runs in ``web/``. If the
  local driver refuses ES 3.00 we fall back to ``#version 330 core`` and report it.

Image convention: returned arrays are (H, W, 3) uint8, row 0 = TOP of the image
(i.e. GL framebuffer flipped), identical to what a browser canvas shows.
"""
from __future__ import annotations

import atexit
import base64  # noqa: F401  (kept for debugging helpers)
import json
import os
import queue
import select
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, NamedTuple, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PRELUDE_ES = (
    "#version 300 es\n"
    "precision highp float;\n"
    "precision highp int;\n"
    "uniform vec3 iResolution;\n"
    "uniform float iTime;\n"
    "out vec4 _sf_out;\n"
)
EPILOGUE = (
    "\nvoid main() {\n"
    "    vec4 c = vec4(0.0, 0.0, 0.0, 1.0);\n"
    "    mainImage(c, gl_FragCoord.xy);\n"
    "    _sf_out = vec4(c.rgb, 1.0);\n"
    "}\n"
)
VERTEX_ES = (
    "#version 300 es\n"
    "precision highp float;\n"
    "void main() {\n"
    "    vec2 q = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));\n"
    "    gl_Position = vec4(q * 2.0 - 1.0, 0.0, 1.0);\n"
    "}\n"
)


def wrap_fragment(code: str, mode: str = "es") -> str:
    pre = PRELUDE_ES if mode == "es" else PRELUDE_ES.replace("#version 300 es", "#version 330 core")
    return pre + code + EPILOGUE


GLSLANG = shutil.which("glslangValidator")


def strict_check(code: str) -> Optional[str]:
    """Reference-compiler gate (Khronos glslang, GLSL ES 3.00).

    Mesa/llvmpipe builds differ in how lenient they are; an old one accepted
    ``float x = vec2(...)``, and GRPO learned to exploit that. Every shader must also pass
    the reference front end, so the reward cannot depend on one driver's quirks.
    Returns an error string, or None when the shader is valid (or the gate is disabled).
    """
    if os.environ.get("SF_STRICT", "1") == "0":
        return None
    if GLSLANG is None:
        raise RuntimeError("glslangValidator not found (apt install glslang-tools) or set SF_STRICT=0")
    r = subprocess.run([GLSLANG, "--stdin", "-S", "frag"], input=wrap_fragment(code).encode(),
                       capture_output=True, timeout=10)
    if r.returncode == 0:
        return None
    msg = r.stdout.decode(errors="replace") + r.stderr.decode(errors="replace")
    return "glslang: " + " | ".join(l for l in msg.splitlines() if "ERROR" in l)[:500]


def wrap_vertex(mode: str = "es") -> str:
    return VERTEX_ES if mode == "es" else VERTEX_ES.replace("#version 300 es", "#version 330 core")


class RenderResult(NamedTuple):
    ok: bool
    stage: str            # ok | compile | render | timeout | worker
    error: str
    image: Optional[np.ndarray]  # (H, W, 3) uint8, row 0 = top


# ----------------------------------------------------------------------------
# worker side
# ----------------------------------------------------------------------------
def _make_context():
    import moderngl
    errors = []
    # Rented GPU containers often expose an NVIDIA EGL device that cannot be initialised
    # (graphics capability not mounted), so probe every EGL device before giving up.
    attempts = [("egl", {"backend": "egl", "device_index": i}) for i in range(4)]
    attempts += [("osmesa", {"backend": "osmesa"}), ("default", {})]
    for name, kw in attempts:
        try:
            ctx = moderngl.create_standalone_context(require=330, **kw)
            return ctx, name
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "found" in msg and "devices" in msg:  # ran past the last EGL device
                continue
            errors.append(f"{name}{kw.get('device_index', '')}: {e}")
    raise RuntimeError("no GL context: " + " | ".join(errors))


def _worker_main() -> None:
    import moderngl
    # keep fd 1 private for the binary protocol; anything else printing goes to stderr
    out = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def send(obj, payload: bytes = b""):
        obj = dict(obj, n=len(payload))
        out.write((json.dumps(obj) + "\n").encode() + payload)

    try:
        ctx, backend = _make_context()
    except Exception as e:  # noqa: BLE001
        send({"ready": False, "error": str(e)})
        return

    def try_compile(mode, frag):
        vs = wrap_vertex(mode)
        return ctx.program(vertex_shader=vs, fragment_shader=wrap_fragment(frag, mode))

    probe = "void mainImage(out vec4 f, in vec2 c) { f = vec4(c / iResolution.xy, 0.0, 1.0); }"
    mode = "es"
    try:
        try_compile("es", probe).release()
    except Exception:  # noqa: BLE001
        mode = "core330"
        try:
            try_compile("core330", probe).release()
        except Exception as e:  # noqa: BLE001
            send({"ready": False, "error": f"probe shader failed: {e}"})
            return
    send({"ready": True, "backend": backend, "mode": mode,
          "renderer": ctx.info.get("GL_RENDERER", "?")})

    fbos = {}
    for line in sys.stdin.buffer:
        try:
            req = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        w, h = int(req["w"]), int(req["h"])
        try:
            prog = try_compile(mode, req["code"])
        except Exception as e:  # noqa: BLE001
            send({"ok": False, "stage": "compile", "error": str(e)[:1500]})
            continue
        try:
            if (w, h) not in fbos:
                fbos[(w, h)] = ctx.simple_framebuffer((w, h), components=4)
            fbo = fbos[(w, h)]
            fbo.use()
            fbo.clear(0.0, 0.0, 0.0, 1.0)
            if "iResolution" in prog:
                prog["iResolution"].value = (float(w), float(h), 1.0)
            if "iTime" in prog:
                prog["iTime"].value = 0.0
            vao = ctx.vertex_array(prog, [])
            vao.render(moderngl.TRIANGLES, vertices=3)
            raw = fbo.read(components=3)
            vao.release()
            prog.release()
            img = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)[::-1]
            send({"ok": True, "stage": "ok", "error": ""}, np.ascontiguousarray(img).tobytes())
        except Exception as e:  # noqa: BLE001
            send({"ok": False, "stage": "render", "error": str(e)[:1500]})


# ----------------------------------------------------------------------------
# parent side
# ----------------------------------------------------------------------------
class _Worker:
    def __init__(self, threads: int = 2, startup_timeout: float = 90.0):
        env = dict(os.environ)
        env.setdefault("LP_NUM_THREADS", str(threads))
        env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "shaderforge.render", "--worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            bufsize=0, env=env, cwd=ROOT)
        self.fd = self.proc.stdout.fileno()
        self._buf = b""
        hdr = self._read_msg(time.time() + startup_timeout)
        if hdr is None or not hdr[0].get("ready"):
            err = hdr[0].get("error") if hdr else "worker startup timeout/died"
            self.kill()
            raise RuntimeError(f"render worker failed to start: {err}")
        self.info = hdr[0]

    def _read_exact(self, n: int, deadline: float) -> Optional[bytes]:
        while len(self._buf) < n:
            left = deadline - time.time()
            if left <= 0:
                return None
            r, _, _ = select.select([self.fd], [], [], left)
            if not r:
                return None
            chunk = os.read(self.fd, 1 << 20)
            if not chunk:
                return None
            self._buf += chunk
        data, self._buf = self._buf[:n], self._buf[n:]
        return data

    def _read_msg(self, deadline: float):
        while b"\n" not in self._buf:
            left = deadline - time.time()
            if left <= 0:
                return None
            r, _, _ = select.select([self.fd], [], [], left)
            if not r:
                return None
            chunk = os.read(self.fd, 1 << 20)
            if not chunk:
                return None
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        hdr = json.loads(line)
        payload = self._read_exact(hdr.get("n", 0), deadline) if hdr.get("n") else b""
        if payload is None:
            return None
        return hdr, payload

    def request(self, code: str, w: int, h: int, timeout: float) -> RenderResult:
        if self.proc.poll() is not None:
            return RenderResult(False, "worker", "worker died", None)
        try:
            self.proc.stdin.write((json.dumps({"code": code, "w": w, "h": h}) + "\n").encode())
        except Exception as e:  # noqa: BLE001
            return RenderResult(False, "worker", f"write failed: {e}", None)
        msg = self._read_msg(time.time() + timeout)
        if msg is None:
            return RenderResult(False, "timeout", f"no result within {timeout}s", None)
        hdr, payload = msg
        if not hdr["ok"]:
            return RenderResult(False, hdr["stage"], hdr["error"], None)
        img = np.frombuffer(payload, dtype=np.uint8).reshape(h, w, 3).copy()
        return RenderResult(True, "ok", "", img)

    def kill(self):
        try:
            self.proc.kill()
            self.proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            pass


class Renderer:
    """Single persistent worker. Thread-safe (serialises requests)."""

    def __init__(self, timeout: float = 5.0, threads: int = 2):
        self.timeout, self.threads = timeout, threads
        self._lock = threading.Lock()
        self._w: Optional[_Worker] = None
        self.info: dict = {}

    def _ensure(self):
        if self._w is None or self._w.proc.poll() is not None:
            self._w = _Worker(self.threads)
            self.info = self._w.info

    def render(self, code: str, size: int = 128, timeout: Optional[float] = None) -> RenderResult:
        with self._lock:
            self._ensure()
            res = self._w.request(code, size, size, timeout or self.timeout)
            if res.stage in ("timeout", "worker"):
                self._w.kill()
                self._w = None
            return res

    def close(self):
        with self._lock:
            if self._w:
                self._w.kill()
                self._w = None


class RenderPool:
    """N workers; ``render_many`` renders a list of shaders in parallel."""

    def __init__(self, n: int = 2, timeout: float = 5.0, threads: int = 1):
        self.renderers = [Renderer(timeout, threads) for _ in range(n)]
        self._free: "queue.Queue[Renderer]" = queue.Queue()
        for r in self.renderers:
            self._free.put(r)
        self._ex = ThreadPoolExecutor(max_workers=n)

    @property
    def info(self) -> dict:
        for r in self.renderers:
            if r.info:
                return r.info
        return {}

    def render(self, code: str, size: int = 128, timeout: Optional[float] = None) -> RenderResult:
        err = strict_check(code)
        if err:
            return RenderResult(False, "compile", err, None)
        r = self._free.get()
        try:
            return r.render(code, size, timeout)
        finally:
            self._free.put(r)

    def render_many(self, codes: List[str], size: int = 128) -> List[RenderResult]:
        return list(self._ex.map(lambda c: self.render(c, size), codes))

    def close(self):
        for r in self.renderers:
            r.close()
        self._ex.shutdown(wait=False)


_default_pool: Optional[RenderPool] = None


def get_pool(n: Optional[int] = None) -> RenderPool:
    """Process-wide pool (size from SHADERFORGE_WORKERS, default 2)."""
    global _default_pool
    if _default_pool is None:
        n = n or int(os.environ.get("SHADERFORGE_WORKERS", "2"))
        _default_pool = RenderPool(n, timeout=float(os.environ.get("SHADERFORGE_TIMEOUT", "5")))
        atexit.register(_default_pool.close)
    return _default_pool


def render_shader(code: str, size: int = 128) -> RenderResult:
    return get_pool().render(code, size)


def save_png(img: np.ndarray, path: str) -> None:
    from PIL import Image
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    Image.fromarray(img).save(path)


def load_png(path: str) -> np.ndarray:
    from PIL import Image
    return np.array(Image.open(path).convert("RGB"))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("shader", nargs="?", help="GLSL file with mainImage")
    ap.add_argument("out", nargs="?", help="output PNG")
    ap.add_argument("--size", type=int, default=128)
    a = ap.parse_args()
    if a.worker:
        _worker_main()
    elif a.shader and a.out:
        res = render_shader(open(a.shader).read(), a.size)
        if not res.ok:
            sys.exit(f"{res.stage}: {res.error}")
        save_png(res.image, a.out)
        print("wrote", a.out, get_pool().info)
    else:
        ap.print_help()
