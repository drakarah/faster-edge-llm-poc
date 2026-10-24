"""Write manifest.txt (sha256, size, relative path per line) for a directory of an exported model, so a plain web server, object store or release page can act as a mirror
(ONNX_MIRROR_URL). Usage: make_mirror_manifest.py DIR   then upload DIR (with manifest.txt) to any HTTPS host that supports range requests."""
import hashlib, os, sys
root = sys.argv[1]
with open(os.path.join(root, "manifest.txt"), "w") as out:
    for base, _, files in sorted(os.walk(root)):
        for f in sorted(files):
            if f in ("manifest.txt",) or f.endswith((".ok", ".complete")) or f.startswith("."):
                continue
            path = os.path.join(base, f); h = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 24), b""):
                    h.update(chunk)
            out.write(f"{h.hexdigest()}  {os.path.getsize(path)}  {os.path.relpath(path, root)}\n")
print("wrote", os.path.join(root, "manifest.txt"))
