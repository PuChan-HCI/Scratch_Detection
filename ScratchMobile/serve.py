"""Dependency-free static server. HTTPS: supply a trusted certificate and key."""
import argparse, functools, http.server, ssl
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--port',type=int,default=8000); p.add_argument('--host',default='0.0.0.0'); p.add_argument('--cert'); p.add_argument('--key'); a=p.parse_args()
if bool(a.cert)!=bool(a.key): p.error('--cert and --key are both required')
class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map={**http.server.SimpleHTTPRequestHandler.extensions_map,'.wasm':'application/wasm','.mjs':'text/javascript','.onnx':'application/octet-stream'}
    def end_headers(self):
        self.send_header('X-Content-Type-Options','nosniff'); super().end_headers()
server=http.server.ThreadingHTTPServer((a.host,a.port),functools.partial(Handler,directory=str(Path(__file__).resolve().parent)))
if a.cert:
    context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER); context.load_cert_chain(a.cert,a.key); server.socket=context.wrap_socket(server.socket,server_side=True)
print(f"Scratch Lens: {'https' if a.cert else 'http'}://localhost:{a.port}",flush=True)
try: server.serve_forever()
except KeyboardInterrupt: server.server_close()
