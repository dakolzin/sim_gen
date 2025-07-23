import socket, struct, msgpack, numpy as np

def _enc(o):
    if isinstance(o, np.ndarray):
        return {b'nd': True, b'dt': str(o.dtype), b's': o.shape, b'b': o.tobytes()}
    raise TypeError

def _dec(o):
    if (b'nd' in o) or ('nd' in o):
        get = lambda k: o.get(k) or o.get(k.decode())
        return np.frombuffer(get(b'b'), dtype=get(b'dt')).reshape(get(b's'))
    return o

pack   = lambda obj: msgpack.packb(obj, default=_enc, use_bin_type=True)
unpack = lambda buf: msgpack.unpackb(buf, object_hook=_dec, raw=False)

def _recvall(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed")
        buf.extend(chunk)
    return bytes(buf)

def send_msg(sock: socket.socket, obj):
    buf = pack(obj)
    sock.sendall(struct.pack('<I', len(buf)) + buf)

def recv_msg(sock: socket.socket):
    sz  = struct.unpack('<I', _recvall(sock, 4))[0]
    return unpack(_recvall(sock, sz))
