#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль обмена сообщений по сокету с поддержкой numpy через msgpack:
- Сериализация numpy массивов (_enc/_dec)
- Упаковка/распаковка объектов (pack/unpack)
- Приём всех байт (_recvall)
- Отправка/приём сообщений с заголовком длины (send_msg/recv_msg)
"""

import socket           # сетевые соединения
import struct           # упаковка фиксированных форматов (заголовок длины)
import msgpack          # сериализация данных в компактный бинарный формат
import numpy as np      # работа с массивами


def _enc(o):
    """
    Конвертация numpy.ndarray в сериализуемую форму.
    Возвращает dict с флагом 'nd', типом, формой и байтовыми данными.
    """
    if isinstance(o, np.ndarray):
        return {
            b'nd': True,           # признак numpy-данных
            b'dt': str(o.dtype),   # строка типа данных
            b's': o.shape,         # форма массива
            b'b': o.tobytes()      # сырой буфер байт
        }
    raise TypeError(f"Type {type(o)} not serializable")


def _dec(o):
    """
    Преобразование dict с numpy-данными обратно в ndarray.
    Проверяет наличие ключа 'nd' и восстанавливает массив.
    """
    # учитывать как bytes, так и str ключи
    if (b'nd' in o) or ('nd' in o):
        # функция для чтения ключей в любом формате
        get = lambda k: o.get(k) or o.get(k.decode())
        data = get(b'b')
        dtype = get(b'dt')
        shape = get(b's')
        return np.frombuffer(data, dtype=dtype).reshape(shape)
    return o

# Функции упаковки/распаковки объектов с использованием msgpack
pack   = lambda obj: msgpack.packb(obj, default=_enc, use_bin_type=True)
unpack = lambda buf: msgpack.unpackb(buf, object_hook=_dec, raw=False)


def _recvall(sock: socket.socket, n: int) -> bytes:
    """
    Читает ровно n байт из сокета, блокируя вызов до получения.
    Если соединение закрыто раньше, бросает ConnectionError.
    """
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed before receiving all data")
        buf.extend(chunk)
    return bytes(buf)


def send_msg(sock: socket.socket, obj) -> None:
    """
    Сериализует объект, отправляет по сокету с 4-байтным заголовком длины (little-endian).
    """
    buf = pack(obj)  # сериализация msgpack
    length = struct.pack('<I', len(buf))  # 4 байта: длина буфера
    sock.sendall(length + buf)


def recv_msg(sock: socket.socket):
    """
    Принимает сообщение: сначала читает 4-байтный заголовок длины,
    затем читает указанное число байт и десериализует обратно в объект.
    """
    # читаем размер сообщения
    header = _recvall(sock, 4)
    sz = struct.unpack('<I', header)[0]
    # читаем само тело и распаковываем
    data = _recvall(sock, sz)
    return unpack(data)
