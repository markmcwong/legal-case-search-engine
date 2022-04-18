from __future__ import division
from struct import pack, unpack

def VBEncodeNumber(number):
    """Variable byte code encode number.
    Usage:
      import vbcode
      vbcode.encode_number(128)
    """
    bytes_list = []
    while True:
        bytes_list.insert(0, number % 128)
        if number < 128:
            break
        number = number // 128
    bytes_list[-1] += 128
    return pack('%dB' % len(bytes_list), *bytes_list)

def VBEncode(numbers):
    """Variable byte code encode numbers.
    Usage:
      import vbcode
      vbcode.encode([32, 64, 128])
    """
    bytes_list = []
    for number in numbers:
        bytes_list.append(VBEncodeNumber(number))
    return b"".join(bytes_list)

def VBDecode(bytestream):
    """Variable byte code decode.
    Usage:
      import vbcode
      vbcode.decode(bytestream)
        -> [32, 64, 128] (a list of numbers)
    """
    n = 0
    numbers = []
    bytestream = unpack('%dB' % len(bytestream), bytestream)
    for byte in bytestream:
        if byte < 128:
            n = 128 * n + byte
        else:
            n = 128 * n + (byte - 128)
            numbers.append(n)
            n = 0
    return numbers

def decode_stream(file_handle, count):
    """Reads in a variable number of numbers from a file. May not
    return the asked for number of numbers if EOF is reached"""
    n = 0
    numbers = []

    while len(numbers) < count:
        read_byte = file_handle.read(1)
        if not read_byte:  # EOF reached
            break

        byte = ord(read_byte)

        if byte < 128:
            n = 128 * n + byte
        else:
            n = 128 * n + (byte - 128)
            numbers.append(n)
            n = 0

    return numbers
