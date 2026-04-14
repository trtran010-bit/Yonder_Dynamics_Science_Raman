rom time import sleep
from serial import Serial
import numpy as np
from serial.tools import list_ports
import matplotlib.pyplot as plt

# def find_port():
#     ports = list_ports.comports()
#     for p in ports:
#         name = p.device
#         # Mac: /dev/tty.usbserial-* or /dev/tty.usbmodem-*
#         # Linux: /dev/ttyUSB* or /dev/ttyACM*
#         if any(pat in name for pat in ("usbserial", "usbmodem", "ttyUSB", "ttyACM")):
#             print(f"Using port: {name}")
#             return name
#     raise RuntimeError("No serial port found")

# PORT = find_port()

from serial.tools import list_ports
port = list(list_ports.comports())
for p in port:
    print(p.device)

# write bytes with delay
def _writeline(ser, data, delay=0.05):
    ser.write(data.encode("utf-8") + b"\r\n")
    sleep(delay)

def _decode_spectrometer_data(data):
    # here is how the data is compressed over serial:
    # - before sending a pixel value, the spectrometer always sends 0x80.
    # - the following two bytes are guaranteed to encode a pixel value, with the
    #  first of the two bytes being MSB
    # - if the next byte is not a 0x80, that byte is an offset from the previous
    #  byte, and this pixel value is found by adding this offset to the previous
    # - pixel value. For example, in saturated regions, this becomes
    #           80 ff ff 00 00 00 00 00 80 ...
    #           (7 saturated pixels of 0xffff)
    # - offsets are signed int8's and can be chained. In very noisy data where
    #  all the data lies within +/-127 of the first value in the spectrum, only
    #  one 0x80 will appear and this simple compression has a 2:1 ratio!
    # Use a state machine to decode this compression scheme:
    state = 0
    # 0 = haven't seen our first 0x80 yet, data hasn't started
    # 1 = just saw a 0x80, ready to read first and second bytes
    # 2 = just read in 1st and 2nd bytes after 0x80, any subsequent byte will
    # be added to the previous pixel number
    output = [0] * 2048
    pixelNum = 0
    byteIndex = 0
    while byteIndex < len(data):
        thisByte = data[byteIndex]
        if state != 1 and thisByte == 128:
            # 0x80 typically signals to change state, but it's possible for it
            # to be the MSB or LSB of the initial two bytes, so only treat it
            # as something special if the state isn't 1
            state = 1
            byteIndex += 1
        elif state == 0:
            byteIndex += 1  # simply advance byte and do nothing
        elif state == 1:
            # these are the first two bytes after a 0x80, put into pixel
            # val
            output[pixelNum] = 256 * data[byteIndex] + data[byteIndex + 1]
            byteIndex += 2
            pixelNum += 1
            state = 2
        elif state == 2:
            # simple compression tactic, we got more than 2 non-0x80
            # bytes in a row, so this is an offset from prev. pixel
            # value
            # even more fun, this is a SIGNED offset, gross uint8 to int8 hack
            if thisByte < 128:
                signedOffset = thisByte
            else:
                signedOffset = -256 + thisByte
            output[pixelNum] = output[pixelNum - 1] + signedOffset
            byteIndex += 1
            pixelNum += 1
            # state does not change
        if pixelNum == 2048:
            extraBytes = len(data) - byteIndex
            # print('Exited decoding with ' + str(extraBytes) + ' extra bytes - checksum not yet implemented!')
            break  # the sensor is only 2048 pixels
    return np.array(output)

def read_spectrometer(integration_time):
    with Serial(port, 9600, timeout=1) as ser:
        _writeline(ser, "Q")  # reset settings
        _writeline(ser, "K0")  # raise baudrate
        ser.baudrate = 115200
        _writeline(ser, "b")  # ???
        _writeline(ser, f"I{integration_time}")  # integration time
        _writeline(ser, "A1")  # averages
        ser.reset_input_buffer()  # clear buffer
        _writeline(ser, "S")  # request data
        data = ser.read(5000)  # read data, should never be more than 4096
        _writeline(ser, "Q")  # reset settings
        return _decode_spectrometer_data(data)  # get numbers

def plot_spectrum(integration_time):
    data = read_spectrometer(integration_time)
    pixels = range(len(data))  # x-axis: pixel index (0–2047)
    
    plt.figure(figsize=(10, 4))
    plt.plot(pixels, data)
    plt.xlabel("Pixel")
    plt.ylabel("Intensity (counts)")
    plt.title(f"Spectrometer Readout (integration time: {integration_time})")
    plt.tight_layout()
    plt.show()

plot_spectrum(1000)  # 1000ms integration time, adjust as needed