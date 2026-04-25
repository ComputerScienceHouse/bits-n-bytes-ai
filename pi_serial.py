import serial

PORT = "/dev/ttyTHS1"
UART = serial.Serial(
    port=PORT,
    baudrate=9600,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    timeout=1
)