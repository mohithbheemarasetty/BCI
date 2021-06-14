#only needed if running remotely on a raspberry pi
import socket
import keypress


def right():
    keypress.press('d')


def left():
    keypress.press('a')


def forward():
    keypress.press('w')


PORT = 2345

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((socket.gethostname(), PORT))
s.listen(1)
conn, addr = s.accept()
with conn:
    print('Connected by', addr)
    while True:
        data = conn.recv(1024)
        if not data:
            break
        result = data.decode('utf-8')
        print(result)
        if result == 'right':
            keypress.pause()
            right()
        if result == 'left':
            keypress.pause()
            left()
        if result == 'forward':
            keypress.pause()
            forward()
        result = ''

