from pynput.keyboard import Key, Controller
import time
key = Controller()


def pause():
    time.sleep(1.5)


def press(keypressed):
    key.press(str(keypressed))


time.sleep(1.5)




