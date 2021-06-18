from pynput.keyboard import Key, Controller
import time
keyb = Controller()


def pres(keypressed):
    keyb.press(str(keypressed))
    time.sleep(1)
    keyb.release(str(keypressed))






