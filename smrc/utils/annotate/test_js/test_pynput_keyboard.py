#  simulate keyboard input
# from pynput.keyboard import Key, Controller
#
# keyboard = Controller()
#
# # Press and release space
# keyboard.press(Key.space)
# keyboard.release(Key.space)
#
# # Type a lower case A; this will work even if no key on the
# # physical keyboard is labelled 'A'
# keyboard.press('a')
# keyboard.release('a')
#
# # Type two upper case As
# keyboard.press('A')
# keyboard.release('A')
# with keyboard.pressed(Key.shift):
#     keyboard.press('a')
#     keyboard.release('a')
#
# # Type 'Hello World' using the shortcut type method
# keyboard.type('Hello World')


# worked version
# from pynput.keyboard import Key, Listener
#
# def on_press(key):
#     print('{0} pressed'.format(
#         key))
#
# def on_release(key):
#     print('{0} release'.format(
#         key))
#     if key == Key.esc:
#         # Stop listener
#         return False
#
# # Collect events until released
# with Listener(
#         on_press=on_press,
#         on_release=on_release) as listener:
#     listener.join()


# listen key combination, Shift a (A) -> do somethinga
#pip install pynput
from pynput import keyboard

# The key combination to check
COMBINATIONS = [
    {keyboard.Key.shift, keyboard.KeyCode(char='a')},
    {keyboard.Key.shift, keyboard.KeyCode(char='A')}
]

# The currently active modifiers
current = set()

def execute():
    print ("Do Something")

def on_press(key):
    if any([key in COMBO for COMBO in COMBINATIONS]):
        current.add(key)
        if any(all(k in current for k in COMBO) for COMBO in COMBINATIONS):
            execute()

def on_release(key):
    if any([key in COMBO for COMBO in COMBINATIONS]):
        current.remove(key)

with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    listener.join()