# dinowork.py
from pyautogui import press, typewrite
from time import sleep

# Starts Chrome and navigates to the Dino game
# Usage: launch_game("chrome://dino")
def launch_game(game_link: str):
    press("win")
    sleep(0.75)
    typewrite("Google Chrome")
    sleep(0.75)
    press("enter")
    sleep(0.75)
    typewrite(game_link)
    press("enter")
    sleep(1)
    press("up")  # Start game
    sleep(0.75)
