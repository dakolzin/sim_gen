from pathlib import Path
from roboticstoolbox import ERobot

# абсолютный путь
urdf = Path("panda_from_mjcf.urdf").resolve()
bot = ERobot.URDF(urdf)

print(bot)                 # инфо о роботе
print("tool =", bot.tool)  # должен совпадать с tcp_site
