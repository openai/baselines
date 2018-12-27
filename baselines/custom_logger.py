import datetime as dt

"""
For Corlor, check this site.
+ https://qiita.com/ironguy/items/8fb3ddadb3c4c986496d
"""

class CustomLoggerObject(object):
    def __init__(self):
        self.LOG_FMT = "{color}| {asctime} | {levelname:<5s} | {message} \033[0m"

    def info(self, msg):
        asctime = dt.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        print(self.LOG_FMT.format(color="\033[37m", asctime=asctime, levelname="INFO", message=msg))

