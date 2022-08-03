import requests
import asyncio

throttled = False
url = "https://instructor-warn.herokuapp.com/warn"

async def warningPing():
    global throttled
    global url

    if not throttled:
        try:
            requests.get(url)
            throttled = True
            await asyncio.sleep(10)
        except:
            print("Unable to ping")
        finally:
            throttled = False