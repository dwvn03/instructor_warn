import requests
import asyncio
import multiprocessing.dummy as mpd

throttled = False
URL = "https://instructor-warn.herokuapp.com/warn"

pool = mpd.Pool(10)

async def warningPing():
    global throttled
    global URL

    if not throttled:
        try:
            pool.apply_async(requests.get, [ URL ])
            # requests.get(url)
            throttled = True
            await asyncio.sleep(10)
        except:
            print("Unable to ping")
        finally:
            throttled = False