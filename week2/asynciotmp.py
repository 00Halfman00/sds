# main.py
import asyncio


async def work(delay):
    print("Hello")
    await asyncio.sleep(delay)
    print("Async done!")
    return delay


async def one_for_all():
    results = await asyncio.gather(work(1), work(1), work(1))
    print(results)


async def countdown(num):
    print("Commencing takeoff in: ")
    for i in range(num, 0, -1):
        await asyncio.sleep(1)
        print(i)
    print("Takeoff!!!!!!!!!!!!")


if __name__ == "__main__":
    asyncio.run(one_for_all())
    asyncio.run(countdown(10))
