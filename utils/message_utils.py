import pynng
import curio
import json

address = "ipc:///tmp/pipeline.ipc"
class Sender(object):
    def __init__(self) -> None:
        pass

    @staticmethod
    def ping():
        try:
            with pynng.Push0() as sock:
                sock.dial(address, block=True)
            return True
        except Exception as e:
            return False

    @staticmethod
    async def send(message: str):
        with pynng.Push0() as sock:
            sock.dial(address, block=True)
            print(f'NODE1: SENDING "{message}"')
            msg = {"msg": message}
            await sock.asend(json.dumps(msg).encode())
            # await sock.asend(message.encode())
            # await curio.sleep(1)  # wait for messages to flush before shutting down

# # receiver
# async def node0(sock):
#     async def recv_eternally():
#         while True:
#             msg = await sock.arecv_msg()
#             content = msg.bytes.decode()
#             print(f'NODE0: RECEIVED "{content}"')

#     sock.listen(address)
#     return await curio.spawn(recv_eternally)

class Receiver(object):
    def __init__(self) -> None:
        pass

    @staticmethod
    async def receive(func=None):
        with pynng.Pull0() as sock:
            sock.listen(address) # must listen before dialing
            while True:
                msg = await sock.arecv_msg()
                content = msg.bytes.decode()
                ct_json = json.loads(content)
                message = ct_json["msg"]
                print(f'NODE0: RECEIVED "{message}", {type(message)}')
                if func:
                    func(message)

    # @staticmethod
    # async def receive1():
    #     with pynng.Pull0() as sock:
    #         n0 = await node0(sock)
    #         await curio.sleep(1)