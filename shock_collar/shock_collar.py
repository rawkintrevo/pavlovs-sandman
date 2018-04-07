from gattlib import GATTRequester
from time import sleep

class Controller:
    def __init__(self, bd_addr = "00:07:80:C2:20:83"):
        self.bd_addr = bd_addr
        self.id_s2 =     "55373734353534"
        self.buzz_s =    '553631333330'
        self.tone_s =    "553631313130"
        self.shock_s = {
            1 : "553631323331",
            2 : "553631323332",
            3 : "553631323333",
            4 : "553631323334",
            5 : "553631323335",
            6 : "553631323336",
            7 : "553631323337",
            8 : "553631323338",
            9 : "553631323339"
        }
        self.connect()
        for i in range(0,3):
            self.tone()
            sleep(0.5)

    def connect(self):
        self.req = GATTRequester(self.bd_addr)
        while not self.req.is_connected():
            pass
        self.req.write_by_handle(0x0013, str(self.id_s2.decode("hex")))

    def tone(self):
        self.req.write_by_handle(0x0013, str(self.tone_s.decode("hex")))

    def buzz(self):
        self.req.write_by_handle(0x0013, str(self.buzz_s.decode("hex")))

    def shock(self, level = 1):
        self.req.write_by_handle(0x0013, str(self.shock_s[level].decode("hex")))