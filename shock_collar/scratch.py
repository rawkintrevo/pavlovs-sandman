
from bluetooth import *
import bluetooth


print("performing inquiry...")

nearby_devices = discover_devices(lookup_names = True)
print("found %d devices" % len(nearby_devices))
for addr, name in nearby_devices:
    print(" %s - %s" % (addr, name))

nearby_devices = discover_devices(duration=4,lookup_names=True,
                                            flush_cache=True, lookup_class=False)

# Device PIN 4554 - set in app- doesn't seem to be needed- see README.md for instructions on pairing

name = "PetSafe Smart Dog Trainer"
addr = "00:07:80:c2:20:83"      # Device Address

sock=bluetooth.BluetoothSocket(bluetooth.L2CAP)

bd_addr = "01:23:45:67:89:AB"
port = 0x1001

sock.connect((bd_addr, port))

sock.send("Hello World")

print "Finished"

client_socket.close()

#sudo rfcomm bind 0 00:07:80:C2:20:83
#sudo rfcomm release 0
bd_addr = "00:07:80:C2:20:83"
port = 0x0004
sock.connect((bd_addr, port))
# When Laptop BT is connected, error is (112, 'Host is down')
# When Laptop / Phone not connected, error is (77, 'File descriptor in bad state')

sock=bluetooth.BluetoothSocket( bluetooth.L2CAP )
for port in range(0,30):
    print(port)
    try:
        sock.connect((bd_addr, port))
    except:
        pass

"02 40 00 0d 00 09 00 04 00 12 13 00 55 36 31 33 "
"33 30"
hex1 = "\x02\x40\x00\x0d\x00\x09\x00\x04\x00\x12\x13\x00\x55\x36\x31\x33"
hex2 = "0240000d000900040012130055363133"
hex2a = "3330"


dev = os.open("/dev/rfcomm0", os.O_RDWR)
os.write(dev, bytearray.fromhex(hex2))
os.write(dev, bytearray.fromhex(hex2a))
os.lseek(dev,0,os.SEEK_SET)
print(os.read(dev, 6))

####################################################################################################
# 3-15-2018
import bluetooth

target_name = "My Phone"
target_address = None

nearby_devices = bluetooth.discover_devices()

####################################################################################################
# 3-18-2018


import bluetooth

sock=bluetooth.BluetoothSocket(bluetooth.L2CAP)

bd_addr = "00:07:80:C2:20:83"


for port in range(4097, 36864, 2):
    print(port, " " , hex(port))
    try:
        sock.close()
        sock.connect((bd_addr, port))
        sock.send(hex(19))
        break
    except bluetooth.btcommon.BluetoothError as e:
        print("nope")


from gattlib import DiscoveryService

service = DiscoveryService("hci0")
devices = service.discover(2)

for address, name in devices.items():
    print("name: {}, address: {}".format(name, address))

from gattlib import GATTRequester, GATTResponse
bd_addr = "00:07:80:C2:20:83"
req = GATTRequester(bd_addr)
resp = GATTResponse()
req.disconnect()
characteristics = req.discover_characteristics()

### Phone to Dev (for Buzz)
# 0000   02 40 00 0d 00 09 00 04 00 12 13 00 55 36 31 33  .@..........U613
# 0010   33 30                                            30

### Ubuntu to Dev (for writing str(bytearray([55,36,31,33,33,30]))
# 0000   02 03 00 0d 00 09 00 04 00 12 13 00 37 24 1f 21  ............7$.!
# 0010   21 1e                                            !.

# Ubuntu to Dev (for writing req.write_by_handle(0x0013, str(s.decode("hex")))
# 0000   02 04 00 0d 00 09 00 04 00 12 13 00 55 36 31 33  ............U613
# 0010   33 30                                            30
s = '553631333330'
# s.decode("hex")
req.write_by_handle(0x0013, str(s.decode("hex")))

### We're currently on connection handle 0x04 but we need to be on 0x040

from gattlib import GATTRequester, GATTResponse
bd_addr = "00:07:80:C2:20:83"
req = GATTRequester(bd_addr)


id_s2 = "55373734353534"
buzz_s = '553631333330'
req.write_by_handle(0x0013, str(id_s2.decode("hex")))
req.write_by_handle(0x0013, str(buzz_s.decode("hex")))

### That did it- we needed the "this is me" string it seems
### Tone:
# 0000   02 40 00 0d 00 09 00 04 00 12 13 00 55 36 31 31  .@..........U611
# 0010   31 30                                            10

from gattlib import GATTRequester, GATTResponse
bd_addr = "00:07:80:C2:20:83"
req = GATTRequester(bd_addr)
while not req.is_connected():
    pass

id_s2 = "55373734353534"
buzz_s = '553631333330'
tone_s = "553631313130"
req.write_by_handle(0x0013, str(id_s2.decode("hex")))
req.write_by_handle(0x0013, str(buzz_s.decode("hex")))
req.write_by_handle(0x0013, str(tone_s.decode("hex")))

### Perfect!  Now to get the shock-fns
# Let's start with a level 1 shock (we'll progress to level 2, then we may be able to infer how to
# work our way up.
# 0000   02 40 00 0d 00 09 00 04 00 12 13 00 55 36 31 32  .@..........U612
# 0010   33 31                                            31


from gattlib import GATTRequester, GATTResponse
bd_addr = "00:07:80:C2:20:83"
req = GATTRequester(bd_addr)
while not req.is_connected():
    pass

id_s2 =     "55373734353534"
buzz_s =    '553631333330'
tone_s =    "553631313130"
shock_s1 =  "553631323331"
req.write_by_handle(0x0013, str(id_s2.decode("hex")))
req.write_by_handle(0x0013, str(buzz_s.decode("hex")))
req.write_by_handle(0x0013, str(tone_s.decode("hex")))
req.write_by_handle(0x0013, str(shock_s1.decode("hex")))

## Shock Lvl 2-4
# Lvl 2
# 0000   02 40 00 0d 00 09 00 04 00 12 13 00 55 36 31 32  .@..........U612
# 0010   33 32                                            32
#
# Lvl 3
# 0000   02 40 00 0d 00 09 00 04 00 12 13 00 55 36 31 32  .@..........U612
# 0010   33 33                                            33
#
# Lvl 4
# 0000   02 40 00 0d 00 09 00 04 00 12 13 00 55 36 31 32  .@..........U612
# 0010   33 34                                            34

shock_s2 = "553631323332"
shock_s3 = "553631323333"
shock_s4 = "553631323334"

# By Inference
shock_s5 = "553631323335"
shock_s6 = "553631323336"
shock_s7 = "553631323337"
shock_s8 = "553631323338"
shock_s9 = "553631323339"
