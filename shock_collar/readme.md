in dev options turn on bluetooth listening

adb pull /sdcard/btsnoop_hci.log


## read this for onetime pairing

https://github.com/scivision/pybluez-examples

```
#sudo hciconfig hci0 up   #enables bt on computer
#hcitool scan  # gets UUID of devices in pairing mode
#hcitool dev # get BT adapter uuid

bluetoothctl -a  #starts interactive prompt
scan on          #scans for UUID of device (BT and BLE) in pairing mode
pair addr        # where "uuid" is what you found with scan
trust addr
connect addr    # after pairing, this is how you connect in the future
```

My dev addr: 00:07:80:C2:20:83
My laptop : C4:8E:8F:C9:DC:6C