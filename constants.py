
frame_headers = {
    "229": [-1, 1, 1, 1, 1, 1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, 1, 1, 1, -1,
            -1, 1, -1, 1, -1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1, 1,
            -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1,
            1, 1, -1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1,
            -1, -1, 1, -1, -1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1, -1,
            1, 1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 0, 1, -1,
            -1, -1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1,
            1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, -1,
            -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, 1,
            -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1,
            1, 1, 1, -1, 1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1,
            -1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, 1],

    "115": [1, -1, 1, 1, -1, -1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, -1, 1,
            1, -1, -1, -1, -1, 1, -1, -1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1,
            -1, 1, -1, -1, 1, 1, -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1,
            -1, 1, 1, 0, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1,
            1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1,
            1, -1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1,
            -1, 1, 1, 1, 1, 1, -1],

    "69": [1, -1, -1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1,
           -1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 0, 1, -1, -1,
           -1, 1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1,
           1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1],

    "23": [1, -1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 0, 1, -1, -1, -1, 1, -1, -1,
           -1, -1, -1, 1]
}

subcarriers = {
    "229": [x for x in range(-114, 115)],
    "115": [x for x in range(-57, 58)],
    "69": [x for x in range(-34, 35)],
    "23": [x for x in range(-11, 12)]
}

pilots = {}

pilot_sequences = {
    "229": [],
    "115": [],
    "69": [],
    "23": []
}


# def check_int_length(field, length):
#     if field not in range(0, 2**length):
#         raise ValueError(str(field))


# class mis():

#     def encode(spect, tis_mod, ds_mod):
#         check_int_length(spect, 2)
#         check_int_length(tis_mod, 1)
#         check_int_length(ds_mod, 2)

#         # TODO: calculate crc, GF(128) Reed–Solomon (4,2)
#         crc = 0xAA
#         return "{:02b}{:01b}{:02b}0{:08b}".format(spect, tis_mod, ds_mod, crc)

#     def decode(field):
#         if len(field) != 14:
#             raise ValueError(
#                 "Expect field length 14 bit, but got {} bits".format(len(field)))

#         if not set(field).issubset({'0', '1'}):
#             raise ValueError

#         if field[5] != '0':
#             raise ValueError("Invalid value of additional bit")

#         spect = field[0:2]
#         tis_mod = field[2]
#         ds_mod = field[3:5]
#         crc = field[-8:]

#         # TODO: check crc
#         if False:
#             raise ValueError

#         return spect, tis_mod, ds_mod


# class tis():
#     def encode(ds_code, tx_id, hours, minutes, tx_len):
#         check_int_length(ds_code, 5)
#         check_int_length(tx_id, 10)
#         check_int_length(hours, 5)
#         check_int_length(minutes, 6)
#         check_int_length(tx_len, 6)

#         # TODO: calculate crc, GF(128) Reed–Solomon (29,9)
#         crc = 0xAA

#         return "{:05b}{:010b}{:05b}{:06b}{:06b}{:023b}{:08b}".format(ds_code,
#                                                                      tx_id,
#                                                                      hours,
#                                                                      minutes,
#                                                                      tx_len,
#                                                                      0, crc)


# if __name__ == '__main__':
#     print(mis.decode(mis.encode(3, 0, 1)))
#     print(tis.encode(0x1f, 0x3ff, 0x1f, 0x3f, 0x3f))
