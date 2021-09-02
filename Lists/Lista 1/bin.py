import struct

def binary(num): 
	s = ''.join('{:0>16b}'.format(c) for c in struct.pack('!f', num))
	return s[0] + ' ' + s[1:12] + ' ' + s[12:]
	
def dec(binary_number):
	s, c, f = binary_number.split()
	sig = int(s, 2)
	exp = int(c, 2) - 1023
	man = 1
	for i in range(len(f)):
		man += int(f[i], 2) * 2**(- i - 1)
	
	num = (-1)**sig * 2**exp * man	
	return num

def float_to_bin(num):
	s = format(struct.unpack('!I', struct.pack('!f', num))[0], '064b')
	# return s[0] + ' ' + s[1:12] + ' ' + s[12:]
	return s
    
def bin_to_float(binary):
	# binary.replace(' ', '')
	return struct.unpack('!f',struct.pack('!I', int(binary, 2)))[0]
