import math
import cv2
import numpy as np
from collections import Counter
import heapq
import pickle


img = cv2.imread("RGB_24bits.jpg")
#exception handling if the image is not found()
if img is None:
    raise FileNotFoundError("Could not read RGB_24bits.jpg")




def bgr_to_ycrcb(img):
    height, width = img.shape[:2]

    Y = np.zeros((height, width), dtype=np.uint8)
    Cr = np.zeros((height, width), dtype=np.uint8)
    Cb = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            # OpenCV loads pixels in BGR order
            B = int(img[i, j, 0])
            G = int(img[i, j, 1])
            R = int(img[i, j, 2])

            y = 0.299 * R + 0.587 * G + 0.114 * B
            cr = 128 + 0.5 * R - 0.418688 * G - 0.081312 * B
            cb = 128 - 0.168736 * R - 0.331264 * G + 0.5 * B

            Y[i, j] = max(0, min(255, round(y)))
            Cr[i, j] = max(0, min(255, round(cr)))
            Cb[i, j] = max(0, min(255, round(cb)))

    return Y, Cr, Cb



Y, Cr, Cb =bgr_to_ycrcb(img)

print("Y channel shape before padding:", Y.shape)
print("Cr channel shape before padding:", Cr.shape)
print("Cb channel shape before padding:", Cb.shape)



# Chroma subsampling: 4:2:0
def chroma_subsample(channel):
    height, width = channel.shape

    new_height = (height + 1) // 2
    new_width = (width + 1) // 2

    subsampled = np.zeros(
        (new_height, new_width),
        dtype=channel.dtype
    )

    for i in range(new_height):
        for j in range(new_width):
            # Average each 2x2 region
            total = 0
            count = 0

            for di in range(2):
                for dj in range(2):
                    source_i = 2 * i + di
                    source_j = 2 * j + dj

                    if source_i < height and source_j < width:
                        total += int(channel[source_i, source_j])
                        count += 1

            subsampled[i, j] = round(total / count)

    return subsampled


# Keep full resolution for brightness
# Reduce chroma resolution by half horizontally and vertically
Cr = chroma_subsample(Cr)
Cb = chroma_subsample(Cb)

print("Y after subsampling:", Y.shape)
print("Cr after subsampling:", Cr.shape)
print("Cb after subsampling:", Cb.shape)




#pad the channels to make their dimensions multiples of 8

def pad_channel(channel):
    height, width = channel.shape

    new_height = ((height + 7) // 8) * 8
    new_width = ((width + 7) // 8) * 8

    padded = np.zeros((new_height, new_width), dtype=channel.dtype)

    for i in range(new_height):
        for j in range(new_width):
            source_i = min(i, height - 1)
            source_j = min(j, width - 1)
            padded[i, j] = channel[source_i, source_j]

    return padded
Y = pad_channel(Y)
Cr = pad_channel(Cr)
Cb = pad_channel(Cb)

print("Padded Y shape:", Y.shape)
print("Padded Cr shape:", Cr.shape)
print("Padded Cb shape:", Cb.shape)

def apply_dct(channel):
    height, width = channel.shape
    dct_image = np.zeros((height, width), dtype=np.float32)


    for i in range(0, height, 8):
        for j in range(0, width, 8):

            # Level shift and store the block using Python lists
            block = []
            for x in range(8):
                row = []
                for y in range(8):
                    row.append(int(channel[i + x, j + y]) - 128)
                block.append(row)

            # Calculate each DCT coefficient F(u, v)
            for u in range(8):
                for v in range(8):
                    total = 0.0

                    for x in range(8):
                        for y in range(8):
                            total += (
                                block[x][y]
                                * math.cos((2 * x + 1) * u * math.pi / 16)
                                * math.cos((2 * y + 1) * v * math.pi / 16)
                            )

                    # Normalization factors
                    Cu = 1 / math.sqrt(2) if u == 0 else 1
                    Cv = 1 / math.sqrt(2) if v == 0 else 1

                    dct_image[i + u, j + v] = 0.25 * Cu * Cv * total

    return dct_image


# Apply native DCT to all three channels
dct_Y = apply_dct(Y)
dct_Cr = apply_dct(Cr)
dct_Cb = apply_dct(Cb)


print("DCT applied to Y channel", dct_Y)
print("DCT applied to Cr channel:", dct_Cr)
print("DCT applied to Cb channel", dct_Cb)



# Standard JPEG luminance quantization table
Q_Y = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32)
Q_C = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
], dtype=np.float32)

# Quantize every 8x8 block
def apply_quantization(dct_image, Q):
    height, width = dct_image.shape

    quantized_image = np.zeros(
        (height, width), dtype=np.int32
    )

    for i in range(0, height, 8):
        for j in range(0, width, 8):

            # Extract the 8x8 DCT block
            block = dct_image[i:i+8, j:j+8]

            # Divide each coefficient by its matching
            # quantization value, then round
            quantized_block = np.round(block / Q).astype(np.int32)

            # Store the quantized coefficients
            quantized_image[i:i+8, j:j+8] = quantized_block

    return quantized_image


# Quantize each channel
quantized_Y = apply_quantization(dct_Y, Q_Y)
quantized_Cr = apply_quantization(dct_Cr, Q_C)
quantized_Cb = apply_quantization(dct_Cb, Q_C)


# Display the first 8x8 quantized block
print("Quantized Y block:")
print(quantized_Y[:8, :8])

print("\nQuantized Cr block:")
print(quantized_Cr[:8, :8])

print("\nQuantized Cb block:")
print(quantized_Cb[:8, :8])



# Generate the standard 8x8 zigzag scanning order
def get_zigzag_order():
    order = []

    for s in range(15):  # Diagonal sums: 0 to 14
        diagonal = []

        for row in range(8):
            col = s - row

            if 0 <= col < 8:
                diagonal.append((row, col))

        # Alternate the direction of each diagonal
        if s % 2 == 0:
            diagonal.reverse()

        order.extend(diagonal)

    return order


# Build the zigzag order once and reuse it for every block
ZIGZAG_ORDER = get_zigzag_order()


# Convert one 8x8 quantized block into a 1D sequence
def zigzag_scan(block):
    return np.array([
        block[row, col] for row, col in ZIGZAG_ORDER
    ])


# Apply zigzag scanning to every 8x8 block
def serialize_channel(quantized_image):
    height, width = quantized_image.shape
    serialized_blocks = []

    for i in range(0, height, 8):
        for j in range(0, width, 8):

            block = quantized_image[i:i+8, j:j+8]

            sequence = zigzag_scan(block)
            serialized_blocks.append(sequence)

    return serialized_blocks


# Serialize all three channels
zigzag_Y = serialize_channel(quantized_Y)
zigzag_Cr = serialize_channel(quantized_Cr)
zigzag_Cb = serialize_channel(quantized_Cb)


#apply DPCM to the DC coefficients of each channel
def apply_dpcm(zigzag_blocks):
    dc_values = [int(block[0]) for block in zigzag_blocks]

    differences = []
    previous_dc = 0

    for dc in dc_values:
        differences.append(dc - previous_dc)
        previous_dc = dc

    return differences


dpcm_Y = apply_dpcm(zigzag_Y)
dpcm_Cr = apply_dpcm(zigzag_Cr)
dpcm_Cb = apply_dpcm(zigzag_Cb)
dc_values_Y = [int(block[0]) for block in zigzag_Y]

print("First 10 Y DC values:", dc_values_Y[:10])

print(" first block of Y-channel DC differences:", dpcm_Y[0:10])
print(" first block of Cr-channel DC differences:", dpcm_Cr[0:10])
print(" first block of Cb-channel DC differences:", dpcm_Cb[0:10])




# Run length encode the AC coefficients of one block
def run_length_encode_ac(sequence):
    ac = sequence[1:]  # Skip the DC 
    symbols = []
    zero_run = 0

    # Find the last nonzero AC coefficient
    nonzero = np.nonzero(ac)[0]
    last = nonzero[-1] if len(nonzero) > 0 else -1

    for k in range(last + 1):
        value = int(ac[k])

        if value == 0:
            zero_run += 1
            continue

        # Encode each group of 16 zeros using ZRL
        while zero_run >= 16:
            symbols.append((15, 0))
            zero_run -= 16

        # Store remaining zero run and nonzero value
        symbols.append((zero_run, value))
        zero_run = 0

    # Signal that the rest of the block contains zeros
    if last < len(ac) - 1:
        symbols.append((0, 0))  # EOB

    return symbols


# Process all blocks in a channel
def run_length_encode_channel(zigzag_blocks):
    symbols = []

    for block in zigzag_blocks:
        symbols.extend(run_length_encode_ac(block))

    return symbols


# Apply RLE to all three channels
rle_ac_Y = run_length_encode_channel(zigzag_Y)
rle_ac_Cr = run_length_encode_channel(zigzag_Cr)
rle_ac_Cb = run_length_encode_channel(zigzag_Cb)



def get_category(value):
    """Number of bits needed to represent a coefficient magnitude."""
    value = int(value)

    if value == 0:
        return 0

    return abs(value).bit_length()


def get_amplitude_bits(value):
    """JPEG-style magnitude bits for positive and negative values."""
    value = int(value)

    if value == 0:
        return ""

    category = get_category(value)

    if value > 0:
        return format(value, f"0{category}b")

    # JPEG represents negative values using the complement
    # of the corresponding positive magnitude bits.
    return format((1 << category) - 1 + value, f"0{category}b")



def prepare_ac_symbols(rle_symbols):
    huffman_symbols = []
    amplitude_bits = []

    for run, value in rle_symbols:
        if (run, value) == (0, 0):       # EOB
            huffman_symbols.append((0, 0))
            amplitude_bits.append("")
        elif (run, value) == (15, 0):    # ZRL
            huffman_symbols.append((15, 0))
            amplitude_bits.append("")
        else:
            category = get_category(value)

            huffman_symbols.append((run, category))
            amplitude_bits.append(get_amplitude_bits(value))

    return huffman_symbols, amplitude_bits



class HuffmanNode:
    def __init__(self, symbol=None, left=None, right=None):
        self.symbol = symbol
        self.left = left
        self.right = right


def build_huffman_codes(data):
    if not data:
        return {}

    frequency = Counter(data)
    heap = []
    counter = 0

    for symbol, freq in frequency.items():
        node = HuffmanNode(symbol=symbol)
        heapq.heappush(heap, (freq, counter, node))
        counter += 1

    if len(heap) == 1:
        symbol = heap[0][2].symbol
        return {symbol: "0"}

    while len(heap) > 1:
        freq1, _, left = heapq.heappop(heap)
        freq2, _, right = heapq.heappop(heap)

        parent = HuffmanNode(left=left, right=right)

        heapq.heappush(
            heap, (freq1 + freq2, counter, parent)
        )
        counter += 1

    root = heap[0][2]
    codes = {}

    def generate_codes(node, code):
        if node.symbol is not None:
            codes[node.symbol] = code
            return

        generate_codes(node.left, code + "0")
        generate_codes(node.right, code + "1")

    generate_codes(root, "")
    return codes



ac_symbols_Y, ac_amplitude_Y = prepare_ac_symbols(rle_ac_Y)
ac_symbols_Cr, ac_amplitude_Cr = prepare_ac_symbols(rle_ac_Cr)
ac_symbols_Cb, ac_amplitude_Cb = prepare_ac_symbols(rle_ac_Cb)

dc_symbols_Y = [get_category(value) for value in dpcm_Y]
dc_symbols_Cr = [get_category(value) for value in dpcm_Cr]
dc_symbols_Cb = [get_category(value) for value in dpcm_Cb]



huffman_AC_Y = build_huffman_codes(ac_symbols_Y)
huffman_AC_Cr = build_huffman_codes(ac_symbols_Cr)
huffman_AC_Cb = build_huffman_codes(ac_symbols_Cb)



huffman_DC_Y = build_huffman_codes(dc_symbols_Y)
huffman_DC_Cr = build_huffman_codes(dc_symbols_Cr)
huffman_DC_Cb = build_huffman_codes(dc_symbols_Cb)


def encode_ac_channel(symbols, amplitude_bits, huffman_codes):
    bitstream = ""

    for symbol, extra_bits in zip(symbols, amplitude_bits):
        bitstream += huffman_codes[symbol] + extra_bits

    return bitstream


def encode_dc_channel(dpcm_values, huffman_codes):
    bitstream = ""

    for difference in dpcm_values:
        category = get_category(difference)

        bitstream += huffman_codes[category]
        bitstream += get_amplitude_bits(difference)

    return bitstream


encoded_Y = (
    encode_dc_channel(dpcm_Y, huffman_DC_Y)
    + encode_ac_channel(ac_symbols_Y, ac_amplitude_Y, huffman_AC_Y)
)

encoded_Cr = (
    encode_dc_channel(dpcm_Cr, huffman_DC_Cr)
    + encode_ac_channel(ac_symbols_Cr, ac_amplitude_Cr, huffman_AC_Cr)
)

encoded_Cb = (
    encode_dc_channel(dpcm_Cb, huffman_DC_Cb)
    + encode_ac_channel(ac_symbols_Cb, ac_amplitude_Cb, huffman_AC_Cb)
)

print("Y encoded bits:", len(encoded_Y))
print("Cr encoded bits:", len(encoded_Cr))
print("Cb encoded bits:", len(encoded_Cb))
print("Total encoded bits:", len(encoded_Y) + len(encoded_Cr) + len(encoded_Cb))




total_bits = len(encoded_Y) + len(encoded_Cr) + len(encoded_Cb)
original_bits = img.shape[0] * img.shape[1] * 24
print(f"Compression ratio: {original_bits / total_bits:.2f} : 1")


# Pack a string of "0"/"1" characters into real bytes (8 bits per byte)
def bits_to_bytes(bitstream):
    padding = (8 - len(bitstream) % 8) % 8
    bitstream += "0" * padding

    data = bytearray()
    for i in range(0, len(bitstream), 8):
        data.append(int(bitstream[i:i+8], 2))

    return bytes(data)


# Everything a decoder needs to rebuild the image
compressed = {
    "image_shape": img.shape[:2],
    "padded_shapes": {"Y": Y.shape, "Cr": Cr.shape, "Cb": Cb.shape},
    "bit_lengths": {"Y": len(encoded_Y), "Cr": len(encoded_Cr), "Cb": len(encoded_Cb)},
    "huffman_DC": {"Y": huffman_DC_Y, "Cr": huffman_DC_Cr, "Cb": huffman_DC_Cb},
    "huffman_AC": {"Y": huffman_AC_Y, "Cr": huffman_AC_Cr, "Cb": huffman_AC_Cb},
    "data": {
        "Y": bits_to_bytes(encoded_Y),
        "Cr": bits_to_bytes(encoded_Cr),
        "Cb": bits_to_bytes(encoded_Cb),
    },
}

with open("compressed_image.bin", "wb") as f:
    pickle.dump(compressed, f)

print("Saved compressed_image.bin")

