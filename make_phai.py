import random
import rstr

NUM = r"(?:[1-9]\d{0,2}(?:,\d{3})?|\d{1,5})(?:\.\d{1,2})?"
PHI = r"(?:[13]φ)"
WIRE = r"(?:[23]W)"
SP = r"(?: )?"  # 半角スペース0〜1個

patterns = [
    rf"{NUM}kVA",
    rf"{NUM}kV",
    rf"{NUM}kA",
    rf"{NUM}V",
    rf"{NUM}W",
    rf"{PHI}{SP}{NUM}kVA",
    rf"{PHI}{SP}{NUM}W",
    rf"{PHI}{SP}{WIRE}",
    rf"{PHI}{SP}{WIRE}{SP}{NUM}(?:[/~]{NUM})?V",
    rf"{PHI}{SP}{WIRE}{SP}{NUM}(?:[/~]{NUM})?V{SP}{NUM}kVA",
    rf"{PHI}{SP}AC{SP}{NUM}V",
    rf"{NUM}kVA{SP}x{SP}\d{1,2}",
    rf"{NUM}AT",
    rf"{NUM}AH(?:/\d{1,2}HR)?",
]


def sample_one():
    pat = random.choice(patterns)
    return rstr.xeger(pat)


corpus = [sample_one() for _ in range(1000)]

with open("denki.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(corpus))
