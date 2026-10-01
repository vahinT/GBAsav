
#!/usr/bin/env python3
"""
Pokemon FireRed .state_1 <-> .sav utility

Supports:
    1. Extracting the embedded 128 KiB FireRed .sav from a .state_1 file.
    2. Replacing the save data inside an existing .state_1 with a .sav.

Important:
    A .sav cannot be converted into a completely new emulator state by itself.
    A state file contains CPU/RAM/emulator information in addition to the save.
    Therefore --inject requires an existing .state_1 as the template.

Usage:

    Extract:
        python firered_state.py extract game.state_1 game.sav

    Inject:
        python firered_state.py inject original.state_1 game.sav new.state_1

"""

from pathlib import Path
import argparse
import struct
import shutil


SECTOR_SIZE = 0x1000          # 4096 bytes
SAVE_SIZE = 0x20000           # 128 KiB
SECTOR_COUNT = 28             # 14 sectors x 2 save copies
SIGNATURE = b"\x25\x20\x01\x08"

# Gen III section data sizes.
# The section footer occupies the last 12 bytes.
SECTION_DATA_SIZE = {
    0: 3884,
    1: 3968,
    2: 3968,
    3: 3968,
    4: 3848,
    5: 3968,
    6: 3968,
    7: 3968,
    8: 3968,
    9: 3968,
    10: 3968,
    11: 3968,
    12: 3968,
    13: 2000,
}


def checksum(data):
    """
    Pokémon Gen III section checksum.

    The game treats the section data as 32-bit little-endian words,
    adds them, then folds the result to 16 bits.
    """
    total = 0

    for offset in range(0, len(data), 4):
        word = struct.unpack_from("<I", data, offset)[0]
        total = (total + word) & 0xFFFFFFFF

    return ((total >> 16) + (total & 0xFFFF)) & 0xFFFF


def parse_sector(sector):
    """
    Read the Gen III footer from one 4096-byte sector.
    """
    if len(sector) != SECTOR_SIZE:
        raise ValueError("Invalid sector size")

    section_id = struct.unpack_from("<H", sector, 0xFF4)[0]
    stored_checksum = struct.unpack_from("<H", sector, 0xFF6)[0]
    signature = sector[0xFF8:0xFFC]
    counter = struct.unpack_from("<I", sector, 0xFFC)[0]

    data_size = SECTION_DATA_SIZE.get(section_id)

    if data_size is None:
        calculated = None
        valid = False
    else:
        calculated = checksum(sector[:data_size])
        valid = (
            signature == SIGNATURE
            and calculated == stored_checksum
        )

    return {
        "id": section_id,
        "checksum": stored_checksum,
        "calculated": calculated,
        "counter": counter,
        "signature": signature,
        "valid": valid,
    }


def find_save_copies(state):
    """
    Locate FireRed save-sector sequences inside a .state_1.

    A normal FireRed save has:
        - 14 sections
        - two copies
        - each sector = 0x1000 bytes
    """

    locations = []

    start = 0

    while True:
        pos = state.find(SIGNATURE, start)

        if pos == -1:
            break

        # Signature is at sector offset 0xFF8.
        sector_start = pos - 0xFF8

        if sector_start >= 0:
            locations.append(sector_start)

        start = pos + 1

    # Deduplicate and sort.
    locations = sorted(set(locations))

    copies = []

    # Look for runs of 14 consecutive sectors.
    for candidate in locations:
        sectors = []

        for i in range(14):
            expected = candidate + i * SECTOR_SIZE

            if expected not in locations:
                break

            sectors.append(expected)

        if len(sectors) != 14:
            continue

        raw = b"".join(
            state[p:p + SECTOR_SIZE]
            for p in sectors
        )

        parsed = [
            parse_sector(raw[i * SECTOR_SIZE:(i + 1) * SECTOR_SIZE])
            for i in range(14)
        ]

        # A legitimate copy should contain section IDs 0..13,
        # although their physical ordering can vary.
        ids = {x["id"] for x in parsed}

        if set(range(14)).issubset(ids):
            copies.append({
                "offset": candidate,
                "data": raw,
                "sections": parsed,
            })

    # Remove duplicate detections.
    unique = {}
    for copy in copies:
        unique[copy["offset"]] = copy

    return list(unique.values())


def choose_newest_copy(copies):
    """
    Choose the newest save copy using the Gen III save counter.

    Each 14-sector copy normally has one common counter.
    """

    if not copies:
        raise ValueError("No FireRed save copies found")

    def copy_counter(copy):
        counters = [
            s["counter"]
            for s in copy["sections"]
            if s["valid"]
        ]

        if not counters:
            return -1

        # Most sectors should share the same counter.
        return max(counters)

    return max(copies, key=copy_counter)


def extract_sav(state_path, sav_path):
    """
    Extract the 28-sector FireRed save representation.

    The state contains two 14-sector save copies. A standard 128 KiB
    GBA save image has room for both copies, so we preserve both.
    """

    state = Path(state_path).read_bytes()

    copies = find_save_copies(state)

    if len(copies) < 2:
        raise RuntimeError(
            f"Expected two FireRed save copies, found {len(copies)}"
        )

    copies.sort(key=lambda x: x["offset"])

    # The state contains two 14-sector copies.
    #
    # For this emulator format they occur consecutively:
    #
    #   copy A: 14 sectors
    #   copy B: 14 sectors
    #
    # Combine them into a 28-sector save image.
    raw_save = copies[0]["data"] + copies[1]["data"]

    # Pad to normal 128 KiB GBA save size.
    raw_save += b"\xFF" * (SAVE_SIZE - len(raw_save))

    Path(sav_path).write_bytes(raw_save)

    print("Extraction successful.")
    print(f"State: {state_path}")
    print(f"SAV:   {sav_path}")
    print(f"SAV size: {len(raw_save):,} bytes")
    print()

    for n, copy in enumerate(copies, 1):
        counters = sorted(
            set(s["counter"] for s in copy["sections"])
        )

        print(
            f"Copy {n}: "
            f"offset=0x{copy['offset']:X}, "
            f"counters={counters}, "
            f"valid sectors="
            f"{sum(s['valid'] for s in copy['sections'])}/14"
        )


def validate_sav(sav):
    """
    Validate the two 14-sector FireRed save copies.
    """

    if len(sav) < 28 * SECTOR_SIZE:
        raise ValueError(
            "SAV is too small. Expected at least "
            f"{28 * SECTOR_SIZE} bytes."
        )

    results = []

    for copy_number in range(2):

        base = copy_number * 14 * SECTOR_SIZE

        sections = []

        for i in range(14):

            sector = sav[
                base + i * SECTOR_SIZE:
                base + (i + 1) * SECTOR_SIZE
            ]

            sections.append(parse_sector(sector))

        results.append(sections)

    for n, sections in enumerate(results, 1):

        print(f"\nSave copy {n}")

        for s in sections:
            print(
                f"  section={s['id']:2d} "
                f"counter={s['counter']:10d} "
                f"checksum=0x{s['checksum']:04X} "
                f"calculated="
                f"{'0x%04X' % s['calculated'] if s['calculated'] is not None else 'N/A'} "
                f"{'OK' if s['valid'] else 'BAD'}"
            )


def inject_sav(state_path, sav_path, output_path):
    """
    Replace the save copies inside an existing state file.

    This does NOT construct a new emulator state from scratch.
    It modifies only the embedded FireRed save data.
    """

    state = bytearray(Path(state_path).read_bytes())
    sav = Path(sav_path).read_bytes()

    if len(sav) < 28 * SECTOR_SIZE:
        raise ValueError(
            "SAV must contain at least 114688 bytes "
            "(28 x 4096-byte sectors)."
        )

    copies = find_save_copies(state)

    if len(copies) < 2:
        raise RuntimeError(
            f"Expected two FireRed save copies in state, "
            f"found {len(copies)}"
        )

    copies.sort(key=lambda x: x["offset"])

    # The SAV's first 14 sectors are copy 1.
    # Its next 14 sectors are copy 2.
    save_copy_1 = sav[:14 * SECTOR_SIZE]
    save_copy_2 = sav[14 * SECTOR_SIZE:28 * SECTOR_SIZE]

    replacements = [
        (copies[0]["offset"], save_copy_1),
        (copies[1]["offset"], save_copy_2),
    ]

    for offset, replacement in replacements:
        state[offset:offset + len(replacement)] = replacement

    Path(output_path).write_bytes(state)

    print("Injection successful.")
    print(f"Template state: {state_path}")
    print(f"Input SAV:      {sav_path}")
    print(f"Output state:   {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract/inject Pokémon FireRed save data."
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True
    )

    extract = sub.add_parser(
        "extract",
        help="Extract .sav from .state_1"
    )

    extract.add_argument(
        "state",
        help="Input .state_1 file"
    )

    extract.add_argument(
        "sav",
        help="Output .sav file"
    )

    inject = sub.add_parser(
        "inject",
        help="Replace save data in an existing .state_1"
    )

    inject.add_argument(
        "state",
        help="Existing .state_1 template"
    )

    inject.add_argument(
        "sav",
        help="Input .sav file"
    )

    inject.add_argument(
        "output",
        help="Output .state_1 file"
    )

    validate = sub.add_parser(
        "validate",
        help="Validate a .sav file"
    )

    validate.add_argument(
        "sav",
        help="Input .sav"
    )

    args = parser.parse_args()

    if args.command == "extract":
        extract_sav(args.state, args.sav)

    elif args.command == "inject":
        inject_sav(args.state, args.sav, args.output)

    elif args.command == "validate":
        sav = Path(args.sav).read_bytes()
        validate_sav(sav)


if __name__ == "__main__":
    main()

