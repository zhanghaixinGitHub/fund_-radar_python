"""本次3份已核制造主题报告的旧Word正文只读适配，标准库实现。

按微软MS-CFB读取分配链、MS-DOC的FIB/CLX/PlcPcd读取主正文；不调用Office、
不执行宏、不读取外部链接。正文仅供人工准入原句定位，连续正文不冒称物理页码。
只接纳本次3个已独立验证的原件摘要，其他Word须另做实际适配与独立核查。
规范依据：https://learn.microsoft.com/en-us/openspecs/office_file_formats/ms-doc/01d5d8c4-cf9c-4ef9-80fd-439e763cfe01
https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-cfb/05060311-bfce-4b12-874d-71fd4ce63aea
"""

import re
import struct

from app.services.material_content import content_digest

KNOWN_ORIGINALS = {
    "63d097482ba987ca63fa9d757d01fe2fb4eb5a41dae628a41947d67191a9c6c3",
    "16e00f554267b7c2ee566385dd6f96b3fa9af0ffcc3071095be3c2791c98c182",
    "889dba584f4b1a02705b00090ad259f6b26c5be0a7f63bdebd1b7c34b52d5a27",
}
END = 0xFFFFFFFE
FREE = 0xFFFFFFFF
MAXREG = 0xFFFFFFFA


def require(value, label):
    if not value:
        raise ValueError("YY_WORD_" + label)


def u16(raw, offset):
    require(0 <= offset <= len(raw) - 2, "SHORT_UINT16")
    return struct.unpack_from("<H", raw, offset)[0]


def u32(raw, offset):
    require(0 <= offset <= len(raw) - 4, "SHORT_UINT32")
    return struct.unpack_from("<I", raw, offset)[0]


class Compound:
    """仅解析正文所需的普通流/小流；所有链均按文件边界及已访问集合限长。"""

    def __init__(self, raw):
        require(
            1536 <= len(raw) <= 50_000_000 and raw[:8] == bytes.fromhex("d0cf11e0a1b11ae1"), "CFB_SIGNATURE_OR_SIZE"
        )
        self.raw = raw
        self.major = u16(raw, 26)
        shift = u16(raw, 30)
        require(u16(raw, 28) == 0xFFFE and (self.major, shift) in {(3, 9), (4, 12)} and u16(raw, 32) == 6, "CFB_HEADER")
        self.sector_size = 1 << shift
        self.sectors = len(raw) // self.sector_size - 1
        require(len(raw) % self.sector_size == 0 and self.sectors >= 2, "CFB_SECTOR_ALIGNMENT")
        require(u32(raw, 56) == 4096, "CFB_MINI_CUTOFF")
        count = u32(raw, 44)
        require(0 < count <= self.sectors, "CFB_FAT_COUNT")
        ids = [u32(raw, 76 + 4 * i) for i in range(109)]
        ids = [i for i in ids if i < MAXREG]
        next_difat = u32(raw, 68)
        difat_count = u32(raw, 72)
        seen = set()
        require(difat_count <= self.sectors, "CFB_DIFAT_COUNT")
        for _ in range(difat_count):
            require(next_difat not in seen, "CFB_DIFAT_CYCLE")
            seen.add(next_difat)
            data = self.sector(next_difat)
            items = struct.unpack("<" + "I" * (self.sector_size // 4), data)
            ids.extend(i for i in items[:-1] if i < MAXREG)
            next_difat = items[-1]
        require(next_difat in {END, FREE} and len(ids) == count and len(set(ids)) == count, "CFB_DIFAT_CLOSURE")
        self.fat = [x for i in ids for x in struct.unpack("<" + "I" * (self.sector_size // 4), self.sector(i))]
        directory = self.chain(u32(raw, 48), self.fat, self.sector, self.sectors)
        entries = []
        for offset in range(0, len(directory), 128):
            entry = directory[offset : offset + 128]
            require(len(entry) == 128, "CFB_DIRECTORY_ENTRY")
            if entry[66] == 0:
                continue
            length = u16(entry, 64)
            require(2 <= length <= 64 and length % 2 == 0, "CFB_DIRECTORY_NAME")
            name = entry[: length - 2].decode("utf-16-le")
            size = u32(entry, 120) if self.major == 3 else struct.unpack_from("<Q", entry, 120)[0]
            require(size <= 50_000_000, "CFB_STREAM_SIZE")
            entries.append({"name": name, "type": entry[66], "start": u32(entry, 116), "size": size})
        roots = [e for e in entries if e["type"] == 5]
        require(len(roots) == 1, "CFB_ROOT")
        self.root = roots[0]
        self.entries = {}
        for entry in entries:
            if entry["type"] == 2:
                require(entry["name"] not in self.entries, "CFB_DUPLICATE_STREAM")
                self.entries[entry["name"]] = entry
        self.mini_fat = None
        self.mini_data = None

    def sector(self, number):
        require(0 <= number < self.sectors, "CFB_SECTOR_OUT_OF_RANGE")
        offset = (number + 1) * self.sector_size
        return self.raw[offset : offset + self.sector_size]

    @staticmethod
    def chain(start, table, reader, limit):
        blocks = []
        seen = set()
        current = start
        while current != END:
            require(
                0 <= current < len(table) and current < MAXREG and current not in seen and len(seen) < limit,
                "CFB_CHAIN_CYCLE_OR_RANGE",
            )
            seen.add(current)
            blocks.append(reader(current))
            current = table[current]
        return b"".join(blocks)

    def normal(self, start, size):
        if size == 0:
            return b""
        raw = self.chain(start, self.fat, self.sector, self.sectors)
        require(len(raw) >= size, "CFB_TRUNCATED_STREAM")
        return raw[:size]

    def read(self, name):
        require(name in self.entries, "CFB_REQUIRED_STREAM:" + name)
        entry = self.entries[name]
        size = entry["size"]
        if size >= 4096 or size == 0:
            return self.normal(entry["start"], size)
        if self.mini_fat is None:
            count = u32(self.raw, 64)
            require(0 < count <= self.sectors, "CFB_MINIFAT_COUNT")
            data = self.normal(u32(self.raw, 60), count * self.sector_size)
            self.mini_fat = list(struct.unpack("<" + "I" * (len(data) // 4), data))
            self.mini_data = self.normal(self.root["start"], self.root["size"])

        def mini(number):
            require(0 <= number * 64 <= len(self.mini_data) - 64, "CFB_MINISECTOR_RANGE")
            return self.mini_data[number * 64 : (number + 1) * 64]

        raw = self.chain(entry["start"], self.mini_fat, mini, len(self.mini_data) // 64)
        require(len(raw) >= size, "CFB_TRUNCATED_MINISTREAM")
        return raw[:size]


def clean_body(text):
    """把Word字段保留为显示结果，规范段落/单元格分隔；不替换或猜补业务数字。"""
    require("\x1f" not in text, "UNREVIEWED_CONTROL_MARKER")
    text = text.translate({2: 0, 5: 0, 7: 9, 8: 0, 10: 10, 11: 10, 12: 10, 13: 10})
    pattern = re.compile(r"\x13[^\x13\x14\x15]*\x14?([^\x13\x14\x15]*)\x15")
    for _ in range(128):
        updated, count = pattern.subn(r"\1", text)
        text = updated
        if count == 0:
            break
    else:
        raise ValueError("YY_WORD_FIELD_NESTING_LIMIT")
    text = re.sub(r"[\x00-\x07]", "", text)
    return text.translate(
        {
            0x2002: " ",
            0x2003: " ",
            0x2012: "-",
            0x2013: "-",
            0x2014: "-",
            0x2018: "'",
            0x2019: "'",
            0x201C: '"',
            0x201D: '"',
        }
    )


def extract_body(raw):
    """原始字节入、主正文出；不使用旧文本缓存、不执行任何文档对象。"""
    require(content_digest(raw) in KNOWN_ORIGINALS, "UNREVIEWED_ORIGINAL")
    from app.services.material_content import SourceContent

    if isinstance(raw, SourceContent) and raw.kind == "LEGACY_DOC":
        return raw.extracted["text"]
    compound = Compound(raw)
    word = compound.read("WordDocument")
    require(u16(word, 0) == 0xA5EC and u16(word, 2) >= 0xC1, "FIB_VERSION")
    flags = u16(word, 10)
    require(not flags & 0x8100, "ENCRYPTED_OR_OBFUSCATED_UNSUPPORTED")
    table = compound.read("1Table" if flags & 0x200 else "0Table")
    csw = u16(word, 32)
    cslw_offset = 34 + 2 * csw
    cslw = u16(word, cslw_offset)
    longs = cslw_offset + 2
    require(csw == 14 and cslw >= 22, "FIB_LAYOUT")
    characters = u32(word, longs + 12)
    require(15000 <= characters <= 10_000_000, "BODY_CHARACTER_COUNT")
    fc_lcb_count_offset = longs + cslw * 4
    require(u16(word, fc_lcb_count_offset) >= 34, "FIB_FC_LCB_COUNT")
    fc_lcb = fc_lcb_count_offset + 2
    clx_start = u32(word, fc_lcb + 33 * 8)
    clx_size = u32(word, fc_lcb + 33 * 8 + 4)
    require(clx_size > 5 and clx_start + clx_size <= len(table), "CLX_BOUNDS")
    clx = table[clx_start : clx_start + clx_size]
    offset = 0
    while offset < len(clx) and clx[offset] == 1:
        offset += 3 + u16(clx, offset + 1)
    require(offset < len(clx) and clx[offset] == 2, "CLX_PCDT")
    size = u32(clx, offset + 1)
    start = offset + 5
    require(size >= 16 and (size - 4) % 12 == 0 and start + size <= len(clx), "PLC_PCD_SIZE")
    pieces = (size - 4) // 12
    require(pieces <= 100000, "PLC_PCD_COUNT")
    positions = [u32(clx, start + i * 4) for i in range(pieces + 1)]
    require(
        positions[0] == 0
        and all(a <= b for a, b in zip(positions, positions[1:], strict=False))
        and positions[-1] >= characters,
        "PLC_CP_COVERAGE",
    )
    body = []
    seen_characters = 0
    for i in range(pieces):
        begin, end = positions[i], positions[i + 1]
        if begin >= characters:
            break
        count = min(end, characters) - begin
        pcd = start + 4 * (pieces + 1) + 8 * i
        compressed = u32(clx, pcd + 2)
        require(not compressed & 0x80000000, "FC_RESERVED_BIT")
        fc = compressed & 0x3FFFFFFF
        ansi = bool(compressed & 0x40000000)
        require(not ansi or fc % 2 == 0, "FC_COMPRESSED_ALIGNMENT")
        offset = fc // 2 if ansi else fc
        length = count if ansi else count * 2
        require(offset + length <= len(word), "PIECE_TEXT_BOUNDS")
        chunk = word[offset : offset + length]
        if ansi:
            # MS-DOC的8位字符采用Windows映射；未定义的控制码保留为相同码位。
            text = "".join(
                bytes([v]).decode("cp1252") if v not in {0x81, 0x8D, 0x8F, 0x90, 0x9D} else chr(v) for v in chunk
            )
        else:
            text = chunk.decode("utf-16-le")
        body.append(text)
        seen_characters += count
    require(seen_characters == characters, "MAIN_STORY_CLOSURE")
    value = clean_body("".join(body))
    require(len(value) > 15000 and "014380" in value and "建信中国制造" in value, "REPORT_BODY_IDENTITY")
    return value
