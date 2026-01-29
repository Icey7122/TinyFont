import struct
import os

class TyfPacker:
    """VECF 矢量字形打包器

    按照 VECF 二进制规范生成字形数据文件，支持按连续区间分段以提高存储效率。

    Attributes:
        font_id (int): 字体标识符，写入文件的头部字段。
        glyph_map (dict): 暂存字形数据的映射，键为 Unicode 码点，值为笔画列表。
    """

    def __init__(self, font_id=0x00000015):
        self.font_id = font_id
        self.glyph_map = {}  # 暂存字形以便去重与排序

    def add_glyph(self, code, strokes):
        """暂存字形数据。

        Args:
            code (int): 字形的 Unicode 码点。
            strokes (list): 描述字形的笔画列表。
        """
        self.glyph_map[code] = strokes

    def _pack_24bit_ptr(self, address):
        """将 32 位地址转换为 24 位指针并以小端返回前三字节。

        设备使用右移两位的 24 位指针表示方式，因此本函数返回小端字节序的低三字节。

        Args:
            address (int): 原始字节偏移地址（32 位）。

        Returns:
            bytes: 3 字节的小端指针表示。
        """
        val = address >> 2
        return struct.pack('<I', val)[:3]

    def _generate_sections(self):
        """根据连续性与硬件限制将字形分段。

        分段规则考虑缺口（gap）、段内最大字数（256）、以及是否跨越 Unicode 平面。

        Returns:
            list: 每个段为包含 'start' 和 'glyphs' 的字典。
        """
        sorted_codes = sorted(self.glyph_map.keys())
        if not sorted_codes:
            return []

        sections = []
        current_sec = {'start': sorted_codes[0], 'glyphs': [self.glyph_map[sorted_codes[0]]]}
        
        for i in range(1, len(sorted_codes)):
            prev_code = sorted_codes[i-1]
            curr_code = sorted_codes[i]
            gap = curr_code - prev_code - 1
            
            # 分段判定条件：缺口过大 / 段容量已满 / 跨平面
            is_too_wide = (len(current_sec['glyphs']) + gap) >= 256
            is_new_plane = (prev_code >> 16) != (curr_code >> 16)
            
            if gap > 14 or is_too_wide or is_new_plane:
                sections.append(current_sec)
                current_sec = {'start': curr_code, 'glyphs': [self.glyph_map[curr_code]]}
            else:
                # 在缺口处插入 None 以保持索引连续性
                for _ in range(gap):
                    current_sec['glyphs'].append(None)
                current_sec['glyphs'].append(self.glyph_map[curr_code])
        
        sections.append(current_sec)
        return sections

    def finish(self, filename):
        """将暂存的字形按 VECF 格式写入文件。

        Args:
            filename (str): 输出文件路径。
        """
        sections = self._generate_sections()
        if not sections: return

        blocks_bin = bytearray()
        index_entries = []
        
        # 初始绝对偏移：文件头(12B) + 索引区(每段 8B)
        current_abs_offset = 12 + len(sections) * 8
        
        for sec in sections:
            # 保证 Section Block 按 4 字节对齐
            padding = (4 - (current_abs_offset % 4)) % 4
            blocks_bin.extend(b'\x00' * padding)
            current_abs_offset += padding
            
            block_start_ptr = current_abs_offset
            
            # 生成段内的字形字节流
            glyph_stream = bytearray()
            for strokes in sec['glyphs']:
                if not strokes:
                    glyph_stream.append(0)
                    continue
                
                glyph_data = bytearray()
                for st in strokes:
                    for i, (fx, fy) in enumerate(st):
                        ix = int(max(0.0, min(1.0, fx)) * 127.0) & 0x7F
                        iy = int(max(0.0, min(1.0, fy)) * 127.0) & 0x7F
                        if i == 0: ix |= 0x80
                        glyph_data.append(ix); glyph_data.append(iy)
                
                point_count = min(255, len(glyph_data) // 2)
                glyph_stream.append(point_count)
                glyph_stream.extend(glyph_data[:point_count*2])
            
            # 计算 Block Header 并拼接完整块
            meta_off = (len(glyph_stream) + 4 + 3) // 4
            block_header = struct.pack('<bbH', 0, 0, meta_off)
            
            full_block = block_header + glyph_stream
            blocks_bin.extend(full_block)
            
            # 构建索引项
            plane_id = (sec['start'] >> 16) & 0x1F
            props = (plane_id << 11)
            start_code_low = sec['start'] & 0xFFFF
            count_field = len(sec['glyphs']) - 1
            
            entry = struct.pack('<HHB', props, start_code_low, count_field)
            entry += self._pack_24bit_ptr(block_start_ptr)
            index_entries.append(entry)
            
            current_abs_offset += len(full_block)

        with open(filename, 'wb') as f:
            f.write(b'VECF')
            f.write(struct.pack('<IHH', self.font_id, 0, len(sections)))
            for entry in index_entries: f.write(entry)
            f.write(blocks_bin)

        print(f"[VECF] 优化完成！分段数: {len(sections)}, 大小: {os.path.getsize(filename)/1024:.1f} KB")

class TyfParser:
    """解析 VECF 格式字形文件并按码点返回笔画数据。

    Attributes:
        data (bytes|None): 原始文件数据。
        sections (list): 解析出的段信息列表，每项包含 'start','count','ptr'。
    """

    def __init__(self):
        self.data = None
        self.sections = []

    def load(self, filename):
        """从文件加载并解析节索引。

        Args:
            filename (str): 要加载的 VECF 文件路径。

        Returns:
            bool: 加载并解析成功返回 True，失败返回 False。
        """
        try:
            with open(filename, 'rb') as f:
                self.data = f.read()

            if self.data[0:4] != b'VECF':
                return False

            # 解析全局头部：Magic(4) 已跳过，后续为 FontID(4), Flags(2), Count(2)
            header = struct.unpack('<IHH', self.data[4:12])
            cnt_section = header[2]

            self.sections = []
            idx_ptr = 12

            for _ in range(cnt_section):
                # 索引项格式：'<HHBBBB' 共 8 字节
                entry_raw = self.data[idx_ptr: idx_ptr + 8]
                props, start_code, count, p0, p1, p2 = struct.unpack('<HHBBBB', entry_raw)

                plane_id = (props >> 11) & 0x1F
                full_start = (plane_id << 16) | start_code

                # 重建 24 位指针并左移两位得到绝对偏移
                abs_offset = (p0 | (p1 << 8) | (p2 << 16)) << 2

                self.sections.append({
                    'start': full_start,
                    'count': count + 1,
                    'ptr': abs_offset
                })
                idx_ptr += 8
            return True
        except Exception as e:
            print(f"Load Error: {e}")
            return False

    def get_strokes(self, unicode_val):
        """按码点返回字形的笔画序列。

        Args:
            unicode_val (int): 请求的 Unicode 码点。

        Returns:
            list: 笔画列表，若未命中或为空笔画则返回空列表。
        """
        if self.data is None:
            return []

        target_sec = None
        for sec in self.sections:
            if sec['start'] <= unicode_val < sec['start'] + sec['count']:
                target_sec = sec
                break
        if not target_sec:
            return []

        blk_ptr = target_sec['ptr']
        local_idx = unicode_val - target_sec['start']
        curr_glyph_ptr = blk_ptr + 4  # 跳过 Block Header

        # 按长度跳过前面的字形条目（变长编码）
        for i in range(local_idx):
            glyph_len = self.data[curr_glyph_ptr]
            curr_glyph_ptr += 1 + (glyph_len << 1)

        target_len = self.data[curr_glyph_ptr]
        if target_len == 0:
            return []

        raw_bytes = self.data[curr_glyph_ptr + 1: curr_glyph_ptr + 1 + (target_len << 1)]

        strokes = []
        current_stroke = []
        for i in range(0, len(raw_bytes), 2):
            bx, by = raw_bytes[i], raw_bytes[i + 1]
            fx, fy = (bx & 0x7F) / 127.0, (by & 0x7F) / 127.0
            if (bx & 0x80) != 0:
                if current_stroke:
                    strokes.append(current_stroke)
                current_stroke = [(fx, fy)]
            else:
                current_stroke.append((fx, fy))
        if current_stroke:
            strokes.append(current_stroke)
        return strokes