import os
import itertools
from hanzi_chaizi import HanziChaizi
from PIL import Image, ImageDraw, ImageFont

# 常见偏旁集合
COMMON_RADICALS = ["氵", "亻", "口", "木", "女", "心", "手", "扌", "辶", "艹", "日", "月"]
LEFT_RIGHT_RADICALS = {"氵", "亻", "口", "木", "女", "心", "手", "扌", "辶"}
TOP_BOTTOM_RADICALS = {"艹", "日", "月"}

# 初始化汉字拆字工具
chaizi_tool = HanziChaizi()

def decompose_hanzi(character):
    """拆解汉字为部件"""
    try:
        components = chaizi_tool.query(character)
        if components:
            return components[0]  # 返回拆分后的第一个结果
    except Exception as e:
        print(f"Error decomposing {character}: {e}")
    return []

def generate_bitmap(character, font_path='simsun.ttc', size=(64, 64)):
    """生成汉字的64x64位图并返回PIL图片"""
    try:
        try:
            font = ImageFont.truetype(font_path, 64)
        except Exception:
            font = ImageFont.load_default()
        image = Image.new('1', size, 1)  # 创建白底黑字的图片
        draw = ImageDraw.Draw(image)
        draw.text((0, 0), character, font=font, fill=0)
        return image
    except Exception as e:
        print(f"Error generating bitmap for {character}: {e}")
        return None

def image_to_array(image):
    """将PIL图片转换为嵌套列表"""
    if image is None:
        return []
    width, height = image.size
    return [[1 if image.getpixel((x, y)) == 0 else 0 for x in range(width)] for y in range(height)]

def compose_bitmaps(radical_img, base_img, structure='left-right'):
    """按结构组合两个部件的位图"""
    if radical_img is None or base_img is None:
        return None
    if structure == 'top-bottom':
        radical_resized = radical_img.resize((64, 32))
        base_resized = base_img.resize((64, 32))
        canvas = Image.new('1', (64, 64), 1)
        canvas.paste(radical_resized, (0, 0))
        canvas.paste(base_resized, (0, 32))
    else:
        radical_resized = radical_img.resize((32, 64))
        base_resized = base_img.resize((32, 64))
        canvas = Image.new('1', (64, 64), 1)
        canvas.paste(radical_resized, (0, 0))
        canvas.paste(base_resized, (32, 0))
    return canvas

def find_related_hanzi(base_character, characters):
    """查找由base_character加偏旁部首后形成的汉字"""
    base_components = decompose_hanzi(base_character)

    if not base_components:
        # 如果原汉字无法拆解，直接找包含它的字符
        for char in characters:
            char_components = decompose_hanzi(char)
            if base_character in char_components:
                return char  # 找到第一个包含的字符即返回
    else:
        for char in characters:
            char_components = decompose_hanzi(char)
            if base_character in char_components and base_components != char_components:
                return char  # 找到第一个匹配的字符即返回

    return None

def save_results_as_array(results, output_dir="output"):
    """保存结果为二维数组的txt文件"""
    os.makedirs(output_dir, exist_ok=True)

    base_file = os.path.join(output_dir, "base_character1.txt")
    new_file = os.path.join(output_dir, "new_character1.txt")

    with open(base_file, "w", encoding="utf-8") as bf:
        bf.write("[\n")
        for base, _ in results:
            bitmap = image_to_array(generate_bitmap(base))
            bf.write("    [\n")
            for row in bitmap:
                bf.write("        [" + ", ".join(map(str, row)) + "],\n")
            bf.write("    ],\n")
        bf.write("]\n")

    with open(new_file, "w", encoding="utf-8") as nf:
        nf.write("[\n")
        for _, new_char in results:
            if new_char:
                bitmap = image_to_array(generate_bitmap(new_char))
                nf.write("    [\n")
                for row in bitmap:
                    nf.write("        [" + ", ".join(map(str, row)) + "],\n")
                nf.write("    ],\n")
        nf.write("]\n")

def save_synth_results(base_arrays, target_arrays, output_dir="output"):
    """保存合成结果"""
    os.makedirs(output_dir, exist_ok=True)
    base_path = os.path.join(output_dir, "synth_base.txt")
    target_path = os.path.join(output_dir, "synth_target.txt")

    def _write(file_path, arrays):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("[\n")
            for bitmap in arrays:
                f.write("    [\n")
                for row in bitmap:
                    f.write("        [" + ", ".join(map(str, row)) + "],\n")
                f.write("    ],\n")
            f.write("]\n")

    _write(base_path, base_arrays)
    _write(target_path, target_arrays)

if __name__ == "__main__":
    characters_to_check = [chr(i) for i in range(0x4e00, 0x9e00)]
    limit = int(os.getenv("LIMIT_CHARS", "0"))
    if limit:
        characters_to_check = characters_to_check[:limit]

    # 预计算已有的组合对
    decomposition_map = {c: decompose_hanzi(c) for c in characters_to_check}
    existing_pairs = set()
    for comps in decomposition_map.values():
        for a, b in itertools.permutations(comps, 2):
            existing_pairs.add((a, b))

    synth_base_arrays = []
    synth_target_arrays = []

    print("Generating synthetic combinations...")
    for base_character in characters_to_check:
        base_img = generate_bitmap(base_character)
        base_array = image_to_array(base_img)
        for radical in COMMON_RADICALS:
            if (base_character, radical) in existing_pairs or (radical, base_character) in existing_pairs:
                continue
            radical_img = generate_bitmap(radical)
            structure = 'top-bottom' if radical in TOP_BOTTOM_RADICALS else 'left-right'
            composed = compose_bitmaps(radical_img, base_img, structure)
            synth_base_arrays.append(base_array)
            synth_target_arrays.append(image_to_array(composed))

    print(f"Generated {len(synth_target_arrays)} synthetic samples.")
    save_synth_results(synth_base_arrays, synth_target_arrays)
    print("Synthetic results saved in the output directory.")
