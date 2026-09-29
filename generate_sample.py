"""合成 RAW 测试素材生成器（CMOS 测试常用场景）。

用法：
    # 暗场 + 固定坏点 + 列缺陷（默认场景，10bit 1920x1080）
    python generate_sample.py

    # 阴影/暗角场景，12bit
    python generate_sample.py --scene shading --bit-depth 12 --out shading.raw

    # 过曝场景
    python generate_sample.py --scene saturated

    # 生成 MIPI RAW10 packed 文件（用来验证 packed 读取）
    python generate_sample.py --scene dark --packing packed --out dark_packed10.raw

    # 带 16B header + 每行 64B padding + 大端
    python generate_sample.py --scene dark --header 16 --stride-pad 64 --endian big --out dark_hdr.raw

    # 3 帧连续 dump（用来验证帧号切换）
    python generate_sample.py --scene dark --frames 3 --out dark_3frames.raw

    # 列出可用场景
    python generate_sample.py --list

每个场景都返回注入缺陷的清单（坐标/类型），方便和软件的检测结果对拍。
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from utils import cfa
from utils.raw_io import RawLoadSpec, pack_array

SCENES = ("dark", "shading", "saturated", "bayer", "flat")


def _plane_fill(pattern, values: dict, shape):
    """按相位把 values 填进 (H, W) 数组。values: {phase_name: DN}。"""
    out = np.zeros(shape, dtype=np.float32)
    names = cfa.PHASE_LABELS[cfa.normalize_pattern(pattern)]
    for phase, (dy, dx) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        out[dy::2, dx::2] = values.get(names[phase], 0)
    return out


def generate(scene: str = "dark", width: int = 1920, height: int = 1080,
             bit_depth: int = 10, pattern: str = "RGGB", seed: int = 0,
             frames: int = 1, dark_level: int = 64, signal: float = 0.6):
    """生成一帧（或 frames 帧）合成 RAW。

    返回 (data, meta)：
      data 为 uint16 数组：(H, W) 或 (frames, H, W)
      meta 含注入缺陷清单与场景说明，便于与检测结果对拍。
    """
    rng = np.random.default_rng(seed)
    max_code = (1 << bit_depth) - 1
    h, w = int(height), int(width)
    meta = {"scene": scene, "width": w, "height": h, "bit_depth": bit_depth,
            "pattern": pattern, "seed": seed, "injected": []}

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    frames_out = []

    for f in range(max(1, int(frames))):
        rng_f = np.random.default_rng(seed + 1000 * f)

        if scene == "flat":
            img = np.full((h, w), max_code // 2, dtype=np.float32)

        elif scene == "dark":
            # 暗场：黑电平 + 行/列 FPN + 读出噪声 + 固定坏点 + 坏行坏列 + 2x2 簇
            img = np.full((h, w), float(dark_level), dtype=np.float32)
            img += rng_f.normal(0, 1.5, (h, 1)).astype(np.float32)      # 行 FPN
            img += rng_f.normal(0, 1.0, (1, w)).astype(np.float32)      # 列 FPN
            img += rng_f.normal(0, max(0.5, dark_level * 0.05), (h, w)).astype(np.float32)

            # 位置用画面比例定义，任意分辨率都落在画内
            def px(fx, fy):
                return int(min(max(fx, 0.0), 0.98) * w), int(min(max(fy, 0.0), 0.98) * h)

            hot = [px(0.01, 0.01), px(0.21, 0.05), px(0.09, 0.09),
                   px(0.52, 0.60), px(0.93, 0.83)]
            dead = [px(0.07, 0.08), px(0.31, 0.65)]
            for (x, y) in hot:
                img[y, x] = max_code
                meta["injected"].append({"type": "hot", "x": x, "y": y, "value": max_code})
            for (x, y) in dead:
                img[y, x] = 0
                meta["injected"].append({"type": "dead", "x": x, "y": y, "value": 0})

            bad_col = int(0.33 * w)
            img[:, bad_col] += 180
            meta["injected"].append({"type": "col", "x": bad_col, "y": 0, "value": "+180DN"})
            bad_row = int(0.37 * h)
            img[bad_row, :] += 150          # 暗场里的"亮行"是典型的行 FPN 缺陷
            meta["injected"].append({"type": "row", "x": 0, "y": bad_row, "value": "+150DN"})

            # 2x2 坏点簇：四个相位各一个，模拟真实的簇状缺陷
            cx, cy = px(0.42, 0.55)
            img[cy:cy + 2, cx:cx + 2] = max_code
            for dy in range(2):
                for dx in range(2):
                    meta["injected"].append({"type": "cluster", "x": cx + dx,
                                             "y": cy + dy, "value": max_code})

        elif scene == "shading":
            # 均匀照明 + 光学暗角（cos^4 近似）+ 中心略亮
            r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
            img = max_code * signal * np.clip(1.0 - 0.55 * r ** 2, 0, 1)
            img = _plane_fill(pattern, {"R": 0.8, "Gr": 1.0, "Gb": 1.0, "B": 0.65}, (h, w)) \
                * (img / max(1e-6, img.max()))
            img += rng_f.normal(0, 2.0, (h, w)).astype(np.float32)
            meta["injected"].append({"type": "shading", "x": 0, "y": 0,
                                     "value": "中心亮、四周暗(约 55%)"})

        elif scene == "saturated":
            # 过曝：中间大面积削顶 + 边缘正常
            r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
            img = _plane_fill(pattern, {"R": 0.55, "Gr": 1.0, "Gb": 1.0, "B": 0.5}, (h, w))
            img *= np.clip(1.6 - 1.3 * r, 0.05, 1.6)
            img += rng_f.normal(0, 2.0, (h, w)).astype(np.float32)
            meta["injected"].append({"type": "sat", "x": w // 2, "y": h // 2,
                                     "value": "中心过曝削顶"})

        elif scene == "bayer":
            img = _plane_fill(pattern, {"R": 0.35, "Gr": 0.6, "Gb": 0.6, "B": 0.2},
                              (h, w)) * max_code
            img += (xx / max(1, w - 1)) * 0.25 * max_code            # 水平渐变
            img += rng_f.normal(0, 1.5, (h, w)).astype(np.float32)
            meta["injected"].append({"type": "pattern", "x": 0, "y": 0,
                                     "value": f"{pattern} 四通道台阶 + 水平渐变"})
        else:
            raise ValueError(f"未知场景 {scene}，可选：{SCENES}")

        img = np.clip(np.rint(img), 0, max_code).astype(np.uint16)
        frames_out.append(img)

    data = frames_out[0] if len(frames_out) == 1 else np.stack(frames_out)
    return data, meta


def write_sample(path: str, data: np.ndarray, meta: dict, packing: str = "unpacked",
                 header: int = 0, stride_pad: int = 0, endian: str = "little") -> dict:
    """按指定布局把合成数据写成文件，返回实际使用的布局信息。"""
    bit_depth = int(meta["bit_depth"])
    h, w = data.shape[-2], data.shape[-1]
    frames = 1 if data.ndim == 2 else data.shape[0]
    header_bytes = b"\xA5" * int(header)

    if packing == "packed":
        rows = [pack_array(data[i] if data.ndim == 3 else data, bit_depth)
                for i in range(frames)]
        stride = rows[0].shape[1] + int(stride_pad)
        body = bytearray()
        for r in rows:
            rr = np.zeros((h, stride), np.uint8)
            rr[:, :r.shape[1]] = r
            body += rr.tobytes()
    else:
        bpp = 1 if bit_depth <= 8 else 2
        dtype = np.dtype((">" if endian == "big" else "<") + ("u1" if bpp == 1 else "u2"))
        tight = w * bpp
        stride_bytes = tight + int(stride_pad)
        body = bytearray()
        for i in range(frames):
            frame = data[i] if data.ndim == 3 else data
            rows = np.zeros((h, stride_bytes), dtype=np.uint8)
            rows[:, :tight] = np.ascontiguousarray(
                frame.astype(dtype)).view(np.uint8).reshape(h, tight)
            body += rows.tobytes()

    with open(path, "wb") as fh:
        fh.write(header_bytes + bytes(body))

    spec = RawLoadSpec(width=w, height=h, bit_depth=bit_depth, packing=packing,
                       header_bytes=int(header), endian=endian,
                       stride_bytes=(row_bytes_of(w, bit_depth, packing) + int(stride_pad)))
    return {"path": path, "spec": spec.describe(), "frames": frames,
            "bytes": header + len(body), "spec_object": spec}


def row_bytes_of(width: int, bit_depth: int, packing: str) -> int:
    from utils.raw_io import row_bytes
    return row_bytes(width, bit_depth, packing)


def main():
    ap = argparse.ArgumentParser(description="生成合成 RAW 测试素材")
    ap.add_argument("--scene", default="dark", choices=SCENES + ("list",))
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--bit-depth", type=int, default=10)
    ap.add_argument("--pattern", default="RGGB",
                    choices=["RGGB", "BGGR", "GRBG", "GBRG", "Mono/None"])
    ap.add_argument("--dark-level", type=int, default=64)
    ap.add_argument("--signal", type=float, default=0.6)
    ap.add_argument("--frames", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--packing", default="unpacked", choices=["unpacked", "packed"])
    ap.add_argument("--header", type=int, default=0, help="文件头字节数")
    ap.add_argument("--stride-pad", type=int, default=0, help="每行填充字节数")
    ap.add_argument("--endian", default="little", choices=["little", "big"])
    ap.add_argument("--out", default="")
    ap.add_argument("--meta", default="", help="把注入缺陷清单写成 json")
    args = ap.parse_args()

    if args.scene == "list":
        print("可用场景：")
        for name, desc in {
            "dark": "暗场：行/列 FPN + 读出噪声 + 冷/热坏点 + 坏行坏列 + 2x2 坏点簇",
            "shading": "阴影/暗角：中心亮四周暗（约 55%），四通道响应不同",
            "saturated": "过曝：中心削顶饱和，边缘正常",
            "bayer": "Bayer 四通道台阶 + 水平渐变",
            "flat": "平坦场（半量程常数）",
        }.items():
            print(f"  {name:10s} {desc}")
        print("\n布局选项：--packing packed / --header N / --stride-pad N / --endian big")
        print("            --frames N / --bit-depth {8,10,12,14,16} / --pattern ...")
        return

    data, meta = generate(args.scene, args.width, args.height, args.bit_depth,
                          args.pattern, args.seed, args.frames, args.dark_level,
                          args.signal)
    out = args.out or (f"sample_{args.scene}_{args.width}x{args.height}_"
                       f"{args.bit_depth}bit.raw")
    info = write_sample(out, data, meta, args.packing, args.header,
                        args.stride_pad, args.endian)
    print(f"已生成 {out}")
    print(f"  场景/布局: {args.scene} / {info['spec']}  帧数={info['frames']}  "
          f"大小={info['bytes']:,}B")
    print(f"  注入缺陷 {len(meta['injected'])} 处:")
    for item in meta["injected"]:
        print(f"    - {item}")
    print("  打开提示：File → Open RAW，若尺寸不自动识别可点『按文件大小推断』")
    if args.meta:
        with open(args.meta, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
        print(f"  缺陷清单已写入 {args.meta}")


if __name__ == "__main__":
    main()
