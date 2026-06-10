#
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024-2026 The XTC Project Authors
#

# List of classical problem sizes.

ddsizes_Yolo: dict[str, dict[str, int]] = dict()
ddsizes_Yolo["Yolo9000_00"] = {
    "n": 1,
    "f": 32,
    "c": 3,
    "y": 544,
    "x": 544,
    "h": 3,
    "w": 3,
    "strx": 1,
    "stry": 1,
}
ddsizes_Yolo["Yolo9000_02"] = {
    "n": 1,
    "f": 64,
    "c": 32,
    "y": 272,
    "x": 272,
    "h": 3,
    "w": 3,
    "strx": 1,
    "stry": 1,
}
ddsizes_Yolo["Yolo9000_04"] = {
    "n": 1,
    "f": 128,
    "c": 64,
    "y": 136,
    "x": 136,
    "h": 3,
    "w": 3,
    "strx": 1,
    "stry": 1,
}
ddsizes_Yolo["Yolo9000_05"] = {
    "n": 1,
    "f": 64,
    "c": 128,
    "y": 136,
    "x": 136,
    "h": 1,
    "w": 1,
    "strx": 1,
    "stry": 1,
}
ddsizes_Yolo["Yolo9000_08"] = {
    "n": 1,
    "f": 256,
    "c": 128,
    "y": 68,
    "x": 68,
    "h": 3,
    "w": 3,
    "strx": 1,
    "stry": 1,
}
ddsizes_Yolo["Yolo9000_09"] = {
    "n": 1,
    "f": 128,
    "c": 256,
    "y": 68,
    "x": 68,
    "h": 1,
    "w": 1,
    "strx": 1,
    "stry": 1,
}
ddsizes_Yolo["Yolo9000_12"] = {
    "n": 1,
    "f": 512,
    "c": 256,
    "y": 34,
    "x": 34,
    "h": 3,
    "w": 3,
    "strx": 1,
    "stry": 1,
}
ddsizes_Yolo["Yolo9000_13"] = {
    "n": 1,
    "f": 256,
    "c": 512,
    "y": 34,
    "x": 34,
    "h": 1,
    "w": 1,
    "strx": 1,
    "stry": 1,
}
ddsizes_Yolo["Yolo9000_18"] = {
    "n": 1,
    "f": 1024,
    "c": 512,
    "y": 17,
    "x": 17,
    "h": 3,
    "w": 3,
    "strx": 1,
    "stry": 1,
}
ddsizes_Yolo["Yolo9000_19"] = {
    "n": 1,
    "f": 512,
    "c": 1024,
    "y": 17,
    "x": 17,
    "h": 1,
    "w": 1,
    "strx": 1,
    "stry": 1,
}
ddsizes_Yolo["Yolo9000_23"] = {
    "n": 1,
    "f": 28272,
    "c": 1024,
    "y": 17,
    "x": 17,
    "h": 1,
    "w": 1,
    "strx": 1,
    "stry": 1,
}
# Note for Yolo9000_23: non padded size for "f" is 28269

# Warning: MobilNet is not a classical 2D convolution (it is depthwise).
# We can still use its sizes for a proof of concept.
ddsizes_MobilNet: dict[str, dict[str, int]] = dict()
ddsizes_MobilNet["MobilNet_01"] = {
    "n": 1,
    "f": 32,
    "c": 32,
    "y": 112,
    "x": 112,
    "h": 3,
    "w": 3,
    "strx": 1,
    "stry": 1,
}
ddsizes_MobilNet["MobilNet_02"] = {
    "n": 1,
    "f": 64,
    "c": 64,
    "y": 112,
    "x": 112,
    "h": 3,
    "w": 3,
    "strx": 2,
    "stry": 2,
}
ddsizes_MobilNet["MobilNet_03"] = {
    "n": 1,
    "f": 128,
    "c": 128,
    "y": 56,
    "x": 56,
    "h": 3,
    "w": 3,
    "strx": 1,
    "stry": 1,
}
ddsizes_MobilNet["MobilNet_04"] = {
    "n": 1,
    "f": 128,
    "c": 128,
    "y": 56,
    "x": 56,
    "h": 3,
    "w": 3,
    "strx": 2,
    "stry": 2,
}
ddsizes_MobilNet["MobilNet_05"] = {
    "n": 1,
    "f": 256,
    "c": 256,
    "y": 28,
    "x": 28,
    "h": 3,
    "w": 3,
    "strx": 1,
    "stry": 1,
}
ddsizes_MobilNet["MobilNet_06"] = {
    "n": 1,
    "f": 256,
    "c": 256,
    "y": 28,
    "x": 28,
    "h": 3,
    "w": 3,
    "strx": 2,
    "stry": 2,
}
ddsizes_MobilNet["MobilNet_07"] = {
    "n": 1,
    "f": 512,
    "c": 512,
    "y": 14,
    "x": 14,
    "h": 3,
    "w": 3,
    "strx": 1,
    "stry": 1,
}
ddsizes_MobilNet["MobilNet_08"] = {
    "n": 1,
    "f": 512,
    "c": 512,
    "y": 14,
    "x": 14,
    "h": 3,
    "w": 3,
    "strx": 2,
    "stry": 2,
}
ddsizes_MobilNet["MobilNet_09"] = {
    "n": 1,
    "f": 1024,
    "c": 1024,
    "y": 7,
    "x": 7,
    "h": 3,
    "w": 3,
    "strx": 1,
    "stry": 1,
}


ddsizes_RN18: dict[str, dict[str, int]] = dict()
ddsizes_RN18["ResNet18_01"] = {
    "n": 1,
    "f": 64,
    "c": 3,
    "y": 224,
    "x": 224,
    "h": 7,
    "w": 7,
    "strx": 2,
    "stry": 2,
}
ddsizes_RN18["ResNet18_02"] = {
    "n": 1,
    "f": 64,
    "c": 64,
    "y": 56,
    "x": 56,
    "h": 3,
    "w": 3,
    "strx": 1,
    "stry": 1,
}
ddsizes_RN18["ResNet18_03"] = {
    "n": 1,
    "f": 64,
    "c": 64,
    "y": 56,
    "x": 56,
    "h": 1,
    "w": 1,
    "strx": 1,
    "stry": 1,
}
ddsizes_RN18["ResNet18_04"] = {
    "n": 1,
    "f": 128,
    "c": 64,
    "y": 56,
    "x": 56,
    "h": 3,
    "w": 3,
    "strx": 2,
    "stry": 2,
}
ddsizes_RN18["ResNet18_05"] = {
    "n": 1,
    "f": 128,
    "c": 64,
    "y": 56,
    "x": 56,
    "h": 1,
    "w": 1,
    "strx": 2,
    "stry": 2,
}
ddsizes_RN18["ResNet18_06"] = {
    "n": 1,
    "f": 128,
    "c": 128,
    "y": 28,
    "x": 28,
    "h": 3,
    "w": 3,
    "strx": 1,
    "stry": 1,
}
ddsizes_RN18["ResNet18_07"] = {
    "n": 1,
    "f": 256,
    "c": 128,
    "y": 28,
    "x": 28,
    "h": 3,
    "w": 3,
    "strx": 2,
    "stry": 2,
}
ddsizes_RN18["ResNet18_08"] = {
    "n": 1,
    "f": 256,
    "c": 128,
    "y": 28,
    "x": 28,
    "h": 3,
    "w": 3,
    "strx": 1,
    "stry": 1,
}
ddsizes_RN18["ResNet18_09"] = {
    "n": 1,
    "f": 256,
    "c": 256,
    "y": 14,
    "x": 14,
    "h": 3,
    "w": 3,
    "strx": 1,
    "stry": 1,
}
ddsizes_RN18["ResNet18_10"] = {
    "n": 1,
    "f": 512,
    "c": 512,
    "y": 14,
    "x": 14,
    "h": 3,
    "w": 3,
    "strx": 2,
    "stry": 2,
}
ddsizes_RN18["ResNet18_11"] = {
    "n": 1,
    "f": 512,
    "c": 256,
    "y": 14,
    "x": 14,
    "h": 1,
    "w": 1,
    "strx": 2,
    "stry": 2,
}
ddsizes_RN18["ResNet18_12"] = {
    "n": 1,
    "f": 512,
    "c": 512,
    "y": 7,
    "x": 7,
    "h": 3,
    "w": 3,
    "strx": 1,
    "stry": 1,
}


# Transform the x/y/h/w entries of dsizes into h/w/r/s
def subst_dimname_xyhw_to_hwrs_conv2D_dsizes(dsizes: dict[str, int]) -> dict[str, int]:
    dsizes_subst = dict()
    for dim_name in dsizes.keys():
        # Copy of the dims
        if dim_name not in ["x", "y", "h", "w"]:
            dsizes_subst[dim_name] = dsizes[dim_name]

    dsizes_subst["h"] = dsizes["x"]
    dsizes_subst["w"] = dsizes["y"]
    dsizes_subst["r"] = dsizes["h"]
    dsizes_subst["s"] = dsizes["w"]

    return dsizes_subst


# Matmult sizes / i=y | j=f | k=c
ddsizes_matmul: dict[str, dict[str, int]] = dict()
ddsizes_matmul["Llama3AttMm"] = {"i": 8192, "j": 8192, "k": 4096}

# Note: j is padded to be a multiple of 16, due to vectorization for AVX512
# in order to have microkernel
ddsizes_matmul["Polybench_medium"] = {"i": 200, "j": 224, "k": 240}  # j = 200
ddsizes_matmul["Polybench_large"] = {"i": 1000, "j": 1104, "k": 1200}  # j = 1100
ddsizes_matmul["Polybench_extralarge"] = {"i": 2000, "j": 2304, "k": 2600}  # j = 2304

ddsizes_matmul["Llama3_In"] = {"i": 4096, "j": 4096, "k": 128}
ddsizes_matmul["Llama3_QK"] = {"i": 128, "j": 4096, "k": 8192}
ddsizes_matmul["Llama3_QKV"] = {"i": 128, "j": 8192, "k": 4096}
ddsizes_matmul["Llama3_FC"] = {"i": 4096, "j": 4096, "k": 4096}

ddsizes_matmul["Gemma27B_In"] = {"i": 4608, "j": 4096, "k": 256}
ddsizes_matmul["Gemma27B_QK"] = {"i": 256, "j": 4608, "k": 8192}
ddsizes_matmul["Gemma27B_QKV"] = {"i": 256, "j": 8192, "k": 4608}
ddsizes_matmul["Gemma27B_FC"] = {"i": 4608, "j": 36864, "k": 4608}

ddsizes_matmul["Gemma9B_In"] = {"i": 3584, "j": 4096, "k": 256}
ddsizes_matmul["Gemma9B_QK"] = {"i": 256, "j": 3584, "k": 8192}
ddsizes_matmul["Gemma9B_QKV"] = {"i": 256, "j": 8192, "k": 3584}
ddsizes_matmul["Gemma9B_FC"] = {"i": 3584, "j": 14336, "k": 3584}

ddsizes_matmul["Gemma7B_In"] = {"i": 3072, "j": 4096, "k": 256}
ddsizes_matmul["Gemma7B_QK"] = {"i": 256, "j": 3072, "k": 8192}
ddsizes_matmul["Gemma7B_QKV"] = {"i": 256, "j": 8192, "k": 3072}
ddsizes_matmul["Gemma7B_FC"] = {"i": 3072, "j": 24576, "k": 3072}

ddsizes_matmul["Gemma2B_In"] = {"i": 2048, "j": 4096, "k": 256}
ddsizes_matmul["Gemma2B_QK"] = {"i": 256, "j": 2048, "k": 8192}
ddsizes_matmul["Gemma2B_QKV"] = {"i": 256, "j": 8192, "k": 2048}
ddsizes_matmul["Gemma2B_FC"] = {"i": 2048, "j": 16384, "k": 2048}
