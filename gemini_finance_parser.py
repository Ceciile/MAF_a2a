#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最简脚本：用 Gemini Flash 解析 PDF/图片中的财务数据

用法：
  export GEMINI_API_KEY="你的key"
  python gemini_finance_parser.py <pdf或图片路径>

支持的文件类型：
  - PDF: .pdf
  - 图片: .png, .jpg, .jpeg, .gif, .webp
"""

import os
import sys
import base64
from pathlib import Path

from google import genai
from google.genai import types


# ===== 配置 =====
MODEL_NAME = "gemini-3.5-flash"  # 可换成 gemini-2.5-flash / gemini-2.0-flash
PROMPT = "进行财务数据项解析"


def get_mime_type(file_path: str) -> str:
    """根据文件后缀判断 MIME 类型"""
    ext = Path(file_path).suffix.lower()
    mime_map = {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    if ext not in mime_map:
        raise ValueError(f"不支持的文件类型: {ext}，请使用 pdf/png/jpg/jpeg/gif/webp")
    return mime_map[ext]


def parse_file(file_path: str) -> str:
    """读取文件并调用 Gemini 解析"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("请先设置环境变量 GEMINI_API_KEY")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")

    mime_type = get_mime_type(file_path)

    # 读取文件并转 base64
    with open(file_path, "rb") as f:
        file_bytes = f.read()

    # 初始化 Gemini 客户端
    client = genai.Client(api_key=api_key)

    # 构造请求：提示词 + 文件
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            PROMPT,
            types.Part(
                inline_data=types.Blob(
                    mime_type=mime_type,
                    data=file_bytes,
                )
            ),
        ],
    )

    return response.text


def main():
    if len(sys.argv) != 2:
        print("用法: python gemini_finance_parser.py <pdf或图片路径>")
        sys.exit(1)

    file_path = sys.argv[1]

    print(f"🤖 使用模型: {MODEL_NAME}")
    print(f"📄 解析文件: {file_path}")
    print(f"💬 提示词: {PROMPT}")
    print("-" * 50)

    try:
        result = parse_file(file_path)
        print(result)
    except Exception as e:
        print(f"❌ 出错了: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
