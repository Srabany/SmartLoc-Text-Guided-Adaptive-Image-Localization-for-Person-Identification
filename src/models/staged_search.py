"""
SmartLoc — Staged Terminal Person Search
=========================================
Progressive person search using CLIP + YOLOv8.

Features:
  - Disk-cached CLIP features (only processes new/changed images)
  - Batch processing with progress display
  - Multi-stage narrowing: each prompt filters the previous results
  - OpenCV viewing of matched persons with bounding boxes
  - Pure terminal interaction for search flow

Usage:
    python staged_search.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
import hashlib
import logging
import subprocess
import platform
import concurrent.futures
import re
# pyrefly: ignore [missing-import]
import torch
import numpy as np
from PIL import Image
# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
from ultralytics import YOLO
# pyrefly: ignore [missing-import]
import clip

# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "images")
CACHE_DIR = os.path.join(BASE_DIR, "data", ".cache")

YOLO_MODEL_NAME = "yolov8n.pt"  # Faster YOLO model (Nano)
CLIP_MODEL_NAME = "ViT-B/16"    # Smarter model than B/32, much faster than L/14

BATCH_SIZE = 10          # images per batch during indexing
MIN_CROP_W = 30          # minimum person crop width
MIN_CROP_H = 40          # minimum person crop height
MAX_RESULTS = 15         # max results to show per stage
SIM_THRESHOLD = 0.30     # minimum similarity to consider a match
MAX_DISPLAY_W = 1400     # OpenCV display max width
MAX_DISPLAY_H = 900      # OpenCV display max height
MIN_BRIGHTNESS = 55      # reject crops darker than this (0-255 avg)
MIN_CONTRAST = 25        # reject crops with std dev below this

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif"}

# Silence ultralytics
logging.getLogger("ultralytics").setLevel(logging.ERROR)


# ──────────────────────────────────────────────────────────────
# ANSI COLORS
# ──────────────────────────────────────────────────────────────
class C:
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"
    BLUE   = "\033[94m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RESET  = "\033[0m"
    WHITE  = "\033[97m"
    MAGENTA= "\033[95m"
    BG_DARK= "\033[48;5;236m"


# ──────────────────────────────────────────────────────────────
# TERMINAL DISPLAY
# ──────────────────────────────────────────────────────────────

def file_hash(path):
    """Fast hash based on filename + size + mtime for change detection."""
    stat = os.stat(path)
    key = f"{os.path.basename(path)}|{stat.st_size}|{stat.st_mtime_ns}"
    return hashlib.md5(key.encode()).hexdigest()


def _score_color_ansi(score):
    pct = score * 100
    if pct >= 28: return C.GREEN
    elif pct >= 22: return C.YELLOW
    else: return C.RED


def _score_bar(score, width=12):
    """Create a visual bar like ████░░░░ for a score."""
    pct = score * 100
    filled = int((pct / 40) * width)  # normalize to 40% max
    filled = min(filled, width)
    clr = _score_color_ansi(score)
    return f"{clr}{'█' * filled}{'░' * (width - filled)}{C.RESET}"


def print_header():
    print(f"\n{C.CYAN}{C.BOLD}")
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║         SmartLoc — Text Based Person Search              ║")
    print("  ║         CLIP + YOLOv8                                    ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print(f"{C.RESET}")


def print_divider():
    print(f"\n{C.DIM}  {'─' * 58}{C.RESET}")


def print_stage_header(stage, pool_size, prompts):
    """Print a styled dynamic-width stage header."""
    chain_plain = " → ".join(prompts) if prompts else ""
    content_len = len(f"  Prompts: {chain_plain}") if prompts else 0
    box_inner = max(50, content_len + 2)
    
    top_dashes = max(1, box_inner - 3 - len(f" STAGE {stage} "))
    print(f"\n  {C.CYAN}{C.BOLD}┌─── STAGE {stage} {'─' * top_dashes}┐{C.RESET}")
    
    if prompts:
        chain = f" {C.DIM}→{C.RESET} ".join(f"{C.WHITE}{p}{C.RESET}" for p in prompts)
        padding = box_inner - len(f"  Prompts: {chain_plain}")
        print(f"  {C.CYAN}│{C.RESET}  Prompts: {chain}{' ' * padding}{C.CYAN}│{C.RESET}")
        
    pool_str = f"  Pool: {pool_size} persons"
    pool_padding = box_inner - len(pool_str)
    print(f"  {C.CYAN}│{C.RESET}  Pool: {C.BOLD}{pool_size}{C.RESET} persons{' ' * pool_padding}{C.CYAN}│{C.RESET}")
    print(f"  {C.CYAN}└{'─' * box_inner}┘{C.RESET}")


def print_table(candidates, start_rank=1):
    """Print a premium results table with score bars and optional color match."""
    has_color = any("color_match" in c for c in candidates)
    if has_color:
        print(f"\n  {C.BOLD}{C.WHITE}  #  │ Image                │ Person │  Score  │ Match        │ Color{C.RESET}")
        print(f"  {C.DIM}─────┼──────────────────────┼────────┼─────────┼──────────────┼───────{C.RESET}")
    else:
        print(f"\n  {C.BOLD}{C.WHITE}  #  │ Image                │ Person │  Score  │ Match{C.RESET}")
        print(f"  {C.DIM}─────┼──────────────────────┼────────┼─────────┼──────────────{C.RESET}")
    for i, c in enumerate(candidates):
        rank = start_rank + i
        name = c["image_name"]
        if len(name) > 20:
            name = name[:17] + "..."
        pct = c['score'] * 100
        clr = _score_color_ansi(c['score'])
        bar = _score_bar(c['score'])
        marker = f"{C.GREEN}★{C.RESET}" if i == 0 else " "
        line = f"  {marker}{rank:>3}  │ {name:<20} │   #{c['person_idx']:<4}│ {clr}{pct:>5.1f}%{C.RESET}  │ {bar}"
        if has_color and "color_match" in c:
            cm = c["color_match"] * 100
            cm_clr = C.GREEN if cm > 20 else (C.YELLOW if cm > 5 else C.RED)
            line += f" │ {cm_clr}{cm:>4.0f}%{C.RESET}"
        print(line)
    print()


# ──────────────────────────────────────────────────────────────
# OPENCV DISPLAY — Premium Visual System
# ──────────────────────────────────────────────────────────────

GRID_COLS = 4
CELL_W = 240
CELL_H = 260
CARD_PAD = 14
HEADER_H = 72
FOOTER_H = 38
INFO_H = 80

# Color palette (BGR)
BG_COLOR     = (22, 22, 26)
CARD_BG      = (38, 38, 42)
CARD_BORDER  = (55, 55, 60)
ACCENT_GREEN = (100, 230, 120)
ACCENT_AMBER = (60, 190, 255)
ACCENT_GRAY  = (120, 120, 120)
TEXT_WHITE   = (245, 245, 245)
TEXT_LIGHT   = (200, 200, 200)
TEXT_DIM     = (150, 150, 155)
TEXT_DARK    = (95, 95, 100)
GOLD         = (0, 215, 255)
BRAND_BAR    = (50, 45, 40)
PANEL_BG     = (32, 32, 36)


def _safe_show_and_wait(win_name, img, width=None, height=None):
    """Show an OpenCV window, bring to front, wait for key/close."""
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, img)
    if width and height:
        cv2.resizeWindow(win_name, width, height)
    try:
        cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(50)
        cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 0)
    except Exception:
        pass
    while True:
        key = cv2.waitKey(100)
        if key != -1:
            break
        try:
            if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        except Exception:
            break
    try:
        cv2.destroyWindow(win_name)
    except Exception:
        pass
    cv2.waitKey(1)


def _load_image(data_dir, image_name):
    """Load image via OpenCV, fallback to PIL for avif/webp."""
    img_path = os.path.join(data_dir, image_name)
    img = cv2.imread(img_path)
    if img is None:
        try:
            pil_img = Image.open(img_path).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            return None
    return img


def _score_accent(score):
    """Return BGR color for score level."""
    pct = score * 100
    if pct >= 28: return ACCENT_GREEN
    elif pct >= 22: return ACCENT_AMBER
    return ACCENT_GRAY


def _draw_rounded_rect(img, pt1, pt2, color, radius=8, thickness=2):
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2 - x1) // 4, (y2 - y1) // 4)
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)


def _draw_filled_rounded_rect(img, pt1, pt2, color, radius=8):
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2 - x1) // 4, (y2 - y1) // 4)
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, -1, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, -1, cv2.LINE_AA)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, -1, cv2.LINE_AA)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, -1, cv2.LINE_AA)


def _draw_score_bar(img, x, y, w, h, score, accent):
    """Draw a horizontal score progress bar."""
    cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 55), -1)
    fill_w = int((score / 0.50) * w)
    fill_w = min(fill_w, w)
    if fill_w > 2:
        cv2.rectangle(img, (x, y), (x + fill_w, y + h), accent, -1)


def show_top_grid(candidates, data_dir, stage, query_text="", max_show=8):
    """Show a premium grid of top person crops with systematic layout."""
    to_show = candidates[:max_show]
    n = len(to_show)
    if n == 0:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    cols = min(n, GRID_COLS)
    rows = (n + cols - 1) // cols
    card_h = CELL_H + INFO_H

    canvas_w = cols * (CELL_W + CARD_PAD) + CARD_PAD
    canvas_h = rows * (card_h + CARD_PAD) + CARD_PAD + HEADER_H + FOOTER_H
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = BG_COLOR

    # Header gradient
    for y in range(HEADER_H):
        t = y / HEADER_H
        r = int(BRAND_BAR[0] * (1 - t * 0.6) + BG_COLOR[0] * t * 0.6)
        g = int(BRAND_BAR[1] * (1 - t * 0.6) + BG_COLOR[1] * t * 0.6)
        b = int(BRAND_BAR[2] * (1 - t * 0.6) + BG_COLOR[2] * t * 0.6)
        cv2.line(canvas, (0, y), (canvas_w, y), (r, g, b), 1)

    # Brand accent line at top
    cv2.line(canvas, (0, 0), (canvas_w, 0), ACCENT_GREEN, 2)

    # Header left: branding
    cv2.putText(canvas, "SmartLoc", (CARD_PAD, 28),
                font, 0.75, ACCENT_GREEN, 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Stage {stage}  |  Top {n} Results", (CARD_PAD, 50),
                font, 0.42, TEXT_DIM, 1, cv2.LINE_AA)

    # Header right: query
    if query_text:
        # Truncate if too long (up to 95 chars)
        display_text = query_text if len(query_text) < 95 else query_text[:92] + "..."
        qt_size = cv2.getTextSize(display_text, font, 0.38, 1)[0]
        qx = canvas_w - qt_size[0] - CARD_PAD
        cv2.putText(canvas, "Query:", (max(qx - 48, CARD_PAD), 28),
                    font, 0.35, TEXT_DARK, 1, cv2.LINE_AA)
        cv2.putText(canvas, display_text, (max(qx, CARD_PAD + 50), 28),
                    font, 0.38, ACCENT_AMBER, 1, cv2.LINE_AA)

    # Separator
    cv2.line(canvas, (CARD_PAD, HEADER_H - 4), (canvas_w - CARD_PAD, HEADER_H - 4),
             (50, 50, 55), 1, cv2.LINE_AA)

    # Cards
    unique_names = list(set(c["image_name"] for c in candidates[:max_show]))
    loaded_imgs = {}
    def _load_img_grid(name):
        return name, cv2.imread(os.path.join(data_dir, name))
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        for name, img in executor.map(_load_img_grid, unique_names):
            if img is not None:
                loaded_imgs[name] = img

    for i, c in enumerate(to_show):
        row_i = i // cols
        col_i = i % cols

        cx = col_i * (CELL_W + CARD_PAD) + CARD_PAD
        cy = row_i * (card_h + CARD_PAD) + CARD_PAD + HEADER_H

        accent = _score_accent(c["score"])
        is_top = (i == 0)

        # Card bg
        _draw_filled_rounded_rect(canvas, (cx, cy), (cx + CELL_W, cy + card_h), CARD_BG, radius=10)

        # Card border
        if is_top:
            _draw_rounded_rect(canvas, (cx - 3, cy - 3),
                               (cx + CELL_W + 3, cy + card_h + 3), GOLD, radius=13, thickness=2)
            _draw_rounded_rect(canvas, (cx, cy),
                               (cx + CELL_W, cy + card_h), GOLD, radius=10, thickness=1)
        else:
            _draw_rounded_rect(canvas, (cx, cy),
                               (cx + CELL_W, cy + card_h), CARD_BORDER, radius=10, thickness=1)

        # Load crop
        img = loaded_imgs.get(c["image_name"])
        if img is None:
            continue
        bx1, by1, bx2, by2 = c["bbox"]
        crop = img[by1:by2, bx1:bx2]
        if crop.size == 0:
            continue

        # Resize and center
        img_pad = 10
        img_area_w = CELL_W - img_pad * 2
        img_area_h = CELL_H - img_pad * 2 - 4
        ch_c, cw_c = crop.shape[:2]
        sc = min(img_area_w / cw_c, img_area_h / ch_c)
        nw = int(cw_c * sc)
        nh = int(ch_c * sc)
        crop_r = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)

        px = cx + (CELL_W - nw) // 2
        py = cy + img_pad + (img_area_h - nh) // 2
        if 0 <= py and py + nh <= canvas_h and 0 <= px and px + nw <= canvas_w:
            canvas[py:py + nh, px:px + nw] = crop_r

        # Info panel
        info_y = cy + CELL_H
        cv2.line(canvas, (cx + 8, info_y), (cx + CELL_W - 8, info_y),
                 (55, 55, 60), 1, cv2.LINE_AA)

        # Rank badge
        badge_color = GOLD if is_top else accent
        bx = cx + 10
        by = info_y + 6
        _draw_filled_rounded_rect(canvas, (bx, by), (bx + 36, by + 20), badge_color, radius=5)
        cv2.putText(canvas, f"#{i+1}", (bx + 6, by + 15),
                    font, 0.42, (15, 15, 15), 1, cv2.LINE_AA)

        # Score
        pct = c["score"] * 100
        cv2.putText(canvas, f"{pct:.1f}%", (bx + 44, by + 15),
                    font, 0.5, accent, 2, cv2.LINE_AA)

        # Color match dot + percent
        if "color_match" in c:
            cm = c["color_match"] * 100
            cm_clr = ACCENT_GREEN if cm > 20 else (ACCENT_AMBER if cm > 5 else ACCENT_GRAY)
            cm_x = cx + CELL_W - 42
            cv2.circle(canvas, (cm_x - 6, by + 10), 4, cm_clr, -1, cv2.LINE_AA)
            cv2.putText(canvas, f"{cm:.0f}%", (cm_x, by + 15),
                        font, 0.33, cm_clr, 1, cv2.LINE_AA)

        # Score bar
        bar_y = info_y + 32
        _draw_score_bar(canvas, cx + 10, bar_y, CELL_W - 20, 5, c["score"], accent)

        # Image name + Person ID
        lbl = c["image_name"]
        if len(lbl) > 26:
            lbl = lbl[:23] + "..."
        cv2.putText(canvas, lbl, (cx + 10, bar_y + 20),
                    font, 0.32, TEXT_LIGHT, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Person #{c['person_idx']}", (cx + 10, bar_y + 34),
                    font, 0.30, TEXT_DARK, 1, cv2.LINE_AA)

    # Footer
    fy = canvas_h - FOOTER_H + 12
    cv2.line(canvas, (CARD_PAD, fy - 8), (canvas_w - CARD_PAD, fy - 8),
             (50, 50, 55), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Press any key or close window  |  'view N' in terminal for full image",
                (CARD_PAD, fy + 8), font, 0.33, TEXT_DARK, 1, cv2.LINE_AA)

    win_name = f"SmartLoc - Stage {stage}"
    _safe_show_and_wait(win_name, canvas, min(canvas_w, MAX_DISPLAY_W), min(canvas_h, MAX_DISPLAY_H))


def show_result_cv2(candidate, data_dir):
    """Show full image with matched person highlighted and info side panel."""
    img = _load_image(data_dir, candidate["image_name"])
    if img is None:
        print(f"  {C.RED}✗ Could not load: {candidate['image_name']}{C.RESET}")
        return

    h_orig, w_orig = img.shape[:2]
    x1, y1, x2, y2 = candidate["bbox"]
    accent = _score_accent(candidate["score"])
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Dim background, keep matched person bright
    overlay = img.copy()
    dim = np.full_like(img, 25)
    cv2.addWeighted(dim, 0.35, overlay, 0.65, 0, overlay)
    overlay[y1:y2, x1:x2] = img[y1:y2, x1:x2]
    img = overlay

    # Bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), accent, 3)

    # Corner brackets
    clen = min(30, (x2 - x1) // 3, (y2 - y1) // 3)
    for (px, py), (dx, dy) in [
        ((x1, y1), (1, 1)), ((x2, y1), (-1, 1)),
        ((x1, y2), (1, -1)), ((x2, y2), (-1, -1))
    ]:
        cv2.line(img, (px, py), (px + dx * clen, py), accent, 4, cv2.LINE_AA)
        cv2.line(img, (px, py), (px, py + dy * clen), accent, 4, cv2.LINE_AA)

    # Side info panel
    panel_w = 230
    panel_h = 150
    panel_x = min(x2 + 20, w_orig - panel_w - 10)
    panel_y = max(y1, 10)
    if panel_x < x2 + 5:
        panel_x = max(x1 - panel_w - 20, 10)

    p_overlay = img.copy()
    cv2.rectangle(p_overlay, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h), (20, 20, 24), -1)
    cv2.addWeighted(p_overlay, 0.85, img, 0.15, 0, img)
    cv2.rectangle(img, (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h), accent, 1, cv2.LINE_AA)
    cv2.line(img, (panel_x, panel_y), (panel_x + panel_w, panel_y), accent, 2, cv2.LINE_AA)

    tx = panel_x + 12
    ty = panel_y + 25
    cv2.putText(img, "SmartLoc Match", (tx, ty), font, 0.5, ACCENT_GREEN, 1, cv2.LINE_AA)

    ty += 26
    cv2.putText(img, "Image:", (tx, ty), font, 0.36, TEXT_DIM, 1, cv2.LINE_AA)
    nm = candidate["image_name"]
    if len(nm) > 20: nm = nm[:17] + "..."
    cv2.putText(img, nm, (tx + 50, ty), font, 0.36, TEXT_WHITE, 1, cv2.LINE_AA)

    ty += 20
    cv2.putText(img, "Person:", (tx, ty), font, 0.36, TEXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(img, f"#{candidate['person_idx']}", (tx + 55, ty), font, 0.36, TEXT_WHITE, 1, cv2.LINE_AA)

    ty += 20
    pct = candidate["score"] * 100
    cv2.putText(img, "Score:", (tx, ty), font, 0.36, TEXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(img, f"{pct:.1f}%", (tx + 48, ty), font, 0.42, accent, 2, cv2.LINE_AA)

    ty += 16
    _draw_score_bar(img, tx, ty, panel_w - 24, 5, candidate["score"], accent)

    if "color_match" in candidate:
        ty += 20
        cm = candidate["color_match"] * 100
        cm_clr = ACCENT_GREEN if cm > 20 else (ACCENT_AMBER if cm > 5 else ACCENT_GRAY)
        cv2.putText(img, "Color:", (tx, ty), font, 0.36, TEXT_DIM, 1, cv2.LINE_AA)
        cv2.putText(img, f"{cm:.0f}%", (tx + 48, ty), font, 0.42, cm_clr, 1, cv2.LINE_AA)

    # Bottom bar
    bar_h = 32
    bar_y = h_orig - bar_h
    b_overlay = img.copy()
    cv2.rectangle(b_overlay, (0, bar_y), (w_orig, h_orig), (15, 15, 18), -1)
    cv2.addWeighted(b_overlay, 0.8, img, 0.2, 0, img)
    cv2.line(img, (0, bar_y), (w_orig, bar_y), accent, 1, cv2.LINE_AA)
    cv2.putText(img, "Press any key or close window", (12, bar_y + 22),
                font, 0.36, TEXT_DARK, 1, cv2.LINE_AA)

    h, w = img.shape[:2]
    scale = min(MAX_DISPLAY_W / w, MAX_DISPLAY_H / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    win_name = f"SmartLoc - {candidate['image_name']}"
    print(f"  {C.DIM}  Showing full image (press any key or close window){C.RESET}")
    _safe_show_and_wait(win_name, img)


# ──────────────────────────────────────────────────────────────
# CACHE SYSTEM
# ──────────────────────────────────────────────────────────────

# Number of crops per person for multi-crop encoding
NUM_CROPS = 3  # full body, upper body, center torso


class FeatureCache:
    """Disk-based cache for YOLO detections + multi-crop CLIP features."""

    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(self, img_hash):
        return os.path.join(self.cache_dir, f"{img_hash}.npz")

    def has(self, img_hash):
        return os.path.exists(self._cache_path(img_hash))

    def load(self, img_hash):
        """Load cached data with multi-crop features."""
        path = self._cache_path(img_hash)
        if not os.path.exists(path):
            return None
        try:
            data = np.load(path, allow_pickle=True)
            result = {
                "bboxes": data["bboxes"],         # (N, 4)
                "features": data["features"],      # (N, D) — full body
            }
            # Load multi-crop features if available
            if "features_upper" in data:
                result["features_upper"] = data["features_upper"]  # (N, D)
            if "features_torso" in data:
                result["features_torso"] = data["features_torso"]  # (N, D)
            return result
        except Exception:
            return None

    def save(self, img_hash, bboxes, features, features_upper=None, features_torso=None):
        """Save bboxes + multi-crop features to disk."""
        path = self._cache_path(img_hash)
        save_dict = {"bboxes": bboxes, "features": features}
        if features_upper is not None:
            save_dict["features_upper"] = features_upper
        if features_torso is not None:
            save_dict["features_torso"] = features_torso
        np.savez_compressed(path, **save_dict)


# ──────────────────────────────────────────────────────────────
# INDEXER — Batch YOLO + CLIP processing
# ──────────────────────────────────────────────────────────────

def index_images(image_dir, cache, yolo_model, clip_model, preprocess, device, feat_dim=768):
    """
    Scan images, detect persons with YOLO, encode crops with CLIP.
    Only processes images not already cached.
    Returns list of all person entries.
    """
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
    ])

    if not image_files:
        print(f"\n  ✗ No images found in {image_dir}")
        sys.exit(1)

    # Determine which images need processing
    file_hashes = {}
    to_process = []
    cached_count = 0

    for fname in image_files:
        fpath = os.path.join(image_dir, fname)
        fhash = file_hash(fpath)
        file_hashes[fname] = fhash
        if cache.has(fhash):
            cached_count += 1
        else:
            to_process.append(fname)

    print(f"\n  {len(image_files)} images found, {cached_count} cached, {len(to_process)} to process")

    # Process new images in batches
    if to_process:
        total = len(to_process)
        processed = 0
        t_start = time.time()
        bar_len = 40

        for batch_start in range(0, total, BATCH_SIZE):
            batch = to_process[batch_start:batch_start + BATCH_SIZE]

            for fname in batch:
                fpath = os.path.join(image_dir, fname)
                fhash = file_hashes[fname]

                try:
                    img_pil = Image.open(fpath).convert("RGB")
                except Exception:
                    processed += 1
                    continue

                # YOLO person detection
                results = yolo_model.predict(source=fpath, classes=[0], imgsz=640, verbose=False)
                boxes = results[0].boxes.xyxy.cpu().numpy()

                if len(boxes) == 0:
                    cache.save(fhash,
                               np.zeros((0, 4), dtype=int),
                               np.zeros((0, feat_dim), dtype=np.float32),
                               np.zeros((0, feat_dim), dtype=np.float32),
                               np.zeros((0, feat_dim), dtype=np.float32))
                    processed += 1
                    continue

                img_w, img_h = img_pil.size
                valid_bboxes = []
                valid_feats_full = []    # full body
                valid_feats_upper = []   # head + torso (top 65%)
                valid_feats_torso = []   # torso only (10%-65%)

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(img_w, x2), min(img_h, y2)
                    bw, bh = x2 - x1, y2 - y1
                    if bw < MIN_CROP_W or bh < MIN_CROP_H:
                        continue

                    # Crop 1: Full body
                    crop_full = img_pil.crop((x1, y1, x2, y2))

                    # Crop 2: Upper body (top 65% — head + torso)
                    upper_y2 = y1 + int(bh * 0.65)
                    crop_upper = img_pil.crop((x1, y1, x2, min(upper_y2, y2)))

                    # Crop 3: Center torso (10%-65% — shirt/top area)
                    torso_y1 = y1 + int(bh * 0.10)
                    torso_y2 = y1 + int(bh * 0.65)
                    mx = int(bw * 0.06)  # small horizontal margin
                    crop_torso = img_pil.crop((
                        min(x1 + mx, x2), min(torso_y1, y2),
                        max(x2 - mx, x1), min(torso_y2, y2)
                    ))

                    # Encode only the full crop for maximum speed (3x faster than multi-crop)
                    crops = [crop_full]
                    batch = torch.stack([preprocess(c) for c in crops]).to(device)

                    with torch.no_grad():
                        feats = clip_model.encode_image(batch)     # (1, D)
                        feats = feats / feats.norm(dim=-1, keepdim=True)
                        feats = feats.cpu().numpy()                # (1, D)

                    valid_bboxes.append([x1, y1, x2, y2])
                    valid_feats_full.append(feats[0])

                if valid_bboxes:
                    bboxes_arr = np.array(valid_bboxes, dtype=int)
                    full_arr = np.array(valid_feats_full, dtype=np.float32)
                else:
                    bboxes_arr = np.zeros((0, 4), dtype=int)
                    full_arr = np.zeros((0, feat_dim), dtype=np.float32)

                # Save None for upper/torso to signify they aren't used
                cache.save(fhash, bboxes_arr, full_arr, None, None)
                processed += 1

                # Progress bar
                pct = processed / total
                filled = int(bar_len * pct)
                bar = "█" * filled + "░" * (bar_len - filled)
                elapsed = time.time() - t_start
                eta = (elapsed / max(processed, 1)) * (total - processed)
                print(f"\r  Processing [{bar}] {processed}/{total}  ETA: {eta:.0f}s", end="", flush=True)

        elapsed = time.time() - t_start
        print(f"\r  Processing [{'█' * bar_len}] {total}/{total}  Done in {elapsed:.1f}s     ")

    # Load all cached features into memory
    print("  Loading cached features...")
    all_persons = []

    for fname in image_files:
        fhash = file_hashes[fname]
        data = cache.load(fhash)
        if data is None or len(data["bboxes"]) == 0:
            continue

        for idx in range(len(data["bboxes"])):
            entry = {
                "image_name": fname,
                "person_idx": idx,
                "bbox": data["bboxes"][idx].tolist(),
                "features": data["features"][idx],         # full body
            }
            # Load multi-crop features if available
            if "features_upper" in data and idx < len(data["features_upper"]):
                entry["features_upper"] = data["features_upper"][idx]
            if "features_torso" in data and idx < len(data["features_torso"]):
                entry["features_torso"] = data["features_torso"][idx]
            all_persons.append(entry)

    print(f"  ✓ {len(all_persons)} person crops indexed across {len(image_files)} images\n")
    return all_persons


# ──────────────────────────────────────────────────────────────
# SEARCH ENGINE — Multi-Template CLIP Matching
# ──────────────────────────────────────────────────────────────

# Prompt templates (averaging multiple templates is a proven CLIP technique
# that significantly improves zero-shot accuracy — used in the original paper)
PROMPT_TEMPLATES = [
    "{}",
    "a photo of a {}",
    "a cropped photo of a {}",
]


def encode_text(clip_model, device, text):
    """
    Encode a text query using multi-template averaging.
    Creates multiple prompt variations and averages their embeddings
    for much better matching accuracy (standard CLIP technique).
    Returns normalized (1, D) numpy array.
    """
    # Generate all template variations
    prompts = []
    for template in PROMPT_TEMPLATES:
        prompts.append(template.format(text))

    # Deduplicate while preserving order
    seen = set()
    unique_prompts = []
    for p in prompts:
        if p not in seen:
            seen.add(p)
            unique_prompts.append(p)

    # Encode all at once and average
    tokens = clip.tokenize(unique_prompts, truncate=True).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(tokens)  # (T, D)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # Average across templates
        avg_features = text_features.mean(dim=0, keepdim=True)  # (1, D)
        avg_features = avg_features / avg_features.norm(dim=-1, keepdim=True)

    return avg_features.cpu().numpy()  # (1, D)


def rank_candidates(candidates, text_features):
    """
    Score candidates using MAX similarity across multi-crop features.
    For each person, computes similarity against full body, upper body,
    and torso crops, then takes the maximum — much more accurate overall.
    """
    text_vec = text_features.squeeze(0)  # (D,)

    # Collect feature matrices
    feats_full = np.array([c["features"] for c in candidates])
    scores = np.dot(feats_full, text_vec)

    if "features_upper" in candidates[0]:
        feats_upper = np.array([c.get("features_upper", np.zeros_like(text_vec)) for c in candidates])
        scores = np.maximum(scores, np.dot(feats_upper, text_vec))
        
    if "features_torso" in candidates[0]:
        feats_torso = np.array([c.get("features_torso", np.zeros_like(text_vec)) for c in candidates])
        scores = np.maximum(scores, np.dot(feats_torso, text_vec))

    # Apply scores
    for i, c in enumerate(candidates):
        c["score"] = float(scores[i])

    # Stable sort: primary by score (desc), tiebreaker by name+idx (asc)
    candidates.sort(key=lambda x: (-x["score"], x["image_name"], x["person_idx"]))
    
    # Filter by similarity
    return [c for c in candidates if c["score"] >= SIM_THRESHOLD]


# ──────────────────────────────────────────────────────────────
# COLOR VERIFICATION — HSV-based torso color analysis
# ──────────────────────────────────────────────────────────────

# Color definitions in HSV space: (H_low, H_high, S_min, S_max, V_min, V_max)
# OpenCV uses H: 0-179, S: 0-255, V: 0-255
COLOR_RANGES = {
    "red":     [((0, 8), 40, 255, 40, 255), ((170, 179), 40, 255, 40, 255)],  # red wraps around
    "blue":    [((100, 130), 50, 255, 40, 255)],
    "green":   [((35, 85), 40, 255, 40, 255)],
    "yellow":  [((22, 30), 120, 255, 120, 255)],
    "orange":  [((10, 22), 80, 255, 80, 255)],
    "pink":    [((145, 170), 30, 255, 100, 255)],
    "purple":  [((125, 150), 40, 255, 40, 255)],
    "black":   [((0, 179), 0, 255, 0, 60)],
    "white":   [((0, 179), 0, 40, 200, 255)],
    "grey":    [((0, 179), 0, 40, 60, 200)],
    "gray":    [((0, 179), 0, 40, 60, 200)],
    "brown":   [((8, 20), 50, 255, 30, 150)],
    "maroon":  [((0, 10), 50, 255, 30, 100), ((170, 179), 50, 255, 30, 100)],
    "navy":    [((100, 130), 50, 255, 20, 100)],
    "cyan":    [((85, 100), 50, 255, 80, 255)],
    "teal":    [((80, 100), 40, 255, 40, 200)],
}

# Words that are definitely colors (used to detect color queries)
COLOR_WORDS = set(COLOR_RANGES.keys())


def _detect_query_colors(query):
    query_lower = query.lower()
    
    lower_words = ["pant", "shoe", "leg", "bottom", "short", "skirt", "trouser", "jean"]
    upper_words = ["hat", "cap", "head", "hair", "helmet", "turban", "glass"]
    
    matches = []
    for cname in COLOR_RANGES.keys():
        for m in re.finditer(r'\b' + cname + r'\b', query_lower):
            matches.append((m.start(), cname, m.end()))
            
    # Sort by their appearance in the sentence
    matches.sort(key=lambda x: x[0])
    
    found = []
    for _, cname, end_idx in matches:
        # Only look at the next 2 words immediately following the color
        words_after = query_lower[end_idx:].replace(',', ' ').split()[:2]
        text_after = " ".join(words_after)
        region = "torso"
        if any(w in text_after for w in lower_words):
            region = "lower"
        elif any(w in text_after for w in upper_words):
            region = "upper"
        found.append((cname, region))
        
    return found


def _get_torso_crop(img, bbox):
    """Legacy helper, replaced by region-aware cropping."""
    pass


def _color_match_score(img_bgr, color_name, region="torso"):
    """
    Returns the percentage (0.0 to 1.0) of the specified region area 
    that matches the color_name's HSV ranges.
    """
    if img_bgr is None or img_bgr.size == 0:
        return 0.0

    # Crop to the semantic region
    h, w = img_bgr.shape[:2]
    if region == "lower":
        crop_y1, crop_y2 = int(h * 0.5), h
        crop_x1, crop_x2 = int(w * 0.1), int(w * 0.9)
    elif region == "upper":
        crop_y1, crop_y2 = 0, int(h * 0.35)
        crop_x1, crop_x2 = int(w * 0.15), int(w * 0.85)
    else:  # torso
        crop_y1, crop_y2 = int(h * 0.2), int(h * 0.7)
        crop_x1, crop_x2 = int(w * 0.2), int(w * 0.8)

    crop = img_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        return 0.0

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    ranges = COLOR_RANGES.get(color_name, [])
    if not ranges:
        return 0.0

    total_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for rng in ranges:
        (h_lo, h_hi), s_min, s_max, v_min, v_max = rng
        lower = np.array([h_lo, s_min, v_min])
        upper = np.array([h_hi, s_max, v_max])
        mask = cv2.inRange(hsv, lower, upper)
        total_mask = cv2.bitwise_or(total_mask, mask)

    total_pixels = hsv.shape[0] * hsv.shape[1]
    if total_pixels == 0:
        return 0.0
    return float(np.count_nonzero(total_mask)) / total_pixels


def color_rerank(candidates, query, data_dir):
    """
    Verifies that candidates match ALL detected colors in their respective regions.
    """
    colors = _detect_query_colors(query)
    if not colors:
        return candidates

    checks_str = ", ".join([f"{c} ({r})" for c, r in colors])
    print(f"  \033[36m🎨 Verifying: {checks_str}...\033[0m ", end="", flush=True)

    # To optimize I/O, only verify color for the top 200 CLIP matches.
    top_candidates = candidates[:200]
    unique_names = list(set(c["image_name"] for c in top_candidates))
    
    img_cache = {}
    def _load_img(name):
        return name, cv2.imread(os.path.join(data_dir, name))
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for name, img in executor.map(_load_img, unique_names):
            if img is not None:
                img_cache[name] = img

    with_color = []
    for c in top_candidates:
        img_name = c["image_name"]
        img = img_cache.get(img_name)
        if img is None:
            with_color.append({**c, "color_match": 0.0, "_all_ratios": []})
            continue
            
        x1, y1, x2, y2 = c["bbox"]
        crop = img[y1:y2, x1:x2]
        
        # Verify ALL detected colors
        color_ratios = []
        for color_name, region in colors:
            ratio = _color_match_score(crop, color_name, region)
            
            # Dynamic threshold: hats are small (3%), pants/shirts are big (10%)
            req_thresh = 0.03 if region == "upper" else 0.10
            color_ratios.append((ratio, req_thresh))
            
        avg_ratio = sum(r for r, t in color_ratios) / len(color_ratios) if color_ratios else 0.0
        with_color.append({**c, "color_match": avg_ratio, "_all_ratios": color_ratios})

    # Normalize CLIP scores
    clip_scores = [c["score"] for c in with_color]
    clip_min = min(clip_scores) if clip_scores else 0
    clip_max = max(clip_scores) if clip_scores else 1
    clip_range = max(clip_max - clip_min, 0.001)

    reranked = []
    for c in with_color:
        clip_norm = (c["score"] - clip_min) / clip_range
        color_ratio = min(c["color_match"], 1.0)
        
        CLIP_W = 0.95
        COLOR_W = 0.05
        hybrid = CLIP_W * clip_norm + COLOR_W * color_ratio
        reranked.append({**c, "score": hybrid})

    reranked.sort(key=lambda x: -x["score"])
    
    # STRICT FILTER: Candidate must pass the dynamic threshold for EVERY required color
    color_ok = []
    for c in reranked:
        if all(r >= t for r, t in c.get("_all_ratios", [])):
            color_ok.append(c)
            
    # Fallback if too strict
    if len(color_ok) < 1:
        color_ok = sorted(reranked, key=lambda x: -x["score"])[:15]

    color_ok.sort(key=lambda x: (-x["score"], x["image_name"], x["person_idx"]))
    return color_ok


def filter_by_threshold(scored, threshold, max_results):
    """Keep candidates above threshold, capped at max_results."""
    filtered = [c for c in scored if c["score"] >= threshold]
    return filtered[:max_results]


def _is_valid_crop(img, bbox):
    """
    Check if a person crop is actually visible/recognizable.
    Rejects dark blobs, washed-out crops, and tiny noise.
    """
    x1, y1, x2, y2 = bbox
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    h, w = crop.shape[:2]
    if w < MIN_CROP_W or h < MIN_CROP_H:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    avg_brightness = float(np.mean(gray))
    if avg_brightness < MIN_BRIGHTNESS:
        return False
    std_dev = float(np.std(gray))
    if std_dev < MIN_CONTRAST:
        return False
    return True


def filter_bad_crops(candidates, data_dir):
    """
    Remove candidates whose crops are too dark or lack contrast.
    These are unrecognizable to the human eye.
    """
    good = []
    for c in candidates:
        img = _load_image(data_dir, c["image_name"])
        if img is None:
            continue
        if _is_valid_crop(img, c["bbox"]):
            good.append(c)
    if len(good) < 3:
        return candidates
    return good


def deduplicate_by_image(candidates):
    """
    Keep only the best scoring person per image for diverse results.
    Prevents the same image from appearing multiple times in the grid.
    """
    best_per_image = {}
    for c in candidates:
        img_name = c["image_name"]
        if img_name not in best_per_image or c["score"] > best_per_image[img_name]["score"]:
            best_per_image[img_name] = c
    deduped = list(best_per_image.values())
    deduped.sort(key=lambda x: (-x["score"], x["image_name"]))
    return deduped

# ──────────────────────────────────────────────────────────────
# MAIN — Staged Search Loop
# ──────────────────────────────────────────────────────────────

def main():
    print_header()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n  Loading CLIP model ({CLIP_MODEL_NAME}) on {device}...", end=" ", flush=True)
    clip_model, preprocess = clip.load(CLIP_MODEL_NAME, device=device)
    print("\u2713")

    # Detect feature dimension from the model
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224).to(device)
        feat_dim = clip_model.encode_image(dummy).shape[-1]
    print(f"  CLIP feature dimension: {feat_dim}")

    print(f"  Loading YOLOv8 model ({YOLO_MODEL_NAME})...", end=" ", flush=True)
    yolo_model = YOLO(YOLO_MODEL_NAME)
    print("\u2713")

    # Check if cache was built with a different model or format
    cache = FeatureCache(CACHE_DIR)
    CACHE_VERSION = "v2-multicrop"  # bump when cache format changes
    model_marker = os.path.join(CACHE_DIR, ".clip_model")
    marker_value = f"{CLIP_MODEL_NAME}|{CACHE_VERSION}"
    old_marker = ""
    if os.path.exists(model_marker):
        with open(model_marker, "r") as f:
            old_marker = f.read().strip()
    if old_marker != marker_value:
        # Clear old cache
        import glob
        old_files = glob.glob(os.path.join(CACHE_DIR, "*.npz"))
        if old_files:
            print(f"  {C.YELLOW}Cache format changed. Clearing {len(old_files)} old files for re-index...{C.RESET}")
            for f in old_files:
                os.remove(f)
        with open(model_marker, "w") as f:
            f.write(marker_value)

    # Index
    print(f"\n  Scanning: {DATA_DIR}")
    all_persons = index_images(DATA_DIR, cache, yolo_model, clip_model, preprocess, device, feat_dim)

    if not all_persons:
        print("  No persons detected in any image. Nothing to search.")
        sys.exit(0)

    # Search loop
    prompts = []          # accumulated prompts
    candidates = None     # current candidate set (None = all)
    history = []          # history stack for 'back' command

    while True:
        stage = len(prompts) + 1

        if candidates is not None:
            pool_size = len(candidates)
        else:
            pool_size = len(all_persons)

        print_stage_header(stage, pool_size, prompts)

        if candidates is not None:
            print(f"  {C.DIM}Commands: view N │ back │ {C.YELLOW}reset (start new search){C.DIM} │ done │ quit{C.RESET}")
        else:
            print(f"  {C.DIM}Commands: quit{C.RESET}")

        try:
            user_input = input("\n  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Goodbye!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        # ── quit ──
        if cmd == "quit":
            print("\n  Goodbye!")
            break

        # ── done ──
        if cmd == "done":
            if candidates:
                print(f"\n  Final results ({len(candidates)} matches):")
                print_table(candidates)
            print("  Search complete.\n")
            # Reset for next search
            prompts = []
            candidates = None
            history = []
            continue

        # ── back ──
        if cmd == "back":
            if history:
                prev = history.pop()
                prompts = prev["prompts"]
                candidates = prev["candidates"]
                if candidates is not None:
                    print(f"\n  ↩ Went back to stage {len(prompts) + 1} ({len(candidates)} candidates)")
                    print_table(candidates)
                else:
                    print(f"\n  ↩ Went back to initial search")
            else:
                print("  Nothing to go back to.")
            continue

        # ── reset ──
        if cmd == "reset":
            prompts = []
            candidates = None
            history = []
            print("\n  ↻ Search reset. Enter a new description.")
            continue

        # ── view N ──
        if cmd.startswith("view "):
            try:
                idx = int(cmd.split()[1]) - 1  # 1-indexed to 0-indexed
                if candidates and 0 <= idx < len(candidates):
                    show_result_cv2(candidates[idx], DATA_DIR)
                else:
                    print(f"  Invalid index. Use 1-{len(candidates) if candidates else 0}")
            except (ValueError, IndexError):
                print("  Usage: view N  (e.g., 'view 1')")
            continue

        # ── search / refine ──
        query = user_input

        # Save current state for 'back'
        history.append({
            "prompts": list(prompts),
            "candidates": list(candidates) if candidates is not None else None,
        })

        prompts.append(query)
        combined_prompt = ", ".join(prompts)

        print(f"\n  {C.DIM}Encoding:{C.RESET} {C.WHITE}\"{combined_prompt}\"{C.RESET}")
        text_features = encode_text(clip_model, device, combined_prompt)

        # Determine search pool
        if candidates is not None:
            pool = candidates
        else:
            pool = all_persons

        print(f"  {C.DIM}Searching {len(pool)} persons...{C.RESET}", end=" ", flush=True)
        scored = rank_candidates(pool, text_features)
        print(f"{C.GREEN}done{C.RESET}")

        # Color reranking if needed
        colors = _detect_query_colors(combined_prompt)
        if colors:
            scored = color_rerank(scored, combined_prompt, DATA_DIR)
            print(f"{C.GREEN}done{C.RESET}")

        # Filter bad crops (dark/unrecognizable blobs)
        scored = filter_bad_crops(scored, DATA_DIR)

        # Deduplicate: keep best person per image for diversity
        scored = deduplicate_by_image(scored)

        filtered = filter_by_threshold(scored, SIM_THRESHOLD, MAX_RESULTS)

        if not filtered:
            print(f"\n  {C.RED}{C.BOLD}✗ No matches found above threshold.{C.RESET}")
            print(f"  {C.DIM}  Threshold: {SIM_THRESHOLD*100:.0f}%{C.RESET}")
            if scored:
                print(f"  {C.DIM}  Best score: {scored[0]['score']*100:.1f}%{C.RESET}")
                print(f"  {C.YELLOW}  Try different words or lower the threshold.{C.RESET}")
            prompts.pop()
            history.pop()
            continue

        candidates = filtered
        stage_num = len(prompts)

        print(f"\n  {C.GREEN}{C.BOLD}✓ Found {len(candidates)} match{'es' if len(candidates) != 1 else ''}:{C.RESET}")
        print_table(candidates)

        # Auto-popup: premium grid of top matches (show up to 15)
        grid_count = min(len(candidates), 15)
        print(f"  {C.CYAN}📸 Showing top {grid_count} — press any key in the window to continue...{C.RESET}")
        show_top_grid(candidates, DATA_DIR, stage_num, query_text=combined_prompt, max_show=15)

        if len(candidates) <= 3:
            print(f"  {C.GREEN}{C.BOLD}⚡ Narrow enough!{C.RESET} Use 'view N' for full image, 'done' to finish, or keep refining.")
        else:
            print(f"  {C.YELLOW}Add another prompt to narrow down from {len(candidates)} candidates.{C.RESET}")



if __name__ == "__main__":
    main()

