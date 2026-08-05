"""
@FileName：Calibration.py
@Description：
@Author：Ferry
@Time：2026 1/13/26 5:07 PM
@Copyright：©2024-2026 ShanghaiTech University-RIMLAB
"""
import argparse
import numpy as np
import cv2
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

def mm_to_pt(mm):  # PDF point
    return mm * 72.0 / 25.4

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dict", default="DICT_APRILTAG_36h11")
    ap.add_argument("--markersX", type=int, default=5)
    ap.add_argument("--markersY", type=int, default=7)
    ap.add_argument("--marker_mm", type=float, default=50.0)
    ap.add_argument("--sep_mm", type=float, default=10.0)
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--out_pdf", default="board.pdf")
    args = ap.parse_args()

    name_to_dict = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
        "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
        "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
        "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
    }
    if args.dict not in name_to_dict:
        raise ValueError(f"Unknown dict: {args.dict}")

    aruco_dict = cv2.aruco.getPredefinedDictionary(name_to_dict[args.dict])

    marker_m = args.marker_mm / 1000.0
    sep_m = args.sep_mm / 1000.0
    board = cv2.aruco.GridBoard((args.markersX, args.markersY), marker_m, sep_m, aruco_dict)

    board_w_mm = args.markersX * args.marker_mm + (args.markersX - 1) * args.sep_mm
    board_h_mm = args.markersY * args.marker_mm + (args.markersY - 1) * args.sep_mm

    # Pixels for rendering (high DPI)
    w_px = int(round(board_w_mm / 25.4 * args.dpi))
    h_px = int(round(board_h_mm / 25.4 * args.dpi))

    # Render board image
    try:
        img = board.generateImage((w_px, h_px), marginSize=0, borderBits=1)
    except Exception:
        img = cv2.aruco.drawPlanarBoard(board, (w_px, h_px), marginSize=0, borderBits=1)

    # Convert to RGB for PDF
    if img.ndim == 2:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img

    # Save temp PNG
    tmp_png = "board_tmp.png"
    cv2.imwrite(tmp_png, img_rgb)

    # Create PDF with exact physical page size = board size
    page_w_pt = mm_to_pt(board_w_mm)
    page_h_pt = mm_to_pt(board_h_mm)
    c = canvas.Canvas(args.out_pdf, pagesize=(page_w_pt, page_h_pt))
    c.drawImage(tmp_png, 0, 0, width=page_w_pt, height=page_h_pt, mask='auto')

    # Add a 100mm calibration line (for sanity check)
    # line from (10mm,10mm) to (110mm,10mm)
    c.setLineWidth(1)
    c.line(mm_to_pt(10), mm_to_pt(10), mm_to_pt(110), mm_to_pt(10))
    c.setFont("Helvetica", 8)
    c.drawString(mm_to_pt(10), mm_to_pt(12), "Calibration: 100 mm line (should measure 100mm)")

    c.showPage()
    c.save()

    print(f"[OK] PDF saved: {args.out_pdf}")
    print(f"Board size: {board_w_mm:.1f} mm x {board_h_mm:.1f} mm")
    print("Print setting MUST be: Actual size / 100% (disable Fit-to-page).")

if __name__ == "__main__":
    main()
