"""
Copyright (c) <2018> YoongiKim

 See the file license.txt for copying permission.
"""

import win32gui
import win32con

def enumHandler(hwnd, lParam):
    if win32gui.IsWindowVisible(hwnd):
        if 'BlueStacks' in win32gui.GetWindowText(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW)

def ShowWindow():
    win32gui.EnumWindows(enumHandler, None)