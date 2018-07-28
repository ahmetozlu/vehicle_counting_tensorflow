#!/usr/bin/python
# -*- coding: utf-8 -*-
# ----------------------------------------------
# --- Author         : Arthur Gustin
# --- Mail           : arthur.gustin@gmail.com
# --- Date           : 28 July 2018
# ----------------------------------------------

import os
import platform
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def get_nest_feeds_form_url(nest_feed_url = 'https://video.nest.com/live/lcrmO3X2oD', chrome_location='C:\Program Files (x86)\Google\Chrome\Application\chrome.exe'):
    mp4_feed=""
    m3u8_feed=""
    driver=None
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.binary_location = chrome_location

        is_windows = any(platform.win32_ver())
        if is_windows:
            driver = webdriver.Chrome(executable_path=os.path.abspath("chromedriver_win32"), chrome_options=chrome_options)
        else:
            driver = webdriver.Chrome(executable_path=os.path.abspath("chromedriver_linux64"), chrome_options=chrome_options)
        driver.get(nest_feed_url)
        wait = WebDriverWait(driver, 30)
        play_button = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "vjs-big-play-button")))
        play_button.click()

        mp4_feed = driver.find_element_by_xpath("//source[@type='rtmp/mp4']").get_attribute("src")
        m3u8_feed = driver.find_element_by_xpath("//source[@type='application/x-mpegURL']").get_attribute("src")

    except Exception as e:
        print(e)
    finally:
        if driver is not None:
            driver.close()
        return mp4_feed, m3u8_feed

if __name__ == "__main__":
    mp4_feed, m3u8_feed = get_nest_feeds_form_url()
    print(mp4_feed)
    print(m3u8_feed)