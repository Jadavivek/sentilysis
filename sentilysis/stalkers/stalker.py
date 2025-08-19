from dataclasses import dataclass
from typing import Protocol
from playwright.sync_api import sync_playwright

class StalkerActions(Protocol):
    def scroll(self, direction: str, amount: int) -> None:
        ...

    def click(self, x: int, y: int) -> None:
        ...

    def screenshot(self, filename: str) -> None:
        ...

    def keypress(self, key: str) -> None:
        ...

@dataclass
class Stalker:
    name: str

    def __post_init__(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch()
        self.context = self.browser.new_context()
        self.page = self.context.new_page()

    def page_text(self) -> str:
        return self.page.content()

    def scroll(self, direction: str, amount: int) -> None:
        if direction == "down":
            self.page.evaluate(f"window.scrollBy(0, {amount})")
        elif direction == "up":
            self.page.evaluate(f"window.scrollBy(0, -{amount})")

    def click(self, x: int, y: int) -> None:
        self.page.mouse.click(x, y)

    def screenshot(self, filename: str) -> None:
        self.page.screenshot(path=filename)

    def keypress(self, key: str) -> None:
        self.page.keyboard.press(key)

    def get_prev_page(self):
        page = self.page.go_back()
        self.page.go_forward()
        return page

    def close(self):
        self.browser.close()
        self.playwright.stop()