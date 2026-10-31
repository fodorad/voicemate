import unittest

from voicemate.hots.screen.macos import pick_window


def window(number: int, owner: str, name: str, width: int, height: int) -> dict:
    return {
        "kCGWindowNumber": number,
        "kCGWindowOwnerName": owner,
        "kCGWindowName": name,
        "kCGWindowBounds": {"Width": width, "Height": height},
    }


class TestPickWindow(unittest.TestCase):
    def test_the_biggest_window_of_the_app_wins(self):
        windows = [window(1, "Heroes", "", 400, 300), window(2, "Heroes", "", 1920, 1080)]
        self.assertEqual(pick_window(windows, "Heroes"), 2)

    def test_a_browser_tab_about_the_game_is_not_the_game(self):
        windows = [
            window(7, "Brave Browser", "Heroes of the Storm - Icy Veins", 2560, 1440),
            window(2, "Heroes", "", 1920, 1080),
        ]
        self.assertEqual(pick_window(windows, "Heroes"), 2)
        self.assertIsNone(pick_window(windows[:1], "Heroes"))

    def test_the_app_name_matches_in_any_case(self):
        self.assertEqual(pick_window([window(3, "Heroes of the Storm", "", 10, 10)], "heroes"), 3)


if __name__ == "__main__":
    unittest.main()
