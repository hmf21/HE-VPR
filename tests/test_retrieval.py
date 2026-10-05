import unittest
from pathlib import Path

from run import coordinates


class RetrievalTests(unittest.TestCase):
    def test_coordinate_conventions(self):
        reference = Path("@tile@120.5@36.5@100.tif")
        query = Path("@query@36.5@120.5@100.png")
        self.assertEqual(coordinates(reference, "gstudio", False), (120.5, 36.5))
        self.assertEqual(coordinates(query, "gstudio", True), (120.5, 36.5))

if __name__ == "__main__":
    unittest.main()
