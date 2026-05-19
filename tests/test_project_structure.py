import unittest
from pathlib import Path


class TestProjectStructure(unittest.TestCase):
    def test_required_paths_exist(self) -> None:
        root = Path(__file__).resolve().parents[1]
        required = [
            ".github/workflows",
            "config",
            "data",
            "feature_store/feature_definition.py",
            "feature_store/feature_store.yaml",
            "src/__init__.py",
            "src/ingestion.py",
            "src/train_base.py",
            "src/fine_tune.py",
            "src/monitor.py",
            "src/evaluate.py",
            "src/app.py",
            "tests",
            "Dockerfile",
            "requirements.txt",
            "README.md",
        ]
        for rel_path in required:
            self.assertTrue((root / rel_path).exists(), rel_path)


if __name__ == "__main__":
    unittest.main()
