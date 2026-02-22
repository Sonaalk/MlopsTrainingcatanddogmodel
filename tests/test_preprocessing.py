import os

def test_processed_data_exists():
    assert os.path.exists("data/processed/train")
    assert os.path.exists("data/processed/val")
    assert os.path.exists("data/processed/test")