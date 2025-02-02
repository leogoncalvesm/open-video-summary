from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.parent


class DataPaths:
    RAW_PATH: str = (PROJECT_DIR / "data/raw/").as_posix()
    PROCESSED_PATH: str = (PROJECT_DIR / "data/processed/").as_posix()


class ModelPaths:
    SUBJECTIVITY_CLASSIFIER: str = (
        PROJECT_DIR / "models/subjectivity_classifier.mdl"
    ).as_posix()
    FACE_CASCADE: str = (
        PROJECT_DIR / "models/lbpcascade_frontalface_improved.xml"
    ).as_posix()
