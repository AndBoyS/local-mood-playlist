from dataclasses import dataclass

from const import REPO_DIR

LIBRARY_DIR = REPO_DIR / ".dev-local/testdata"


@dataclass(frozen=True)
class Track:
    name: str
    sha256: str


TRACKS = (
    Track(
        name="mudeth - Antibirth- OST - 34 Fitnah (Lamb Fight).mp3",
        sha256="8322dc9764fe3ed375cd3b62e410d36fe50fbb2469c9fbade93bc872a7d98733",
    ),
    Track(
        name="mudeth - Antibirth- OST - 35 Flashpoint (Burning Basement).mp3",
        sha256="ce670fbb324c74ec8bffc18ea7773c162b0ef8131b500530a343ece52fb1780f",
    ),
)
