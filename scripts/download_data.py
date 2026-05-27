import sys
import urllib.request
from pathlib import Path

BASE_URL = 'https://raw.githubusercontent.com/zanfranceschi/rinha-de-backend-2026/main/resources'
FILES = (
    'references.json.gz',
    'mcc_risk.json',
    'normalization.json',
)


def download(name: str, dest_dir: Path) -> None:
    target = dest_dir / name
    if target.exists():
        print(f'skip  {name} (already at {target})')
        return
    url = f'{BASE_URL}/{name}'
    print(f'fetch {url}')
    tmp = target.with_suffix(target.suffix + '.part')
    urllib.request.urlretrieve(url, tmp)  # noqa: S310 — trusted GitHub raw URL
    tmp.rename(target)
    print(f'done  {target} ({target.stat().st_size} bytes)')


def main() -> int:
    dest = Path(__file__).resolve().parent.parent / 'data'
    dest.mkdir(exist_ok=True)
    for name in FILES:
        download(name, dest)
    return 0


if __name__ == '__main__':
    sys.exit(main())
