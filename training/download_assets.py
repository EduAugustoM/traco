"""Baixa os assets de treino: fontes cursivas (Google Fonts, OFL/Apache) e léxico PT+EN.

Uso:  python training/download_assets.py
"""
import json
import re
import sys
import unicodedata
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FONTS_DIR = ROOT / "training" / "fonts"
DATA_DIR = ROOT / "data"

# Famílias cursivas (letras ligadas) do repositório google/fonts
FAMILIES = [
    "ofl/dancingscript", "ofl/greatvibes", "ofl/allura", "ofl/parisienne", "ofl/sacramento",
    "ofl/alexbrush", "apache/satisfy", "apache/yellowtail", "apache/homemadeapple",
    "ofl/cedarvillecursive", "ofl/labelleaurore", "ofl/mrssaintdelafield", "ofl/meddon",
    "ofl/dawningofanewday", "ofl/kristi", "ofl/herrvonmuellerhoff", "ofl/marckscript",
    "ofl/italianno", "ofl/pinyonscript", "ofl/tangerine", "ofl/arizonia", "ofl/clickerscript",
    "ofl/qwigley", "ofl/monsieurladoulaise", "ofl/msmadi", "ofl/engagement", "ofl/grandhotel",
    "ofl/courgette", "ofl/cookie", "ofl/mrdafoe", "ofl/norican", "ofl/playball",
    "ofl/euphoriascript", "ofl/loversquarrel", "ofl/ruthie", "ofl/nothingyoucoulddo",
    "ofl/zeyada", "ofl/niconne", "ofl/damion", "ofl/petitformalscript", "ofl/leaguescript",
    "ofl/rougescript", "ofl/mrdehaviland", "ofl/missfajardose", "ofl/allison", "ofl/corinthia",
    "ofl/ephesis", "ofl/windsong", "ofl/waterbrush", "ofl/imperialscript", "ofl/meaculpa",
    "ofl/thenautigal", "ofl/islandmoments", "ofl/licorice", "ofl/hurricane", "ofl/comforterbrush",
    "ofl/babylonica", "ofl/ole", "ofl/puppiesplay", "ofl/squarepeg", "ofl/grechenfuemen",
    "ofl/caveat", "ofl/kalam", "ofl/yesteryear", "ofl/sofia", "ofl/bilbo", "ofl/vibur",
]

LEX_URLS = {
    "en": "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/en/en_50k.txt",
    "pt": "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/content/2018/pt_br/pt_br_50k.txt",
}


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "desafio-final"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def download_fonts() -> None:
    FONTS_DIR.mkdir(parents=True, exist_ok=True)
    ok = 0
    for fam in FAMILIES:
        name = fam.split("/")[1]
        if any(FONTS_DIR.glob(f"{name}__*.ttf")):
            ok += 1
            continue
        try:
            listing = json.loads(get(f"https://api.github.com/repos/google/fonts/contents/{fam}"))
            ttfs = [f for f in listing if f["name"].lower().endswith(".ttf")]
            if not ttfs:
                print(f"  [sem ttf] {fam}")
                continue
            ttfs.sort(key=lambda f: (0 if "regular" in f["name"].lower() or "[" in f["name"] else 1,
                                     "italic" in f["name"].lower(), len(f["name"])))
            f = ttfs[0]
            (FONTS_DIR / f"{name}__{f['name'].replace('[', '_').replace(']', '_')}").write_bytes(get(f["download_url"]))
            ok += 1
            print(f"  ok  {fam} -> {f['name']}")
        except Exception as e:  # noqa: BLE001
            print(f"  [erro] {fam}: {e}")
    print(f"Fontes: {ok}/{len(FAMILIES)}")


def strip_accents(w: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", w) if not unicodedata.combining(c))


def download_lexicon(top: int = 30000) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    words: dict[str, int] = {}
    for lang, url in LEX_URLS.items():
        lines = get(url).decode("utf-8").splitlines()
        n = 0
        for line in lines:
            parts = line.split()
            if len(parts) != 2:
                continue
            w = strip_accents(parts[0].lower())
            if not re.fullmatch(r"[a-z]{2,14}", w):
                continue
            words[w] = words.get(w, 0) + int(parts[1])
            n += 1
            if n >= top:
                break
        print(f"Léxico {lang}: {n} palavras")
    for w in ["a", "e", "o", "i", "u"]:
        words.setdefault(w, 1)
    ordered = sorted(words.items(), key=lambda kv: -kv[1])
    (DATA_DIR / "lexicon_pt_en.txt").write_text("\n".join(f"{w}\t{c}" for w, c in ordered), encoding="utf-8")
    print(f"Léxico total: {len(ordered)} -> data/lexicon_pt_en.txt")


if __name__ == "__main__":
    what = sys.argv[1:] or ["fonts", "lexicon"]
    if "lexicon" in what:
        download_lexicon()
    if "fonts" in what:
        download_fonts()
