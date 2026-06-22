"""Download the Prosey target corpus from the GITenberg GitHub mirror.

gutenberg.org blocks scraping and is off the sandbox whitelist; GITenberg mirrors
each book as a repo `<slug>_<gutenberg_id>` with a clean `<id>.txt` at the root.

Writes books/<stem>.txt (raw, boilerplate stripped later by the pipeline) and
books/metadata.json. Resilient: logs and skips any book that can't be resolved
or fetched. Run:  .venv/bin/python download_corpus.py
"""

import json
import re
import time
import urllib.request
from pathlib import Path

BOOKS_DIR = Path("books")
SEARCH_SLEEP = 6.5  # GitHub unauthenticated Search API: ~10 req/min

# (author, title, year, genre, nationality, gender, gutenberg_id)
BOOKS = [
    # British fiction
    ("Jane Austen", "Emma", 1815, "fiction", "British", "F", 158),
    ("Jane Austen", "Persuasion", 1818, "fiction", "British", "F", 105),
    ("Jane Austen", "Pride and Prejudice", 1813, "fiction", "British", "F", 1342),
    ("George Eliot", "Middlemarch", 1872, "fiction", "British", "F", 145),
    ("George Eliot", "Silas Marner", 1861, "fiction", "British", "F", 550),
    ("Charlotte Bronte", "Jane Eyre", 1847, "fiction", "British", "F", 1260),
    ("Emily Bronte", "Wuthering Heights", 1847, "fiction", "British", "F", 768),
    ("Charles Dickens", "Bleak House", 1853, "fiction", "British", "M", 1023),
    ("Charles Dickens", "Great Expectations", 1861, "fiction", "British", "M", 1400),
    ("Charles Dickens", "David Copperfield", 1850, "fiction", "British", "M", 766),
    ("Thomas Hardy", "Tess of the d'Urbervilles", 1891, "fiction", "British", "M", 110),
    ("Thomas Hardy", "Far from the Madding Crowd", 1874, "fiction", "British", "M", 107),
    ("Joseph Conrad", "Heart of Darkness", 1899, "fiction", "British", "M", 219),
    ("Oscar Wilde", "The Picture of Dorian Gray", 1890, "fiction", "Irish", "M", 174),
    ("Elizabeth Gaskell", "North and South", 1855, "fiction", "British", "F", 4276),
    ("Anthony Trollope", "Barchester Towers", 1857, "fiction", "British", "M", 3409),
    ("Wilkie Collins", "The Moonstone", 1868, "fiction", "British", "M", 155),
    ("William Makepeace Thackeray", "Vanity Fair", 1848, "fiction", "British", "M", 599),
    ("Mary Shelley", "Frankenstein", 1818, "fiction", "British", "F", 84),
    ("Bram Stoker", "Dracula", 1897, "fiction", "Irish", "M", 345),
    ("Lewis Carroll", "Alice's Adventures in Wonderland", 1865, "fiction", "British", "M", 11),
    # American fiction
    ("Mark Twain", "Adventures of Huckleberry Finn", 1884, "fiction", "American", "M", 76),
    ("Mark Twain", "The Adventures of Tom Sawyer", 1876, "fiction", "American", "M", 74),
    ("Nathaniel Hawthorne", "The Scarlet Letter", 1850, "fiction", "American", "M", 33),
    ("Henry James", "The Portrait of a Lady", 1881, "fiction", "American", "M", 2833),
    ("Henry James", "Washington Square", 1880, "fiction", "American", "M", 2870),
    ("Edith Wharton", "The House of Mirth", 1905, "fiction", "American", "F", 284),
    ("Edith Wharton", "The Age of Innocence", 1920, "fiction", "American", "F", 541),
    ("Willa Cather", "My Antonia", 1918, "fiction", "American", "F", 242),
    ("Kate Chopin", "The Awakening", 1899, "fiction", "American", "F", 160),
    ("Jack London", "The Call of the Wild", 1903, "fiction", "American", "M", 215),
    ("Herman Melville", "Moby-Dick", 1851, "fiction", "American", "M", 2701),
    ("Edgar Allan Poe", "Tales of Mystery and Imagination", 1845, "fiction", "American", "M", 2147),
    # American nonfiction
    ("Henry David Thoreau", "Walden", 1854, "nonfiction", "American", "M", 205),
    ("Ralph Waldo Emerson", "Essays: First Series", 1841, "nonfiction", "American", "M", 2944),
    ("Frederick Douglass", "Narrative of the Life", 1845, "nonfiction", "American", "M", 23),
    ("W. E. B. Du Bois", "The Souls of Black Folk", 1903, "nonfiction", "American", "M", 408),
    ("Booker T. Washington", "Up from Slavery", 1901, "nonfiction", "American", "M", 2376),
    ("Hamilton, Madison & Jay", "The Federalist Papers", 1788, "nonfiction", "American", "M", 1404),
    # British essays and nonfiction
    ("Thomas Carlyle", "Sartor Resartus", 1836, "nonfiction", "British", "M", 1051),
    ("John Ruskin", "Unto This Last", 1860, "nonfiction", "British", "M", 26905),
    ("Thomas De Quincey", "Confessions of an English Opium-Eater", 1821, "nonfiction", "British", "M", 2040),
    ("William Hazlitt", "Table Talk", 1821, "nonfiction", "British", "M", 3020),
    # British poetry and drama
    ("William Shakespeare", "Hamlet", 1603, "drama", "British", "M", 1524),
    ("William Shakespeare", "Macbeth", 1623, "drama", "British", "M", 1533),
    ("William Shakespeare", "King Lear", 1606, "drama", "British", "M", 1532),
    ("John Milton", "Paradise Lost", 1667, "poetry", "British", "M", 26),
    ("John Keats", "Poems Published in 1820", 1820, "poetry", "British", "M", 23684),
    ("William Blake", "Songs of Innocence and of Experience", 1794, "poetry", "British", "M", 1934),
    ("Walt Whitman", "Leaves of Grass", 1855, "poetry", "American", "M", 1322),
    # Translated works
    ("Fyodor Dostoyevsky", "The Brothers Karamazov", 1880, "fiction", "Russian", "M", 28054),
    ("Fyodor Dostoyevsky", "Crime and Punishment", 1866, "fiction", "Russian", "M", 2554),
    ("Leo Tolstoy", "Anna Karenina", 1877, "fiction", "Russian", "M", 1399),
    ("Leo Tolstoy", "War and Peace", 1869, "fiction", "Russian", "M", 2600),
    ("Ivan Turgenev", "Fathers and Sons", 1862, "fiction", "Russian", "M", 30723),
    ("Anton Chekhov", "The Cherry Orchard", 1904, "drama", "Russian", "M", 7986),
    ("Gustave Flaubert", "Madame Bovary", 1857, "fiction", "French", "M", 2413),
    ("Victor Hugo", "Les Miserables", 1862, "fiction", "French", "M", 135),
    ("Henrik Ibsen", "A Doll's House", 1879, "drama", "Norwegian", "M", 2542),
    ("Henrik Ibsen", "Hedda Gabler", 1891, "drama", "Norwegian", "M", 4093),
    # Religious / classical
    ("Various", "King James Bible", 1611, "religious", "Other", "-", 10),
    ("Various", "Book of Common Prayer", 1662, "religious", "British", "-", 29622),
    # Early modern essays (translated)
    ("Michel de Montaigne", "Essays", 1580, "nonfiction", "French", "M", 3600),
]


def slug(s):
    return re.sub(r"[^A-Za-z0-9]", "", s)


def stem_for(author, title, year, genre, nat, gender):
    last = slug(author.split()[-1]) or slug(author)
    return f"{last}_{slug(title)}_{year}_{genre}_{nat}_{gender}"


def _get(url, accept_json=False):
    headers = {"User-Agent": "prosey-corpus"}
    if accept_json:
        headers["Accept"] = "application/vnd.github+json"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30)


def resolve_repo(gid):
    """Return the GITenberg repo name whose suffix is _<gid>, with 403 backoff."""
    api = f"https://api.github.com/search/repositories?q=_{gid}+in:name+user:GITenberg&per_page=20"
    for attempt in range(4):
        try:
            with _get(api, accept_json=True) as r:
                items = json.load(r).get("items", [])
            return next((it["name"] for it in items if it["name"].endswith(f"_{gid}")), None)
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                time.sleep(20 * (attempt + 1))  # rate limited — back off
                continue
            raise
    return None


def download_text(repo, gid, dest):
    # GITenberg names the clean text either <gid>.txt or <gid>-0.txt, on master or main.
    for branch in ("master", "main"):
        for fname in (f"{gid}.txt", f"{gid}-0.txt"):
            url = f"https://raw.githubusercontent.com/GITenberg/{repo}/{branch}/{fname}"
            try:
                with _get(url) as r:
                    data = r.read()
                if len(data) > 5000:  # sanity: real book, not an error page
                    dest.write_bytes(data)
                    return len(data)
            except urllib.error.HTTPError:
                continue
    return 0


def main():
    BOOKS_DIR.mkdir(exist_ok=True)
    metadata = {}
    ok, failed = [], []

    for i, (author, title, year, genre, nat, gender, gid) in enumerate(BOOKS, 1):
        stem = stem_for(author, title, year, genre, nat, gender)
        dest = BOOKS_DIR / f"{stem}.txt"
        print(f"[{i}/{len(BOOKS)}] {author} — {title} (#{gid})")

        if dest.exists() and dest.stat().st_size > 5000:
            print("    already present, skipping download")
        else:
            repo = resolve_repo(gid)
            time.sleep(SEARCH_SLEEP)
            if not repo:
                print("    !! could not resolve GITenberg repo")
                failed.append((stem, gid, "no repo"))
                continue
            size = download_text(repo, gid, dest)
            if not size:
                print(f"    !! download failed from {repo}")
                failed.append((stem, gid, "download failed"))
                continue
            print(f"    {repo}: {size // 1024} KB")

        metadata[stem] = {
            "author": author, "title": title, "year": year,
            "genre": genre, "nationality": nat, "gender": gender,
        }
        ok.append(stem)

    (BOOKS_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n=== Done: {len(ok)} downloaded, {len(failed)} failed ===")
    for stem, gid, why in failed:
        print(f"  FAILED #{gid}: {stem} ({why})")
    print(f"metadata.json written with {len(metadata)} entries")


if __name__ == "__main__":
    main()
