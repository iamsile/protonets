import mmap


def mapcount(path: str):
    """Returns number of lines in a text file."""
    f = open(path, "r+")
    buf = mmap.mmap(f.fileno(), 0)
    lines = 0
    readline = buf.readline
    while readline():
        lines += 1
    return lines
