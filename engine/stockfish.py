"""
engine/stockfish.py — Minimal synchronous Stockfish wrapper.
"""

import subprocess
import os

STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH", "/usr/games/stockfish")


class Stockfish:
    def __init__(self, path=STOCKFISH_PATH, depth=15, elo=None):
        self.depth = depth
        self.proc  = subprocess.Popen(
            [path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1,
        )
        self._send("uci")
        self._wait("uciok")
        if elo:
            self._send("setoption name UCI_LimitStrength value true")
            self._send(f"setoption name UCI_Elo value {elo}")
        self._send("ucinewgame")
        self._send("isready")
        self._wait("readyok")

    def best_move(self, board) -> str | None:
        """Return best UCI move string, or None if game over."""
        moves = " ".join(m.uci() for m in board.move_stack)
        self._send("position startpos" + (f" moves {moves}" if moves else ""))
        self._send(f"go depth {self.depth}")
        while True:
            line = self.proc.stdout.readline().strip()
            if line.startswith("bestmove"):
                bm = line.split()[1]
                return None if bm == "(none)" else bm

    def close(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=3)
        except Exception:
            self.proc.kill()

    def _send(self, cmd: str):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _wait(self, token: str):
        while True:
            if token in self.proc.stdout.readline():
                return
