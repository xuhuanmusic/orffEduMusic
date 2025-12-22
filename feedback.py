# feedback.py
"""
Feedback generation module for the Children Rhythm AI project.
Provides:
  • visual feedback  (matplotlib image)
  • text feedback    (Chinese prompt)
  • audio feedback   (text‑to‑speech via pyttsx3, optional)

Usage example
-------------
>>> fb = FeedbackManager()
>>> text = fb.generate_text(rhythm_class=2, rhythm_deviation=0.18)
>>> fb.generate_audio(text)
>>> fb.generate_visual(pred_skel, true_skel, save_path="viz.png")
"""

from typing import Optional
import numpy as np

# ---------- Visual feedback -------------------------------------------------
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt


class FeedbackManager:
    DRUM_LABELS = ["Kick", "Snare", "Hi‑Hat"]

    # ---------------- TEXT --------------------------------------------------
    def generate_text(self, rhythm_class: int, rhythm_deviation: float) -> str:
        """Return an English textual feedback string.
        rhythm_class   : 0‑7 (rhythm density level, 0 sparse — 7 extremely dense)
        rhythm_deviation: average micro-deviation, -1.0~1.0 (negative = too fast, positive = too slow)
        """
        density_msgs = [
            "The rhythm is very sparse; consider adding more drum beats.",
            "The rhythm is somewhat sparse; try adding more fills.",
            "The rhythm is well-balanced; keep it up.",
            "The rhythm is slightly tight; remember to breathe.",
            "The rhythm is quite dense; consider leaving some space.",
            "The rhythm is dense; be careful not to overdo it.",
            "The rhythm is very dense; consider simplifying it.",
            "The rhythm is extremely dense; this may affect clarity."
        ]
        rhythm_class = int(np.clip(rhythm_class, 0, 7))
        dens_msg = density_msgs[rhythm_class]

        if rhythm_deviation < -0.1:
            dev_msg = "Overall too fast; try to slow down the tempo."
        elif rhythm_deviation > 0.1:
            dev_msg = "Overall too slow; try to pick up the pace."
        else:
            dev_msg = "The rhythm is stable overall; keep it up!"
        return dens_msg + " " + dev_msg

    # ---------------- AUDIO -------------------------------------------------
    def generate_audio(self, text: str, lang: str = "zh") -> None:
        """Speak the feedback via TTS (pyttsx3). If unavailable, silently skip."""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 160)  # speech speed
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print("[Feedback] TTS failed or pyttsx3 not installed ->", e)

    # ---------------- VISUAL -----------------------------------------------
    def generate_visual(
        self,
        pred_skel: np.ndarray,
        true_skel: np.ndarray,
        save_path: Optional[str] = None,
        title: str = "Rhythm Skeleton Comparison",
    ) -> str:
        """Plot predicted vs true skeleton matrices.
        pred_skel / true_skel: shape [T, 3] or [B, T, 3] (use first sample)
        Returns the path of saved image.
        """
        if pred_skel.ndim == 3:
            pred_skel = pred_skel[0]
            true_skel = true_skel[0]
        pred_bin = (pred_skel >= 0.5).astype(float)
        true_bin = (true_skel >= 0.5).astype(float)

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.imshow(true_bin.T, aspect="auto", cmap="Reds", alpha=0.5)
        ax.imshow(pred_bin.T, aspect="auto", cmap="Blues", alpha=0.5)
        ax.set_yticks(range(3))
        ax.set_yticklabels(self.DRUM_LABELS)
        ax.set_xlabel("Time Step")
        ax.set_title(title)
        plt.tight_layout()

        if save_path is None:
            save_path = "feedback_visual.png"
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return save_path
