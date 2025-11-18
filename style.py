from matplotlib import pyplot as plt
from typing import Optional, List
from matplotlib.axes import Axes
from matplotlib import collections
from enum import Enum

from typing import Optional, List
from matplotlib.axes import Axes
from matplotlib import pyplot as plt
from matplotlib import collections
from enum import Enum


class MyStyle:
    THEMES = {
        "slate": {
            "dark_1": "#073763",
            "dark_2": "#0b5394",
            "dark_3": "#4687b0",
            "light_2": "#d6d3c6",
            "light_1": "#f3f3f3",
        },
        "zinc": {
            "dark_1": "#333333",
            "dark_2": "#585858",
            "dark_3": "#7F7F7F",
            "light_2": "#E5E5E5",
            "light_1": "#EFEFEF",
        }
    }
    
        

    class Palette(str, Enum):
        RED = "#c42d2c"
        ORANGE = "#c4692d"
        YELLOW_ORANGE = "#c4a52d"
        YELLOW_GREEN = "#8bc42d"
        GREEN = "#2dc459"
        CYAN = "#2dc4a5"
        BLUE = "#2d8bc4"
        DEEP_BLUE = "#2d4fc4"
        PURPLE = "#692dc4"
        MAGENTA = "#a52dc4"
        PINK = "#c42d8b"
        ROSE = "#c42d4f"
        
        @classmethod
        def values(cls) -> list[str]:
            return [member.value for member in cls]

    
    def __init__(self, theme: str = "zinc"):
        if theme not in self.THEMES:
            raise ValueError(f"Theme must be one of {list(self.THEMES.keys())}")
        
        colors = self.THEMES[theme]
        self.dark_1 = colors["dark_1"]
        self.dark_2 = colors["dark_2"]
        self.dark_3 = colors["dark_3"]
        self.light_2 = colors["light_2"]
        self.light_1 = colors["light_1"]
    
    @staticmethod
    def _get_font_family(weight: str = "") -> str:
        import matplotlib.font_manager as fm
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        if "Inter" in available_fonts:
            return f"Inter {weight}".strip()
        return "sans-serif"
    
    def _apply_style_to_ax(self, ax: Axes, legend_frame_on: bool, colored_labels: bool, colored_bg: bool) -> None:
        is_heatmap = any(isinstance(coll, collections.QuadMesh) for coll in ax.collections)
        
        ax.grid(not is_heatmap, linestyle="dashed", linewidth=1.2, color=self.light_2, alpha=1)
        
        if colored_bg:
            ax.set_facecolor(self.light_1)
        else:
            ax.set_facecolor("none")
        
        for spine in ax.spines.values():
            spine.set_color(self.dark_1)
            spine.set_linewidth(1.2)
        
        if colored_labels:
            ax.tick_params(axis="both", colors=self.dark_1, width=1.2)
        
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(self._get_font_family())
            label.set_fontsize(12)
        
        if ax.get_title():
            ax.title.set_fontfamily(self._get_font_family())
            ax.title.set_fontsize(16)
            ax.title.set_color(self.dark_1)
        
        for axis_label in [ax.xaxis.label, ax.yaxis.label]:
            if axis_label.get_text():
                axis_label.set_fontfamily(self._get_font_family())
                axis_label.set_fontsize(14)
                if colored_labels:
                    axis_label.set_color(self.dark_1)
        
        legend = ax.get_legend()
        if legend:
            legend.set_frame_on(legend_frame_on)
            for label in legend.get_texts():
                label.set_fontfamily(self._get_font_family())
                label.set_fontsize(12)
            legend.get_title().set_fontfamily(self._get_font_family())
            legend.get_title().set_fontsize(14)
            legend.get_title().set_color(self.dark_1)
    
    def apply(self, axes: Optional[Axes | List[Axes]] = None, legend_frame_on: bool = False, 
              colored_labels: bool = True, colored_bg: bool = False) -> None:
        """Apply custom styling to matplotlib plots."""
        fig = plt.gcf()
        
        if colored_bg:
            fig.set_facecolor(self.light_1)
        else:
            fig.set_facecolor("none")
        
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=self.Palette.values())
        
        if fig.get_suptitle():
            fig.suptitle(
                fig.get_suptitle(), 
                fontfamily=self._get_font_family(), 
                fontsize=21, 
                color=self.dark_1
            )
        
        if axes is None:
            axes = plt.gca()
        
        try:
            axes_list = axes.flatten().tolist() if hasattr(axes, 'flatten') else \
                       [axes] if isinstance(axes, Axes) else list(axes)
        except TypeError:
            axes_list = [axes]
        
        for ax in fig.get_axes():
            self._apply_style_to_ax(ax, legend_frame_on, colored_labels, colored_bg)