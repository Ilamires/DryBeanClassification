import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons

MENU_RECT = [0.01, 0.30, 0.12, 0.45]
MAIN_RECT = [0.23, 0.08, 0.75, 0.86]


def add_viewer(plots: dict, title: str = "Dry Bean Viewer"):
    fig = plt.figure(figsize=(14, 9))
    fig.canvas.manager.set_window_title(title)

    ax_menu = fig.add_axes(MENU_RECT)
    ax_main = fig.add_axes(MAIN_RECT)
    radio = RadioButtons(ax_menu, list(plots.keys()), active=0)

    def draw(label):
        for a in list(fig.axes):
            if a is not ax_menu and a is not ax_main:
                fig.delaxes(a)
        ax_main.clear()
        ax_main.set_position(MAIN_RECT)
        ax_main.set_aspect("auto")
        plots[label](ax_main)
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ("left", "right"):
            keys = list(plots.keys())
            i = keys.index(radio.value_selected)
            i = (i + 1) % len(keys) if event.key == "right" else (i - 1) % len(keys)
            radio.set_active(i)

    radio.on_clicked(draw)
    fig.canvas.mpl_connect("key_press_event", on_key)
    draw(radio.value_selected)