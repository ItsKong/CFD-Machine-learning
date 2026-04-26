import os

os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")

import tkinter as tk
from tkinter import messagebox
from cfd_model_rdfr import CP_Predict_Model_RDFR


class App(tk.Tk):
    def __init__(self, title, size):
        super().__init__()
        self.title(title)
        self.geometry(f'{size[0]}x{size[1]}')
        self.minsize(900, 600)
        self.configure(bg="#f4f4f4")

        self.cp_model = None
        self.status_text = tk.StringVar(value="Loading model...")

        main = tk.Frame(self, bg="#f4f4f4")
        main.pack(fill="both", expand=True)

        self.menu = Menu(main, self.handle_submit, self.status_text)
        self.menu.pack(side="left", fill="y", padx=16, pady=16)

        self.output = Output(main)
        self.output.pack(side="right", fill="both", expand=True, padx=(0, 16), pady=16)

        self.after_idle(self.load_model)
        self.mainloop()

    def load_model(self):
        self.update_idletasks()
        try:
            self.cp_model = CP_Predict_Model_RDFR()
        except Exception as exc:
            self.status_text.set("Model failed to load")
            messagebox.showerror("Model failed to load", str(exc))
            return

        self.menu.set_ready(True)
        self.status_text.set("Model ready")
        self.handle_submit(self.menu.get_aoa(), self.menu.get_mach())

    def handle_submit(self, aoa_float, mach_float):
        if self.cp_model is None:
            self.status_text.set("Model is still loading...")
            return

        self.status_text.set("Predicting...")
        self.update_idletasks()
        try:
            result = self.cp_model.predict_airfoil_cp(aoa_float, mach_float)
        except Exception as exc:
            self.status_text.set("Prediction failed")
            messagebox.showerror("Prediction failed", str(exc))
            return

        self.output.update_canvas(result, aoa_float)
        self.status_text.set(f"Showing AoA = {aoa_float:.2f} deg, Mach = {mach_float:.3f}")


class Menu(tk.Frame):
    def __init__(self, parent, submit_callback, status_text):
        super().__init__(parent, width=230, bg="#ededed", highlightthickness=1,
                         highlightbackground="#c8c8c8")
        self.pack_propagate(False)
        self.submit_callback = submit_callback
        self.status_text = status_text
        self.create_input()

    def create_input(self):
        # 1. Variables for both inputs
        self.aoa_text = tk.StringVar(value="0.0")
        self.mach_text = tk.StringVar(value="0.75")  # Default Mach value

        # Labels
        title_label = tk.Label(
            self, text="CFD Predictor", bg="#ededed", fg="#111111",
            font=("Arial", 18, "bold"))
        subtitle_label = tk.Label(
            self, text="RAE2822 Cp prediction", bg="#ededed", fg="#444444",
            font=("Arial", 11))
        
        # AoA Widgets
        aoa_label = tk.Label(
            self, text="Angle of Attack", bg="#ededed", fg="#111111",
            font=("Arial", 12))
        aoa_entry = tk.Entry(
            self, textvariable=self.aoa_text, font=("Arial", 14), width=10,
            relief="solid", bd=1)
            
        # 2. Mach Widgets
        mach_label = tk.Label(
            self, text="Mach Number", bg="#ededed", fg="#111111",
            font=("Arial", 12))
        mach_entry = tk.Entry(
            self, textvariable=self.mach_text, font=("Arial", 14), width=10,
            relief="solid", bd=1)

        self.submit_button = tk.Button(
            self, text="Submit", state="disabled", font=("Arial", 12),
            relief="solid", bd=1, command=self.submit)
        status_label = tk.Label(
            self, textvariable=self.status_text, bg="#ededed", fg="#333333",
            font=("Arial", 11), justify="left", wraplength=190)

        # Packing (slightly adjusted pady to fit the new input)
        title_label.pack(anchor="w", padx=18, pady=(20, 2))
        subtitle_label.pack(anchor="w", padx=18, pady=(0, 20))
        
        aoa_label.pack(anchor="w", padx=18, pady=(0, 4))
        aoa_entry.pack(fill="x", padx=18, pady=(0, 12))
        
        # 3. Pack the Mach widgets
        mach_label.pack(anchor="w", padx=18, pady=(0, 4))
        mach_entry.pack(fill="x", padx=18, pady=(0, 16))

        self.submit_button.pack(fill="x", padx=18, pady=(0, 18))
        status_label.pack(anchor="w", fill="x", padx=18)

        # Bind the Enter key for both entry boxes
        aoa_entry.bind("<Return>", lambda _event: self.submit())
        mach_entry.bind("<Return>", lambda _event: self.submit())

    def get_aoa(self):
        return float(self.aoa_text.get())

    # 4. Helper to get the Mach float
    def get_mach(self):
        return float(self.mach_text.get())

    def set_ready(self, ready):
        state = "normal" if ready else "disabled"
        self.submit_button.configure(state=state)

    def submit(self):
        try:
            aoa_float = self.get_aoa()
            mach_float = self.get_mach()
        except ValueError:
            self.status_text.set("Error: Enter numeric values.")
            return
            
        # --- NEW VALIDATION LOGIC ---
        if not (0.0 <= aoa_float <= 20.0):
            self.status_text.set("Error: AoA must be between 0 and 20.")
            return
            
        if not (0.68 <= mach_float <= 0.78):
            self.status_text.set("Error: Mach must be between 0.68 and 0.78.")
            return
        # ----------------------------

        # Clear the status text if everything is valid
        self.status_text.set("Calculating...") 
        
        self.submit_callback(aoa_float, mach_float)

class Output(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg="#ffffff", highlightthickness=1,
                         highlightbackground="#c8c8c8")
        self.latest_result = None
        self.latest_aoa = None
        self.canvas = tk.Canvas(self, bg="#ffffff", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self.redraw())
        self.draw_placeholder()

    def draw_placeholder(self):
        self.canvas.delete("all")
        self.canvas.create_text(
            40, 40, anchor="nw", text="Loading CFD predictor...",
            fill="#111111", font=("Arial", 18, "bold"))
        self.canvas.create_text(
            40, 76, anchor="nw", text="The pressure coefficient plot will appear here.",
            fill="#444444", font=("Arial", 12))

    def update_canvas(self, result, aoa_float):
        self.latest_result = result
        self.latest_aoa = aoa_float
        self.redraw()

    def redraw(self):
        if self.latest_result is None:
            self.draw_placeholder()
            return

        width = max(self.canvas.winfo_width(), 1)
        height = max(self.canvas.winfo_height(), 1)
        margin_left = 72
        margin_right = 28
        margin_top = 72
        margin_bottom = 64
        plot_w = max(width - margin_left - margin_right, 1)
        plot_h = max(height - margin_top - margin_bottom, 1)

        result = self.latest_result
        x_min = float(result["x"].min())
        x_max = float(result["x"].max())
        cp_min = float(result["Cp_predicted"].min())
        cp_max = float(result["Cp_predicted"].max())
        cp_pad = max((cp_max - cp_min) * 0.08, 0.05)
        cp_min -= cp_pad
        cp_max += cp_pad

        self.canvas.delete("all")
        self._draw_axes(width, height, margin_left, margin_top, plot_w, plot_h,
                        x_min, x_max, cp_min, cp_max)
        self._draw_surface(result[result["y"] > 0].sort_values(by="x"),
                           "#7a2cb0", margin_left, margin_top, plot_w, plot_h,
                           x_min, x_max, cp_min, cp_max)
        self._draw_surface(result[result["y"] <= 0].sort_values(by="x"),
                           "#d98219", margin_left, margin_top, plot_w, plot_h,
                           x_min, x_max, cp_min, cp_max)

        title = f"NN Prediction for AoA = {self.latest_aoa:.2f} deg"
        self.canvas.create_text(
            margin_left, 28, anchor="w", text=title, fill="#111111",
            font=("Arial", 18, "bold"))
        self._draw_legend(width)

    def _draw_axes(self, width, height, left, top, plot_w, plot_h,
                   x_min, x_max, cp_min, cp_max):
        right = left + plot_w
        bottom = top + plot_h
        self.canvas.create_rectangle(left, top, right, bottom, outline="#333333")
        self.canvas.create_text(
            left + plot_w / 2, height - 24, text="x",
            fill="#111111", font=("Arial", 12))
        self.canvas.create_text(
            24, top - 18, text="Cp", fill="#111111",
            font=("Arial", 12))

        for i in range(6):
            frac = i / 5
            x = left + frac * plot_w
            x_value = x_min + frac * (x_max - x_min)
            self.canvas.create_line(x, bottom, x, bottom + 5, fill="#333333")
            self.canvas.create_text(
                x, bottom + 20, text=f"{x_value:.2f}", fill="#333333",
                font=("Arial", 9))

        for i in range(6):
            frac = i / 5
            y = top + frac * plot_h
            cp_value = cp_min + frac * (cp_max - cp_min)
            self.canvas.create_line(left - 5, y, left, y, fill="#333333")
            self.canvas.create_text(
                left - 10, y, anchor="e", text=f"{cp_value:.2f}",
                fill="#333333", font=("Arial", 9))

    def _draw_surface(self, surface, color, left, top, plot_w, plot_h,
                      x_min, x_max, cp_min, cp_max):
        if surface.empty:
            return

        coords = []
        x_span = max(x_max - x_min, 1e-12)
        cp_span = max(cp_max - cp_min, 1e-12)
        for row in surface.itertuples(index=False):
            x = left + ((row.x - x_min) / x_span) * plot_w
            y = top + ((row.Cp_predicted - cp_min) / cp_span) * plot_h
            coords.extend((x, y))

        if len(coords) >= 4:
            self.canvas.create_line(*coords, fill=color, width=2, smooth=True)

        for x, y in zip(coords[0::2], coords[1::2]):
            self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2,
                                    outline=color, fill=color)

    def _draw_legend(self, width):
        x = max(width - 250, 120)
        y = 28
        self.canvas.create_line(x, y, x + 34, y, fill="#7a2cb0", width=3)
        self.canvas.create_text(
            x + 42, y, anchor="w", text="Top surface",
            fill="#333333", font=("Arial", 10))
        self.canvas.create_line(x, y + 22, x + 34, y + 22,
                                fill="#d98219", width=3)
        self.canvas.create_text(
            x + 42, y + 22, anchor="w", text="Bottom surface",
            fill="#333333", font=("Arial", 10))


if __name__ == "__main__":
    App('CFD Predictor', (1100, 700))
