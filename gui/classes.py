import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from cfd_model import CP_Predict_Model
import pandas as pd


def submit_action():
    print("Hello submit")
    return "Hello text"


class App(tk.Tk):
    def __init__(self, title, size):
        # main setup
        super().__init__()
        self.title(title)
        self.geometry(f'{size[0]}x{size[1]}')
        self.minsize(size[0], size[1])

        # widgets
        self.cp_model = CP_Predict_Model()
        self.menu = Menu(self, self.handle_submit)
        self.output = Output(self)

        # run
        self.mainloop()

    def handle_submit(self, aoa_float):
        result = self.cp_model.predict_airfoil_cp(aoa_float)
        self.output.update_canvas(result)


class Menu(ttk.Frame):
    def __init__(self, parent, submit_callback):
        super().__init__(parent)

        self.submit_callback = submit_callback
        self.create_input()

        self.place(x=0, y=0, relwidth=0.3, relheight=1)

    def create_input(self):
        frame = ttk.Frame(self)
        # create widgets
        self.aoa_float = tk.DoubleVar(value=0.0)
        aoa_label = ttk.Label(frame, text='Angle of Attack')
        aoa_entry = ttk.Entry(frame, textvariable=self.aoa_float)
        submit_button = ttk.Button(
            frame,
            text='Submit',
            command=lambda: self.submit_callback(self.aoa_float.get()))

        # config layout
        frame.rowconfigure((0, 1), weight=1, uniform='a')
        frame.columnconfigure((0, 1), weight=1, uniform='a')

        # create layout
        aoa_label.grid(row=0, column=0)
        aoa_entry.grid(row=0, column=1)
        submit_button.grid(row=1, column=0, columnspan=2)
        frame.pack()


class Output(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        self.create_cfd_graph()

        self.place(relx=0.3, y=0, relwidth=0.7, relheight=1)

    def create_cfd_graph(self):
        frame = ttk.Frame(self)
        self.fig = Figure(figsize=(5, 5), dpi=100)
        y = [i**2 for i in range(101)]

        # adding the subplot
        self.plot1 = self.fig.add_subplot(111)

        # plotting the graph
        self.plot1.plot(y)

        # creating the Tkinter canvas
        # containing the Matplotlib figure
        self.canvas = FigureCanvasTkAgg(self.fig,
                                        master=frame)
        self.canvas.draw()

        # placing the canvas on the Tkinter window
        self.canvas.get_tk_widget().pack()

        # creating the Matplotlib toolbar
        toolbar = NavigationToolbar2Tk(self.canvas,
                                       frame)
        toolbar.update()

        # placing the toolbar on the Tkinter window
        self.canvas.get_tk_widget().pack()
        frame.pack()

    def update_canvas(self, result):
        top_surface_pred = result[result['y'] > 0].sort_values(by='x')
        bottom_surface_pred = result[result['y'] <= 0].sort_values(by='x')

        self.plot1.clear()
        self.plot1.scatter(top_surface_pred['x'], top_surface_pred['Cp_predicted'],
                           color='purple', linewidth=1, label='ML Predicted (Top Surface)')
        self.plot1.scatter(bottom_surface_pred['x'], bottom_surface_pred['Cp_predicted'],
                           color='orange', linewidth=1, label='ML Predicted (Bottom Surface)')
        self.plot1.invert_yaxis()
        self.plot1.grid(True)
        self.plot1.set_title(f"NN Prediction for Unseen Case: AoA")
        self.canvas.draw()


if __name__ == "__main__":
    App('CFD Predictor', (800, 600))
