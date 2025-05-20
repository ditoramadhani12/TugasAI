import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import graphviz
from datetime import datetime
import os
from ttkthemes import ThemedTk
from tkinter import colorchooser

class FermentationMonitor:
    def __init__(self, root):
        self.root = root
        self.root.title("Fermentation Monitor")
        self.root.geometry("1400x800")
        
        # Set theme and style
        self.style = ttk.Style()
        self.style.theme_use('clam')  # Use clam theme as base
        
        # Configure colors and styles
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabelframe', background='#f0f0f0')
        self.style.configure('TLabelframe.Label', font=('Helvetica', 10, 'bold'), background='#f0f0f0')
        self.style.configure('TButton', font=('Helvetica', 9), padding=5)
        self.style.configure('TLabel', font=('Helvetica', 9), background='#f0f0f0')
        self.style.configure('Header.TLabel', font=('Helvetica', 12, 'bold'), background='#f0f0f0')
        self.style.configure('TNotebook', background='#f0f0f0')
        self.style.configure('TNotebook.Tab', font=('Helvetica', 9), padding=[10, 2])
        
        # Tambahkan style untuk setiap tab
        self.style.configure('Input.TFrame', background='#eaf6fb')      # Biru muda
        self.style.configure('Data.TFrame', background='#eafbe6')       # Hijau muda
        self.style.configure('Model.TFrame', background='#fffbe6')      # Kuning muda
        self.style.configure('Eval.TFrame', background='#f3e6fb')       # Ungu muda
        self.style.configure('Input.TLabelframe', background='#eaf6fb')
        self.style.configure('Data.TLabelframe', background='#eafbe6')
        self.style.configure('Model.TLabelframe', background='#fffbe6')
        self.style.configure('Eval.TLabelframe', background='#f3e6fb')
        self.style.configure('Input.TLabelframe.Label', background='#eaf6fb')
        self.style.configure('Data.TLabelframe.Label', background='#eafbe6')
        self.style.configure('Model.TLabelframe.Label', background='#fffbe6')
        self.style.configure('Eval.TLabelframe.Label', background='#f3e6fb')
        
        # Initialize variables
        self.model = None
        self.dataset = None
        self.current_data = None
        self.color_rgb = None  # Store RGB values
        
        # Create main container with scrollbar
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs with tk.Frame for background color support
        self.input_tab = tk.Frame(self.notebook, bg='#eaf6fb')
        self.data_tab = tk.Frame(self.notebook, bg='#eafbe6')
        self.model_tab = tk.Frame(self.notebook, bg='#fffbe6')
        self.evaluation_tab = tk.Frame(self.notebook, bg='#f3e6fb')
        
        self.notebook.add(self.input_tab, text="Input Data")
        self.notebook.add(self.data_tab, text="Data Management")
        self.notebook.add(self.model_tab, text="Model")
        self.notebook.add(self.evaluation_tab, text="Evaluation")
        
        # Initialize all components
        self.create_input_frame()
        self.create_data_frame()
        self.create_model_frame()
        self.create_evaluation_frame()
        
    def get_color_name(self, rgb):
        """Convert RGB values to color name"""
        r, g, b = rgb
        
        # Define color ranges
        if r > 200 and g > 200 and b > 200:
            return "Bright"
        elif r > 150 and g > 100 and b < 100:
            return "Light Brown"
        elif r > 100 and g > 50 and b < 50:
            return "Dark Brown"
        elif r < 50 and g < 50 and b < 50:
            return "Black"
        else:
            return f"RGB({r},{g},{b})"

    def create_input_frame(self):
        input_frame = ttk.LabelFrame(self.input_tab, text="Sensor Data Input", padding="15", style='Input.TLabelframe')
        input_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        input_frame.config(style='Input.TLabelframe')
        
        # Frame kiri dan kanan
        left_frame = ttk.Frame(input_frame, style='Input.TFrame')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        right_frame = ttk.Frame(input_frame, style='Input.TFrame')
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        
        # Manual Input Section
        manual_frame = ttk.LabelFrame(left_frame, text="Manual Input", padding="10", style='Input.TLabelframe')
        manual_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create a grid layout for inputs
        input_grid = ttk.Frame(manual_frame, style='Input.TFrame')
        input_grid.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Temperature
        ttk.Label(input_grid, text="Temperature (°C):", style='TLabel').grid(row=0, column=0, padx=10, pady=8, sticky="w")
        self.temp_var = tk.StringVar()
        ttk.Entry(input_grid, textvariable=self.temp_var, width=20).grid(row=0, column=1, padx=10, pady=8)
        
        # Humidity
        ttk.Label(input_grid, text="Humidity (%):", style='TLabel').grid(row=1, column=0, padx=10, pady=8, sticky="w")
        self.humidity_var = tk.StringVar()
        ttk.Entry(input_grid, textvariable=self.humidity_var, width=20).grid(row=1, column=1, padx=10, pady=8)
        
        # pH
        ttk.Label(input_grid, text="pH:", style='TLabel').grid(row=2, column=0, padx=10, pady=8, sticky="w")
        self.ph_var = tk.StringVar()
        ttk.Entry(input_grid, textvariable=self.ph_var, width=20).grid(row=2, column=1, padx=10, pady=8)
        
        # Gas
        ttk.Label(input_grid, text="Gas (ppm):", style='TLabel').grid(row=3, column=0, padx=10, pady=8, sticky="w")
        self.gas_var = tk.StringVar()
        ttk.Entry(input_grid, textvariable=self.gas_var, width=20).grid(row=3, column=1, padx=10, pady=8)
        
        # Fermentation Time
        ttk.Label(input_grid, text="Fermentation Time (days):", style='TLabel').grid(row=4, column=0, padx=10, pady=8, sticky="w")
        self.time_var = tk.StringVar()
        ttk.Entry(input_grid, textvariable=self.time_var, width=20).grid(row=4, column=1, padx=10, pady=8)
        
        # Color Inputs
        ttk.Label(input_grid, text="Color (RGB):", style='TLabel').grid(row=5, column=0, padx=10, pady=8, sticky="w")
        color_frame = ttk.Frame(input_grid)
        color_frame.grid(row=5, column=1, padx=10, pady=8, sticky="w")
        
        # Red
        ttk.Label(color_frame, text="R:").pack(side=tk.LEFT, padx=2)
        self.color_r_var = tk.StringVar()
        ttk.Entry(color_frame, textvariable=self.color_r_var, width=5).pack(side=tk.LEFT, padx=2)
        
        # Green
        ttk.Label(color_frame, text="G:").pack(side=tk.LEFT, padx=2)
        self.color_g_var = tk.StringVar()
        ttk.Entry(color_frame, textvariable=self.color_g_var, width=5).pack(side=tk.LEFT, padx=2)
        
        # Blue
        ttk.Label(color_frame, text="B:").pack(side=tk.LEFT, padx=2)
        self.color_b_var = tk.StringVar()
        ttk.Entry(color_frame, textvariable=self.color_b_var, width=5).pack(side=tk.LEFT, padx=2)
        
        # Color Preview
        self.color_preview = tk.Label(color_frame, width=8, height=1, relief="solid")
        self.color_preview.pack(side=tk.LEFT, padx=5)
        
        # Bind events to update color preview
        self.color_r_var.trace_add("write", self.update_color_preview)
        self.color_g_var.trace_add("write", self.update_color_preview)
        self.color_b_var.trace_add("write", self.update_color_preview)
        
        # Prediction Frame
        pred_frame = ttk.LabelFrame(right_frame, text="Prediction Results", padding="10")
        pred_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Prediction Result
        ttk.Label(pred_frame, text="Fermentation Status:", style='Header.TLabel').grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.prediction_var = tk.StringVar()
        ttk.Label(pred_frame, textvariable=self.prediction_var, font=('Helvetica', 11, 'bold')).grid(row=0, column=1, padx=10, pady=10)
        
        # Confidence Score
        ttk.Label(pred_frame, text="Confidence Score:", style='Header.TLabel').grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.confidence_var = tk.StringVar()
        ttk.Label(pred_frame, textvariable=self.confidence_var, font=('Helvetica', 11, 'bold')).grid(row=1, column=1, padx=10, pady=10)
        
        # Decision Rules
        ttk.Label(pred_frame, text="Decision Rules:", style='Header.TLabel').grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.rules_text = tk.Text(pred_frame, height=8, width=40, font=('Helvetica', 9))
        self.rules_text.grid(row=2, column=1, padx=10, pady=10)
        
        # Buttons
        button_frame = ttk.Frame(input_frame)
        button_frame.pack(fill=tk.X, pady=15)
        
        # Style for buttons
        self.style.configure('Action.TButton', font=('Helvetica', 10, 'bold'), padding=8)
        
        ttk.Button(button_frame, text="Predict", command=self.predict, style='Action.TButton').grid(row=0, column=0, padx=10, pady=2, sticky='w')
        ttk.Button(button_frame, text="Add to Dataset", command=self.add_to_dataset, style='Action.TButton').grid(row=0, column=1, padx=10, pady=2, sticky='w')
        ttk.Button(button_frame, text="Clear Input", command=self.clear_input, style='Action.TButton').grid(row=0, column=2, padx=10, pady=2, sticky='w')
        ttk.Button(button_frame, text="Clear Data", command=self.clear_all, style='Action.TButton').grid(row=1, column=0, columnspan=3, padx=10, pady=(10,2), sticky='w')
        
    def create_data_frame(self):
        data_frame = ttk.LabelFrame(self.data_tab, text="Data Management", padding="15", style='Data.TLabelframe')
        data_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Buttons
        button_frame = ttk.Frame(data_frame, style='Data.TFrame')
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Load Data", command=self.load_data, style='Action.TButton').pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Save Data", command=self.save_data, style='Action.TButton').pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Export Data", command=self.export_data, style='Action.TButton').pack(side=tk.LEFT, padx=10)
        
        # Data Table
        table_frame = ttk.Frame(data_frame, style='Data.TFrame')
        table_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create Treeview with custom style
        self.style.configure("Treeview", font=('Helvetica', 9))
        self.style.configure("Treeview.Heading", font=('Helvetica', 9, 'bold'))
        
        columns = ("Temperature", "Humidity", "pH", "Gas", "Time", "Color_R", "Color_G", "Color_B", "Status")
        self.tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=15)
        
        # Set column headings and widths
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120, anchor=tk.CENTER)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack tree and scrollbar
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def create_model_frame(self):
        model_frame = ttk.LabelFrame(self.model_tab, text="Model Management", padding="15", style='Model.TLabelframe')
        model_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Buttons
        button_frame = ttk.Frame(model_frame, style='Model.TFrame')
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Load Model", command=self.load_model, style='Action.TButton').pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Train Model", command=self.train_model, style='Action.TButton').pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Save Model", command=self.save_model, style='Action.TButton').pack(side=tk.LEFT, padx=10)
        
        # Model Visualization Frame
        viz_frame = ttk.LabelFrame(model_frame, text="Model Visualization", padding="10", style='Model.TLabelframe')
        viz_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create figure for tree visualization
        self.fig = plt.Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def create_evaluation_frame(self):
        eval_frame = ttk.LabelFrame(self.evaluation_tab, text="Model Evaluation", padding="15", style='Eval.TLabelframe')
        eval_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Metrics Frame
        metrics_frame = ttk.LabelFrame(eval_frame, text="Performance Metrics", padding="10", style='Eval.TLabelframe')
        metrics_frame.pack(fill=tk.X, pady=10)
        
        # Accuracy
        ttk.Label(metrics_frame, text="Accuracy:", style='Header.TLabel').grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.accuracy_var = tk.StringVar()
        ttk.Label(metrics_frame, textvariable=self.accuracy_var, font=('Helvetica', 11, 'bold')).grid(row=0, column=1, padx=10, pady=10)
        
        # Confusion Matrix Frame
        cm_frame = ttk.LabelFrame(eval_frame, text="Confusion Matrix", padding="10", style='Eval.TLabelframe')
        cm_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create figure for confusion matrix
        self.cm_fig = plt.Figure(figsize=(6, 4), dpi=100)
        self.cm_canvas = FigureCanvasTkAgg(self.cm_fig, master=cm_frame)
        self.cm_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def predict(self):
        if not self.model:
            messagebox.showwarning("Peringatan", "Silakan muat atau latih model terlebih dahulu")
            return

        # Validasi input
        error_fields = []
        try:
            temp = float(self.temp_var.get())
        except ValueError:
            error_fields.append("Suhu (Temperature)")
        try:
            humidity = float(self.humidity_var.get())
        except ValueError:
            error_fields.append("Kelembaban (Humidity)")
        try:
            ph = float(self.ph_var.get())
        except ValueError:
            error_fields.append("pH")
        try:
            gas = float(self.gas_var.get())
        except ValueError:
            error_fields.append("Gas")
        try:
            time = float(self.time_var.get())
        except ValueError:
            error_fields.append("Waktu Fermentasi (Fermentation Time)")
        try:
            r = int(self.color_r_var.get())
            g = int(self.color_g_var.get())
            b = int(self.color_b_var.get())
            if not all(0 <= x <= 255 for x in [r, g, b]):
                error_fields.append("Warna RGB: Nilai harus antara 0-255")
        except ValueError:
            error_fields.append("Warna RGB: Masukkan angka bulat (0-255)")

        if error_fields:
            error_message = "Mohon periksa input berikut:\n"
            for field in error_fields:
                error_message += f"- {field}\n"
            messagebox.showerror("Error Input", error_message)
            return

        try:
            # Siapkan data input
            X = np.array([[temp, humidity, ph, gas, time, r, g, b]])

            # Prediksi
            prediction = self.model.predict(X)
            probabilities = self.model.predict_proba(X)

            # Tampilkan hasil
            self.prediction_var.set(prediction[0])
            self.confidence_var.set(f"{max(probabilities[0])*100:.2f}%")
            self.show_decision_rules(X)

        except Exception as e:
            import traceback
            messagebox.showerror("Error", f"Terjadi kesalahan saat prediksi:\n{str(e)}\n\n{traceback.format_exc()}")
            
    def show_decision_rules(self, X):
        # Extract and display decision rules
        if not self.model:
            self.rules_text.delete(1.0, tk.END)
            self.rules_text.insert(tk.END, "Aturan keputusan akan ditampilkan di sini")
            return

        # Get feature names
        feature_names = ['Suhu', 'Kelembaban', 'pH', 'Gas', 'Waktu', 'Warna_R', 'Warna_G', 'Warna_B']
        
        # Get the path of the decision tree for this sample
        path = self.model.decision_path(X)
        node_indices = path.indices
        
        # Get the rules
        rules = []
        for node_idx in node_indices:
            if node_idx == 0:  # Root node
                continue
                
            # Get parent node
            parent = self.model.tree_.children_left[node_idx]
            if parent == -1:
                parent = self.model.tree_.children_right[node_idx]
                
            # Get feature and threshold
            feature = self.model.tree_.feature[parent]
            threshold = self.model.tree_.threshold[parent]
            
            # Determine if it's a left or right child
            if self.model.tree_.children_left[parent] == node_idx:
                rule = f"{feature_names[feature]} <= {threshold:.2f}"
            else:
                rule = f"{feature_names[feature]} > {threshold:.2f}"
                
            rules.append(rule)
        
        # Display rules in Indonesian
        self.rules_text.delete(1.0, tk.END)
        if rules:
            self.rules_text.insert(tk.END, "Aturan Keputusan:\n\n")
            for i, rule in enumerate(rules, 1):
                self.rules_text.insert(tk.END, f"{i}. {rule}\n")
        else:
            self.rules_text.insert(tk.END, "Tidak ada aturan keputusan yang tersedia")
        
    def add_to_dataset(self):
        if not self.prediction_var.get():
            messagebox.showwarning("Warning", "Please make a prediction first")
            return
        try:
            # Get RGB values
            r = int(self.color_r_var.get() or 0)
            g = int(self.color_g_var.get() or 0)
            b = int(self.color_b_var.get() or 0)
            # Validate RGB values
            if not all(0 <= x <= 255 for x in [r, g, b]):
                raise ValueError("RGB values must be between 0 and 255")
            # Create new row
            new_row = {
                'Temperature': float(self.temp_var.get()),
                'Humidity': float(self.humidity_var.get()),
                'pH': float(self.ph_var.get()),
                'Gas': float(self.gas_var.get()),
                'Time': float(self.time_var.get()),
                'Color_R': r,
                'Color_G': g,
                'Color_B': b,
                'Status': self.prediction_var.get()
            }
            # Add to treeview
            self.tree.insert('', 'end', values=list(new_row.values()))
            # Add to dataset (DataFrame)
            if self.dataset is None:
                import pandas as pd
                self.dataset = pd.DataFrame(columns=['Temperature','Humidity','pH','Gas','Time','Color_R','Color_G','Color_B','Status'])
            self.dataset = pd.concat([self.dataset, pd.DataFrame([new_row])], ignore_index=True)
            # Clear input
            self.clear_input()
        except ValueError as e:
            messagebox.showerror("Error", str(e))

    def clear_input(self):
        self.temp_var.set('')
        self.humidity_var.set('')
        self.ph_var.set('')
        self.gas_var.set('')
        self.time_var.set('')
        self.color_r_var.set('')
        self.color_g_var.set('')
        self.color_b_var.set('')
        self.color_preview.configure(bg='white')
        self.prediction_var.set('')
        self.confidence_var.set('')
        self.rules_text.delete(1.0, tk.END)

    def clear_data(self):
        """Clear all data from the table and dataset"""
        if messagebox.askyesno("Confirm Clear", "Are you sure you want to clear all data? This action cannot be undone."):
            # Clear treeview
            for item in self.tree.get_children():
                self.tree.delete(item)
            # Reset dataset to empty DataFrame with correct columns
            import pandas as pd
            self.dataset = pd.DataFrame(columns=['Temperature','Humidity','pH','Gas','Time','Color_R','Color_G','Color_B','Status'])
            # Show confirmation message
            messagebox.showinfo("Success", "All data has been cleared successfully")

    def load_data(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")]
        )
        if file_path:
            try:
                if file_path.endswith('.csv'):
                    self.dataset = pd.read_csv(file_path)
                else:
                    self.dataset = pd.read_excel(file_path)
                    
                # Update treeview
                self.update_treeview()
                
            except Exception as e:
                messagebox.showerror("Error", f"Error loading data: {str(e)}")
                
    def save_data(self):
        if not self.dataset is not None:
            messagebox.showwarning("Warning", "No data to save")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")]
        )
        if file_path:
            try:
                if file_path.endswith('.csv'):
                    self.dataset.to_csv(file_path, index=False)
                else:
                    self.dataset.to_excel(file_path, index=False)
                    
                messagebox.showinfo("Success", "Data saved successfully")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error saving data: {str(e)}")
                
    def export_data(self):
        if not self.dataset is not None:
            messagebox.showwarning("Warning", "No data to export")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")]
        )
        if file_path:
            try:
                # Get data from treeview
                data = []
                for item in self.tree.get_children():
                    data.append(self.tree.item(item)['values'])
                    
                # Create DataFrame
                df = pd.DataFrame(data, columns=self.tree['columns'])
                
                # Save to file
                if file_path.endswith('.csv'):
                    df.to_csv(file_path, index=False)
                else:
                    df.to_excel(file_path, index=False)
                    
                messagebox.showinfo("Success", "Data exported successfully")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error exporting data: {str(e)}")
                
    def update_treeview(self):
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        # Add new items
        for _, row in self.dataset.iterrows():
            self.tree.insert('', 'end', values=list(row))
            
    def load_model(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Pickle files", "*.pkl")]
        )
        if file_path:
            try:
                import pickle
                with open(file_path, 'rb') as f:
                    self.model = pickle.load(f)
                    
                # Update model visualization
                self.update_model_visualization()
                
                messagebox.showinfo("Success", "Model loaded successfully")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error loading model: {str(e)}")
                
    def train_model(self):
        if self.dataset is None:
            messagebox.showwarning("Warning", "Please load data first")
            return
            
        try:
            # Prepare data
            X = self.dataset.drop('Status', axis=1)
            y = self.dataset['Status']
            
            # Train model
            self.model = DecisionTreeClassifier(random_state=42)
            self.model.fit(X, y)
            
            # Update visualization
            self.update_model_visualization()
            
            # Update evaluation metrics
            self.update_evaluation_metrics(X, y)
            
            messagebox.showinfo("Success", "Model trained successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error training model: {str(e)}")
            
    def save_model(self):
        if not self.model:
            messagebox.showwarning("Warning", "No model to save")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl")]
        )
        if file_path:
            try:
                import pickle
                with open(file_path, 'wb') as f:
                    pickle.dump(self.model, f)
                    
                messagebox.showinfo("Success", "Model saved successfully")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error saving model: {str(e)}")
                
    def update_model_visualization(self):
        if not self.model:
            return
            
        # Clear figure
        self.fig.clear()
        
        # Plot decision tree
        ax = self.fig.add_subplot(111)
        plot_tree(self.model, ax=ax, feature_names=self.dataset.columns[:-1],
                 class_names=self.model.classes_, filled=True, rounded=True)
        
        # Update canvas
        self.canvas.draw()
        
    def update_evaluation_metrics(self, X, y):
        if not self.model:
            return
            
        # Calculate predictions
        y_pred = self.model.predict(X)
        
        # Update accuracy
        accuracy = accuracy_score(y, y_pred)
        self.accuracy_var.set(f"{accuracy*100:.2f}%")
        
        # Update confusion matrix
        self.cm_fig.clear()
        ax = self.cm_fig.add_subplot(111)
        cm = confusion_matrix(y, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        
        # Update canvas
        self.cm_canvas.draw()

    def update_color_preview(self, *args):
        """Update color preview based on RGB values"""
        try:
            r = int(self.color_r_var.get() or 0)
            g = int(self.color_g_var.get() or 0)
            b = int(self.color_b_var.get() or 0)
            
            # Validate RGB values
            if all(0 <= x <= 255 for x in [r, g, b]):
                self.color_rgb = (r, g, b)
                self.color_preview.configure(bg=f'#{r:02x}{g:02x}{b:02x}')
        except ValueError:
            pass

    def clear_all(self):
        self.clear_input()
        self.clear_data()

if __name__ == "__main__":
    root = ThemedTk(theme="arc")  # Use a modern theme
    app = FermentationMonitor(root)
    root.mainloop() 