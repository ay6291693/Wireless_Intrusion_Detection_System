import tkinter as tk
from tkinter import ttk, messagebox
from scapy.all import sniff, IP
from threading import Thread
import pandas as pd
import joblib
from scapy.layers.inet import TCP, UDP, ICMP
from datetime import datetime
import signal
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from collections import deque

model = joblib.load("ids_model.pkl")
feature_order = joblib.load("feature_names.pkl")

total_packets = 0
anomaly_count = 0
normal_count = 0
prediction_log = []
gui_queue = []

time_history = deque(maxlen=50)
anomaly_history = deque(maxlen=50)
normal_history = deque(maxlen=50)

WIFI_INTERFACE = "Wi-Fi"

def protocol_name(pkt):
    if pkt.haslayer(TCP): return "TCP"
    elif pkt.haslayer(UDP): return "UDP"
    elif pkt.haslayer(ICMP): return "ICMP"
    else: return "OTHER"

def extract_features(pkt):
    if not pkt.haslayer(IP): return None
    return {
        'Protocol_' + protocol_name(pkt): 1,
        'Source_' + pkt[IP].src: 1,
        'Destination_' + pkt[IP].dst: 1,
        'Length': len(pkt)
    }

def predict_packet(pkt):
    global total_packets, anomaly_count, normal_count
    features = extract_features(pkt)
    if not features: return

    full_feature = {col: 0 for col in feature_order}
    full_feature.update(features)
    df = pd.DataFrame([full_feature], columns=feature_order)

    pred = model.predict(df)[0]
    label = "Anomaly" if pred == 1 else "Normal"

    total_packets += 1
    if pred == 1: anomaly_count += 1
    else: normal_count += 1

    timestamp = datetime.now().strftime("%H:%M:%S")
    prediction_log.append({
        "Timestamp": timestamp,
        "Source": pkt[IP].src,
        "Destination": pkt[IP].dst,
        "Protocol": protocol_name(pkt),
        "Length": len(pkt),
        "Prediction": label
    })

    display = f"[{timestamp}] {pkt[IP].src} → {pkt[IP].dst} | {protocol_name(pkt)} | Len: {len(pkt)} | {label}"
    color = "red" if pred == 1 else "green"
    gui_queue.append((display, color))

    time_history.append(timestamp)
    anomaly_history.append(anomaly_count)
    normal_history.append(normal_count)

def start_sniffing():
    sniff(prn=predict_packet, store=0, iface=WIFI_INTERFACE)

def save_log():
    if prediction_log:
        df_log = pd.DataFrame(prediction_log)
        df_log.to_csv(r"Wifi_Logs\ids_predictions_log.csv", mode='a', index=False, header=not os.path.exists(r"Wifi_Logs\ids_predictions_log.csv"))
        messagebox.showinfo("Log Saved", "Log saved to 'Wifi_Logs\ids_predictions_log.csv'")
    else:
        messagebox.showinfo("No Data", "No predictions to save yet.")

def on_closing():
    save_log()
    root.destroy()
    sys.exit(0)

def update_gui():
    while gui_queue:
        text, color = gui_queue.pop(0)
        log.insert('', 'end', values=(text,), tags=(color,))
        log.yview_moveto(1.0)
        log.tag_configure("red", background="#ffcccc")
        log.tag_configure("green", background="#ccffcc")

    total_label.config(text=f"Total: {total_packets}")
    anomaly_label.config(text=f"Anomalies: {anomaly_count}")
    normal_label.config(text=f"Normal: {normal_count}")

    if total_packets > 0:
        ax.clear()
        ax.pie([normal_count, anomaly_count], labels=["Normal", "Anomaly"], autopct='%1.1f%%', colors=["#90ee90", "#ff6961"])
        ax.set_title("Live Packet Classification", fontsize=16)
        canvas.draw()

        ax_bar.clear()
        ax_bar.bar(['Total', 'Normal', 'Anomaly'], [total_packets, normal_count, anomaly_count], color=['blue', 'green', 'red'])
        ax_bar.set_title("Traffic Stats", fontsize=14)
        ax_bar.set_ylim(0, max(total_packets, 10))
        canvas_bar.draw()

        ax_line.clear()
        ax_line.plot(time_history, normal_history, label='Normal', color='green')
        ax_line.plot(time_history, anomaly_history, label='Anomaly', color='red')
        ax_line.set_title("Live Traffic Trend", fontsize=14)
        ax_line.set_ylabel("Packet Count")
        ax_line.set_xlabel("Time")
        ax_line.legend(loc='upper left')
        canvas_line.draw()

    root.after(1000, update_gui)

root = tk.Tk()
root.title("AI-Based IDS for Wireless Network")
from tkinter import font
style = ttk.Style()
default_font = font.nametofont("TkDefaultFont")
default_font.configure(size=12)
style.configure("Treeview", font=("Arial", 13))
style.configure("Treeview.Heading", font=("Arial", 14, "bold"))
root.state('zoomed')
root.configure(bg="#f0f2f5")

header = tk.Label(root, text="Real-Time Intrusion Detection (Wi-Fi Only)", font=("Arial", 20, "bold"), bg="#f0f2f5")
header.pack(pady=10)

button_frame = tk.Frame(root, bg="#f0f2f5")
button_frame.pack(pady=5)

start_btn = tk.Button(button_frame, text="Start Monitoring", font=("Arial", 14), bg="#4caf50", fg="white", command=lambda: Thread(target=start_sniffing, daemon=True).start())
start_btn.grid(row=0, column=0, padx=15)

save_btn = tk.Button(button_frame, text="Save Log", font=("Arial", 14), bg="#2196f3", fg="white", command=save_log)
save_btn.grid(row=0, column=1, padx=15)

content_frame = tk.Frame(root, bg="#f0f2f5")
content_frame.pack(fill="both", expand=True, padx=10, pady=10)

log_frame = tk.Frame(content_frame, bg="#f0f2f5")
log_frame.pack(side="left", fill="both", expand=True, padx=10)

log_scrollbar = tk.Scrollbar(log_frame)
log_scrollbar.pack(side="right", fill="y")

log = ttk.Treeview(log_frame, columns=("Packet Info",), show="headings", height=25, yscrollcommand=log_scrollbar.set)
log.heading("Packet Info", text="Live Packet Predictions")
log.column("Packet Info", width=900)
log.pack(side="left", fill="both", expand=True)
log_scrollbar.config(command=log.yview)

stats_graph_frame = tk.Frame(content_frame, bg="#f0f2f5")
stats_graph_frame.pack(side="right", fill="both", expand=True, padx=10)

stats_frame = tk.Frame(stats_graph_frame, bg="#f0f2f5")
stats_frame.pack(pady=10)

total_label = tk.Label(stats_frame, text="Total: 0", font=("Arial", 14, "bold"), fg="black", bg="#f0f2f5")
total_label.grid(row=0, column=0, padx=20)

anomaly_label = tk.Label(stats_frame, text="Anomalies: 0", font=("Arial", 14, "bold"), fg="red", bg="#f0f2f5")
anomaly_label.grid(row=0, column=1, padx=20)

normal_label = tk.Label(stats_frame, text="Normal: 0", font=("Arial", 14, "bold"), fg="green", bg="#f0f2f5")
normal_label.grid(row=0, column=2, padx=20)

fig, ax = plt.subplots(figsize=(5, 5))
canvas = FigureCanvasTkAgg(fig, master=stats_graph_frame)
canvas.get_tk_widget().pack()

fig_bar, ax_bar = plt.subplots(figsize=(5, 3))
canvas_bar = FigureCanvasTkAgg(fig_bar, master=stats_graph_frame)
canvas_bar.get_tk_widget().pack(pady=10)

fig_line, ax_line = plt.subplots(figsize=(5, 2.5))
canvas_line = FigureCanvasTkAgg(fig_line, master=stats_graph_frame)
canvas_line.get_tk_widget().pack(pady=10)

root.protocol("WM_DELETE_WINDOW", on_closing)
signal.signal(signal.SIGINT, lambda sig, frame: on_closing())
update_gui()
root.mainloop()