# Radio Environment Map Interpolation

This project is focused on possibilities of Radio Environment Map interpolations. This repository contains scripts for full reproducibility of research article: **Influence of Measured Radio Environment Map Interpolation on Indoor Positioning Algorithms** submitted to IEEE Sensors.

The repository is split into 3 sections:

- Data Analysis
- Probe Request Sniffer
- Probe Request Sender

## [Probe Request Sniffer](ESP32_Probe_Request_Sniffer/)

The sniffer is based on ESP32 microcontroller, specifically ESP32-CAM which has integrated microSD card slot on the board itself. The sniffer is based on another [project](https://gitlab.com/tbravenec/esp32-probe-sniffer) and the only modification is an extension of the sniffer with MAC address filter. This filter allows us to gather data from only one device and create radiomap of the monitored space.

Before deployment, the [config.h](ESP32_Probe_Request_Sniffer/main/config.h) should be modified accordingly with nearby Wi-Fi credentials. The connection to Wi-Fi is required only during the initialization to get current time for pcap files from NTP server. 2nd change should be change of the MAC address filter in [sniffer.c](ESP32_Probe_Request_Sniffer/main/sniffer.c).

The sniffers should be placed around the space to be monitored with enough distance between them to show the difference between RSSI values of incoming packets. When powered the sniffer goes through initial setup. It connects to Wi-Fi and downloads current time. Initializes SD card and shuts down LED when the sniffing process starts, The button press will stop sniffing and save current file and indicates it is not sniffing packets anymore by lighting up the LED.

## [Probe Request Sender](ESP32_Probe_Request_Sender/)

Just like the Probe Request Sniffer, the sender is based on ESP32-CAM. Before usage the [config.h](ESP32_Probe_Request_Sender/main/config.h) should be modified in similar way to the config.h of the Probe Request Sniffer. The credentials for Wi-Fi has to be changed to a valid Wi-Fi for the download of current time. The amount of Probes to be sent as well as the filename can be set here.

The sender then on the press of button sends the set amount of probes and indicates the process of sending probe requests by lighting up an LED. The LED shuts down when the cycle is done. The ESP32 also writes the time the button was pressed and the time the sending cycle finished into the file.

## [Data Analysis](Data_Analysis)

This section of the repository utilizes Python for scripting. The folder contains several scripts:
- [data_pcap_to_csv.py](Data_Analysis/data_pcap_to_csv.py)
    - This script goes through the pcap files created by the Probe Request Sniffers into CSV files.
    - Except of just combining files, this script also approximates missing values of the radiomap and stores the approximated data in separate folder.
- [data_rem_generation.py](Data_Analysis/data_rem_generation.py)
    - This script loads the data files created by [data_pcap_to_csv.py](Data_Analysis/data_pcap_to_csv.py) and creates radio maps in SVG and EPS format
- [data_knn.py](Data_Analysis/data_knn.py)
    - This script contains evaluation of Indoor Localisation through kNN, creation of mean and median error dependency on K and creation of histograms for selected K
- [data_helper.py](Data_Analysis/data_helper.py)
    - This python file contains constants definitions
- [common.py](Data_Analysis/common.py)
    - This script contains common functions

## Data Availability

All originally captured pcap files are part of the dataset, files from all 5 ESP32 sniffers combined into CSV files with X, Y coordinates and corresponding RSSI values, as well as all 14 used training/evaluation sets are available from Zenodo:

Data Download: [Here](https://doi.org/10.5281/zenodo.7193602)

DOI: [10.5281/zenodo.7193602](https://doi.org/10.5281/zenodo.7193602)
