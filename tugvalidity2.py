import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, detrend
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import io

