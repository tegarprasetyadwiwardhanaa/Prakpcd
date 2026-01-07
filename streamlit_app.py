import streamlit as st
import cv2
import numpy as np
import pandas as pd

from processing.noise import (
    gaussian_noise,
    rayleigh_noise,
    gamma_noise,
    salt_pepper_noise,
    exponential_noise,
    uniform_noise
)

from processing.restoration import (
    mean_filter,
    median_filter,
    wiener_filter
)

# ==============================
# CONFIG
# ==============================
st.set_page_config(
    page_title="Pengolahan Citra Digital",
    layout="wide"
)

# ==============================
# SESSION STATE INIT
# ==============================
for key in ["file_bytes", "original", "noisy", "result"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ==============================
# HEADER
# ==============================
st.title("Pengolahan Citra Digital")
st.caption("Simulasi Penambahan Noise dan Restorasi Citra")
st.divider()

# ==============================
# UPLOAD IMAGE
# ==============================
uploaded_file = st.file_uploader(
    "Upload Gambar (Grayscale)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    new_bytes = uploaded_file.getvalue()

    if st.session_state.file_bytes != new_bytes:
        st.session_state.file_bytes = new_bytes

        img_array = np.frombuffer(new_bytes, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

        if image is None:
            st.error("Gagal membaca gambar")
            st.stop()

        st.session_state.original = image
        st.session_state.noisy = None
        st.session_state.result = None

# ==============================
# MODE
# ==============================
mode = st.radio(
    "Mode Proses",
    ["Tambah Noise", "Restorasi Citra"],
    horizontal=True
)

st.divider()

# ==============================
# DISPLAY IMAGES
# ==============================
c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("Citra Asli")
    if st.session_state.original is not None:
        st.image(st.session_state.original, clamp=True)
    else:
        st.info("Belum ada gambar")

with c2:
    st.subheader("Citra Noisy")
    if st.session_state.noisy is not None:
        st.image(st.session_state.noisy, clamp=True)
    else:
        st.info("Belum ada noise")

with c3:
    st.subheader("Hasil Restorasi")
    if st.session_state.result is not None:
        st.image(st.session_state.result, clamp=True)
    else:
        st.info("Belum ada restorasi")

st.divider()

# ==============================
# TAMBAH NOISE (PRESET)
# ==============================
if mode == "Tambah Noise" and st.session_state.original is not None:

    noise_type = st.selectbox(
        "Jenis Noise",
        ["Gaussian", "Salt & Pepper", "Uniform", "Rayleigh", "Exponential", "Gamma"]
    )

    noise_level = st.selectbox(
        "Tingkat Noise",
        ["Ringan", "Sedang", "Berat"]
    )

    st.caption(
        "Tingkat noise ditentukan oleh sistem agar degradasi citra tetap realistis "
        "dan proses restorasi dapat diamati dengan jelas."
    )

    if st.button("Tambahkan Noise"):
        img = st.session_state.original

        if noise_type == "Gaussian":
            params = {"Ringan": 15, "Sedang": 35, "Berat": 60}
            st.session_state.noisy = gaussian_noise(img, sigma=params[noise_level])

        elif noise_type == "Salt & Pepper":
            params = {"Ringan": 0.01, "Sedang": 0.03, "Berat": 0.06}
            st.session_state.noisy = salt_pepper_noise(img, prob=params[noise_level])

        elif noise_type == "Uniform":
            params = {"Ringan": 15, "Sedang": 35, "Berat": 60}
            v = params[noise_level]
            st.session_state.noisy = uniform_noise(img, -v, v)

        elif noise_type == "Rayleigh":
            params = {"Ringan": 15, "Sedang": 30, "Berat": 50}
            st.session_state.noisy = rayleigh_noise(img, scale=params[noise_level])

        elif noise_type == "Exponential":
            params = {"Ringan": 15, "Sedang": 30, "Berat": 50}
            st.session_state.noisy = exponential_noise(img, scale=params[noise_level])

        elif noise_type == "Gamma":
            params = {"Ringan": 8, "Sedang": 15, "Berat": 25}
            st.session_state.noisy = gamma_noise(img, scale=params[noise_level])

        st.session_state.result = None
        st.success(f"Noise {noise_type} ({noise_level}) berhasil ditambahkan")
        st.rerun()

# ==============================
# MODE: RESTORASI
# ==============================
if mode == "Restorasi Citra":

    if st.session_state.noisy is None:
        st.warning("Tambahkan noise terlebih dahulu sebelum restorasi")
    else:
        method = st.selectbox(
            "Metode Restorasi",
            ["Mean", "Median", "Wiener"]
        )

        ksize = st.number_input(
            "Ukuran Kernel (bilangan ganjil 3–15)",
            min_value=3,
            max_value=15,
            step=2,
            value=3
        )

        st.caption(
            "Kernel berukuran ganjil digunakan agar terdapat piksel pusat "
            "dan dibatasi hingga 15 untuk mencegah citra menjadi terlalu blur."
        )

        if st.button("Lakukan Restorasi"):
            img = st.session_state.noisy

            if method == "Mean":
                st.session_state.result = mean_filter(img, ksize)
            elif method == "Median":
                st.session_state.result = median_filter(img, ksize)
            elif method == "Wiener":
                st.session_state.result = wiener_filter(img, ksize)

            st.success("Restorasi citra berhasil")
            st.rerun()

