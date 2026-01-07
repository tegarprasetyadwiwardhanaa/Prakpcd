import streamlit as st
import cv2
import numpy as np

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
# UPLOAD IMAGE (STABIL)
# ==============================
uploaded_file = st.file_uploader(
    "Upload Gambar (Grayscale)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    new_bytes = uploaded_file.getvalue()

    # reset hanya jika file BERUBAH
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
# DISPLAY IMAGE
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
# TAMBAH NOISE (SEDANG OTOMATIS)
# ==============================
if mode == "Tambah Noise" and st.session_state.original is not None:

    noise_type = st.selectbox(
        "Jenis Noise",
        ["Gaussian", "Salt & Pepper", "Uniform", "Rayleigh", "Exponential", "Gamma"]
    )

    if st.button("Tambahkan Noise"):
        img = st.session_state.original

        # === NOISE LEVEL: SEDANG ===
        if noise_type == "Gaussian":
            st.session_state.noisy = gaussian_noise(img, sigma=35)

        elif noise_type == "Salt & Pepper":
            st.session_state.noisy = salt_pepper_noise(img, prob=0.03)

        elif noise_type == "Uniform":
            st.session_state.noisy = uniform_noise(img, -35, 35)

        elif noise_type == "Rayleigh":
            st.session_state.noisy = rayleigh_noise(img, scale=30)

        elif noise_type == "Exponential":
            st.session_state.noisy = exponential_noise(img, scale=30)

        elif noise_type == "Gamma":
            st.session_state.noisy = gamma_noise(img, scale=15)

        st.session_state.result = None
        st.success(f"Noise {noise_type} (Sedang) berhasil ditambahkan")
        st.rerun()

# ==============================
# RESTORASI
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
            "Kernel dibuat ganjil dan dibatasi hingga 15 "
            "agar terdapat piksel pusat dan citra tidak terlalu blur."
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
