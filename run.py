import streamlit as st
from video_compression.engine import infer, setup
import os
import numpy as np

st.image("assets\codec_4x_1 crop.jpg")
st.title("Neural Codec: Video Compression")
st.write('''We introduce deep learning-based compression codecs for videos.''')
st.markdown("----")

st.subheader("Configuration")
ckt_path = st.text_input("Your compressed video", placeholder="Insert the path to compressed video")
decoder_path = st.text_input("Your decoder", placeholder="Insert the path to decoder")
frame_idx = int(st.text_input("Frame no.", placeholder=16))

start = st.button("Decode")

if os.path.exists(ckt_path) and os.path.exists(decoder_path):
    st.success("Files are ready to load!")
    decoder, vid_embed = setup(ckt_path, decoder_path)
else:
    st.warning("Files are not existed!", icon="🚨")

st.markdown("----")
frame = st.empty()
if start and os.path.exists(ckt_path) and os.path.exists(decoder_path):
    while frame_idx < vid_embed.size(0):
        frame_i = infer(decoder, vid_embed, frames=[frame_idx])
        frame.image(frame_i[0].numpy())


